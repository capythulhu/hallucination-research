"""
Claim Extraction Module for Legal Hallucination Detection

Extracts typed, atomic claims from long-form legal documents using overlapping
chunk-based processing and LLM-powered analysis. Each claim is classified by
type (citation, factual, reasoning, statistical) and severity level, enabling
downstream verification pipelines to apply type-specific verification strategies.
"""

from __future__ import annotations

import hashlib
import re
import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ClaimType(str, Enum):
    """Classification of verifiable claim types found in legal documents."""

    CITATION = "citation"
    """Reference to a case, statute, regulation, or other legal authority."""

    FACTUAL = "factual"
    """Assertion about a real-world event, date, name, or procedural history."""

    REASONING = "reasoning"
    """Logical inference or legal argument connecting premises to a conclusion."""

    STATISTICAL = "statistical"
    """Quantitative assertion — percentages, counts, rates, or numerical trends."""


class SeverityLevel(str, Enum):
    """
    How damaging a hallucination of this claim would be in a legal context.

    Severity is assessed from the perspective of a practicing attorney who would
    rely on this text. A fabricated case citation that a court cannot locate is
    far more harmful than an imprecise statistic in a footnote.
    """

    CRITICAL = "critical"
    """Could result in sanctions, malpractice liability, or case dismissal."""

    HIGH = "high"
    """Materially misleading; would undermine credibility if discovered."""

    MEDIUM = "medium"
    """Incorrect but unlikely to be dispositive; still professionally embarrassing."""

    LOW = "low"
    """Minor inaccuracy with negligible practical impact."""


class Claim(BaseModel):
    """A single atomic, verifiable claim extracted from a legal document."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    text: str = Field(description="The verbatim or minimally paraphrased claim text.")
    claim_type: ClaimType
    severity: SeverityLevel
    source_span: dict[str, int] = Field(
        description="Character offsets in the original document: {'start': int, 'end': int}."
    )
    entities: list[str] = Field(
        default_factory=list,
        description="Named entities referenced (case names, statutes, parties, courts).",
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="IDs of other claims this claim logically depends on.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value pairs — e.g., chunk_index, extraction_model.",
    )


class ExtractionResult(BaseModel):
    """Aggregated output of the claim extraction pipeline for one document."""

    claims: list[Claim]
    document_id: str
    extraction_time_seconds: float
    chunk_count: int


# ---------------------------------------------------------------------------
# LLM prompt — the core intellectual property of the extraction step.
# ---------------------------------------------------------------------------

CLAIM_EXTRACTION_PROMPT = """\
You are a legal document analyst specializing in the identification and \
classification of verifiable claims within legal briefs, memoranda, and \
judicial opinions. Your task is to decompose the following text chunk into \
a list of atomic, independently verifiable claims.

## Claim Taxonomy

Classify every claim into exactly one of the following types:

1. **CITATION** — Any reference to a legal authority: case law (with or \
without a reporter citation), statutes (e.g., 42 U.S.C. Section 1983), \
regulations (e.g., 29 C.F.R. Part 1630), treatises, or restatements. \
Include the full citation string AND the proposition the author attributes \
to that authority. A citation claim is only as good as the pairing of \
"source + attributed holding/rule."

2. **FACTUAL** — An assertion about the real world that can be checked \
against public records, legislative history, procedural dockets, or \
widely-accepted reference material. Examples: dates of enactment, names \
of judges on a panel, procedural posture of a case, historical events \
referenced in argument.

3. **REASONING** — A logical inference where the author draws a conclusion \
from stated premises. Extract both the premises and the conclusion so that \
the reasoning step itself can be evaluated for validity. Pay special \
attention to: (a) analogical reasoning ("like in X, here too..."), \
(b) distinguishing ("unlike X, the present case..."), and (c) policy \
arguments ("the purpose of the statute is best served by...").

4. **STATISTICAL** — Any numerical assertion: percentages, counts, rates, \
dollar amounts tied to empirical claims, or quantitative trends. Include \
the source attribution if one is given.

## Severity Assessment

For each claim, assess the severity of a potential hallucination:

- **CRITICAL**: The claim, if fabricated, could lead to judicial sanctions \
(e.g., Rule 11), malpractice liability, or case dismissal. Typical \
examples: fabricated case citations, invented statutory provisions, \
misattributed holdings of controlling authority.

- **HIGH**: The claim is materially important to the argument. An error \
would undermine the brief's credibility and could mislead the court, \
even if it might not independently warrant sanctions.

- **MEDIUM**: The claim supports but is not central to the argument. An \
error would be professionally embarrassing but unlikely to change the \
outcome.

- **LOW**: Minor or tangential assertion. Useful for completeness but a \
mistake here has negligible practical impact.

## Entity Extraction

For every claim, extract all named entities: case names (both parties), \
court names, judge names, statute numbers, regulatory section numbers, \
party names, and any organization referenced.

## Output Format

Return a JSON array where each element has the following structure:

```json
{
  "text": "<verbatim or closely paraphrased claim>",
  "claim_type": "CITATION | FACTUAL | REASONING | STATISTICAL",
  "severity": "CRITICAL | HIGH | MEDIUM | LOW",
  "entities": ["entity1", "entity2"],
  "approximate_offset": <integer character offset from chunk start>
}
```

## Guidelines

- Break compound sentences into separate claims when they contain \
independently verifiable parts. "The Court held X and noted Y" should \
become two claims if X and Y are distinct propositions.
- Do NOT extract purely procedural or structural text ("Respondent \
respectfully submits this brief") — only substantive claims.
- When a claim attributes a holding to a case, the CITATION claim should \
include both the citation AND the attributed holding, so downstream \
verification can check both existence and accuracy.
- Err on the side of extracting too many claims rather than too few. \
Downstream deduplication will handle overlaps.

## Text Chunk

{chunk_text}
"""


class ClaimExtractor:
    """
    Extracts typed, atomic claims from legal documents.

    The extraction process:
    1. Split the document into overlapping chunks to stay within LLM context
       limits while preserving cross-boundary context.
    2. Send each chunk to an LLM with a domain-aware extraction prompt.
    3. Deduplicate claims that appear in the overlap region of adjacent chunks.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        chunk_size: int = 3000,
        chunk_overlap: int = 500,
    ) -> None:
        self.model = model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def chunk_document(self, text: str) -> list[str]:
        """
        Split *text* into overlapping chunks of approximately *chunk_size*
        characters, with *chunk_overlap* characters shared between consecutive
        chunks. Splits are made at paragraph or sentence boundaries when
        possible to avoid cutting mid-sentence.
        """
        if len(text) <= self.chunk_size:
            return [text]

        chunks: list[str] = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # If we're not at the very end, try to break at a paragraph boundary.
            if end < len(text):
                # Look for a double-newline (paragraph break) near the end.
                paragraph_break = text.rfind("\n\n", start + self.chunk_size // 2, end)
                if paragraph_break != -1:
                    end = paragraph_break + 2  # include the newline chars
                else:
                    # Fall back to sentence boundary (period followed by space).
                    sentence_break = text.rfind(". ", start + self.chunk_size // 2, end)
                    if sentence_break != -1:
                        end = sentence_break + 2

            chunks.append(text[start:end])
            start = end - self.chunk_overlap

        return chunks

    # ------------------------------------------------------------------
    # LLM interaction (mock for demo; production would call Anthropic API)
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str, chunk: str) -> list[dict]:
        """
        Send the extraction prompt + chunk to the LLM and parse the response.

        In production, this method would:
            1. Call the Anthropic Messages API with self.model.
            2. Parse the JSON array from the assistant response.
            3. Validate each element against the expected schema.
            4. Retry with exponential backoff on transient failures.

        For this demo, we use lightweight heuristics to produce plausible
        mock extractions so the full pipeline can run without API keys.
        """
        claims: list[dict] = []

        def _is_sentence_boundary(text: str, dot_pos: int) -> bool:
            """Check if the period at *dot_pos* is a true sentence boundary.

            Returns False for abbreviations common in legal writing (e.g.,
            'v.', 'U.S.', 'Corp.', 'Inc.', 'Ct.', 'Ed.', 'No.', 'Sec.').
            """
            if dot_pos + 1 >= len(text) or text[dot_pos + 1] not in " \n\t":
                return False

            # Extract the word ending at this period.
            word_start = dot_pos - 1
            while word_start >= 0 and text[word_start] not in " \n\t":
                word_start -= 1
            word = text[word_start + 1 : dot_pos + 1].lower()

            # Common legal abbreviations that should NOT end a sentence.
            non_boundary_words = {
                "v.", "u.s.", "s.", "f.", "ct.", "ed.", "corp.", "inc.",
                "ltd.", "no.", "sec.", "supp.", "rev.", "app.", "cir.",
                "dept.", "dist.", "div.", "gov.", "jr.", "sr.", "dr.",
                "mr.", "mrs.", "ms.", "st.", "ave.", "blvd.",
            }

            # Also reject if the period is part of a multi-period abbreviation
            # like "U.S." or "F.3d" — check if there's another period nearby.
            if word in non_boundary_words:
                return False

            # Reject single-letter abbreviations (e.g., middle initials).
            if len(word) == 2 and word[0].isalpha():
                return False

            # Reject if the next non-space character is lowercase (likely
            # a continuation, not a new sentence).
            next_char_pos = dot_pos + 1
            while next_char_pos < len(text) and text[next_char_pos] in " \t":
                next_char_pos += 1
            if next_char_pos < len(text) and text[next_char_pos].islower():
                return False

            return True

        def _find_sentence_bounds(text: str, anchor: int) -> tuple[int, int]:
            """Find the sentence containing *anchor*, respecting legal abbreviations."""
            # Search backward for a true sentence boundary.
            start = 0
            search_pos = anchor - 1
            while search_pos > 0:
                if text[search_pos] == "." and _is_sentence_boundary(text, search_pos):
                    start = search_pos + 1
                    # Skip whitespace after the period.
                    while start < len(text) and text[start] in " \t\n":
                        start += 1
                    break
                search_pos -= 1

            # Search forward for a true sentence boundary.
            end = len(text)
            search_pos = anchor
            while search_pos < len(text) - 1:
                if text[search_pos] == "." and _is_sentence_boundary(text, search_pos):
                    end = search_pos + 1
                    break
                search_pos += 1

            return start, end

        # --- Heuristic 1: Legal citation pattern ---
        # Matches patterns like "Name v. Name, 123 F.3d 456 (Cir. 2005)"
        # Uses \w+ before "v." (no dots in the class) to avoid the greedy
        # character class consuming the literal "v." token. After "v.\s+",
        # we allow anything up to the comma preceding the volume number.
        citation_pattern = re.compile(
            r"(\w+\s+v\.\s+.+?,\s*"
            r"\d+\s+(?:U\.S\.|F\.\d\w*|S\.\s*Ct\.|L\.\s*Ed)"
            r"\.?\s*\d+\s*\([^)]*\d{4}\))"
        )
        for match in citation_pattern.finditer(chunk):
            # Grab the full sentence containing the citation for context.
            sent_start, sent_end = _find_sentence_bounds(chunk, match.start())
            sentence = chunk[sent_start:sent_end].strip()

            claims.append(
                {
                    "text": sentence,
                    "claim_type": "CITATION",
                    "severity": "CRITICAL",
                    "entities": [match.group(0).strip()],
                    "approximate_offset": match.start(),
                }
            )

        # --- Heuristic 2: Statutory references ---
        statute_pattern = re.compile(
            r"(\d+\s+U\.S\.C\.\s*(?:§|Section|Sec\.)\s*\d+[a-z]*(?:\([a-z0-9]+\))*)"
        )
        for match in statute_pattern.finditer(chunk):
            sent_start, sent_end = _find_sentence_bounds(chunk, match.start())
            sentence = chunk[sent_start:sent_end].strip()

            claims.append(
                {
                    "text": sentence,
                    "claim_type": "FACTUAL",
                    "severity": "HIGH",
                    "entities": [match.group(0).strip()],
                    "approximate_offset": match.start(),
                }
            )

        # --- Heuristic 3: Numerical / statistical assertions ---
        stat_pattern = re.compile(
            r"(?:approximately|roughly|nearly|over|more than|less than|about)?"
            r"\s*\d+(?:\.\d+)?%"
        )
        for match in stat_pattern.finditer(chunk):
            sent_start, sent_end = _find_sentence_bounds(chunk, match.start())
            sentence = chunk[sent_start:sent_end].strip()

            claims.append(
                {
                    "text": sentence,
                    "claim_type": "STATISTICAL",
                    "severity": "LOW",
                    "entities": [],
                    "approximate_offset": match.start(),
                }
            )

        # --- Heuristic 4: Reasoning indicators ---
        reasoning_indicators = [
            "therefore", "thus", "accordingly", "it follows that",
            "consequently", "this demonstrates", "this establishes",
            "the logical conclusion", "necessarily means",
        ]
        for indicator in reasoning_indicators:
            idx = chunk.lower().find(indicator)
            if idx != -1:
                sent_start, sent_end = _find_sentence_bounds(chunk, idx)
                sentence = chunk[sent_start:sent_end].strip()

                claims.append(
                    {
                        "text": sentence,
                        "claim_type": "REASONING",
                        "severity": "MEDIUM",
                        "entities": [],
                        "approximate_offset": idx,
                    }
                )

        # --- Heuristic 5: Date-based factual claims ---
        date_pattern = re.compile(
            r"(?:enacted|signed|passed|effective|ratified|amended)\s+(?:in|on)\s+"
            r"(?:January|February|March|April|May|June|July|August|September|"
            r"October|November|December)?\s*\d{1,2}?,?\s*\d{4}"
        )
        for match in date_pattern.finditer(chunk):
            sent_start, sent_end = _find_sentence_bounds(chunk, match.start())
            sentence = chunk[sent_start:sent_end].strip()

            claims.append(
                {
                    "text": sentence,
                    "claim_type": "FACTUAL",
                    "severity": "HIGH",
                    "entities": [],
                    "approximate_offset": match.start(),
                }
            )

        return claims

    # ------------------------------------------------------------------
    # Extraction from a single chunk
    # ------------------------------------------------------------------

    def extract_claims_from_chunk(
        self, chunk: str, chunk_index: int
    ) -> list[Claim]:
        """
        Extract claims from a single text chunk.

        Returns a list of :class:`Claim` objects with source spans relativized
        to the original chunk and metadata recording the chunk index.
        """
        prompt = CLAIM_EXTRACTION_PROMPT.replace("{chunk_text}", chunk)
        raw_claims = self._call_llm(prompt, chunk)

        claims: list[Claim] = []
        for raw in raw_claims:
            offset = raw.get("approximate_offset", 0)
            claim = Claim(
                text=raw["text"],
                claim_type=ClaimType(raw["claim_type"].lower()),
                severity=SeverityLevel(raw["severity"].lower()),
                source_span={"start": offset, "end": offset + len(raw["text"])},
                entities=raw.get("entities", []),
                metadata={"chunk_index": chunk_index, "model": self.model},
            )
            claims.append(claim)

        return claims

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def deduplicate(self, claims: list[Claim]) -> list[Claim]:
        """
        Remove near-duplicate claims that arise from overlapping chunks.

        Two claims are considered duplicates if their text shares more than 80%
        of the same 4-gram shingles. When duplicates are found, we keep the one
        from the earlier chunk (which typically has more surrounding context).
        """
        def _shingle_set(text: str, n: int = 4) -> set[str]:
            tokens = text.lower().split()
            return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}

        def _jaccard(a: set[str], b: set[str]) -> float:
            if not a or not b:
                return 0.0
            return len(a & b) / len(a | b)

        deduplicated: list[Claim] = []
        seen_shingles: list[tuple[set[str], Claim]] = []

        for claim in claims:
            shingles = _shingle_set(claim.text)
            is_dup = False
            for existing_shingles, _ in seen_shingles:
                if _jaccard(shingles, existing_shingles) > 0.80:
                    is_dup = True
                    break

            if not is_dup:
                deduplicated.append(claim)
                seen_shingles.append((shingles, claim))

        return deduplicated

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def extract(
        self, document: str, document_id: str | None = None
    ) -> ExtractionResult:
        """
        Run the full claim extraction pipeline on a legal document.

        Steps:
            1. Chunk the document with overlap.
            2. Extract claims from each chunk via LLM.
            3. Adjust source spans to document-level offsets.
            4. Deduplicate across chunk boundaries.

        Returns an :class:`ExtractionResult` with all extracted claims and
        pipeline metadata.
        """
        if document_id is None:
            document_id = hashlib.sha256(document.encode()).hexdigest()[:16]

        start_time = time.time()
        chunks = self.chunk_document(document)

        all_claims: list[Claim] = []
        cumulative_offset = 0

        for i, chunk in enumerate(chunks):
            chunk_claims = self.extract_claims_from_chunk(chunk, chunk_index=i)

            # Adjust source spans from chunk-relative to document-relative.
            chunk_start_in_doc = document.find(chunk[:80])
            if chunk_start_in_doc == -1:
                chunk_start_in_doc = cumulative_offset

            for claim in chunk_claims:
                claim.source_span["start"] += chunk_start_in_doc
                claim.source_span["end"] += chunk_start_in_doc

            all_claims.extend(chunk_claims)
            cumulative_offset += len(chunk) - self.chunk_overlap

        deduplicated = self.deduplicate(all_claims)
        elapsed = time.time() - start_time

        return ExtractionResult(
            claims=deduplicated,
            document_id=document_id,
            extraction_time_seconds=round(elapsed, 3),
            chunk_count=len(chunks),
        )
