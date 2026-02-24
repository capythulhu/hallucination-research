"""
Multi-Source Verification Pipeline for Legal Hallucination Detection

Implements three complementary verification strategies — citation lookup,
natural language inference (NLI), and regeneration consistency — and aggregates
their results using claim-type-specific weights. The weighted design reflects
the insight that different claim types are best verified by different methods:
citation claims need authoritative source lookup, factual claims benefit from
NLI against reference text, and reasoning claims are most reliably flagged by
checking whether a model can consistently reproduce the same logical step.
"""

from __future__ import annotations

import hashlib
import random
import re
import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from claim_extractor import Claim, ClaimType, SeverityLevel


class VerificationStatus(str, Enum):
    """Outcome of verifying a single claim."""

    VERIFIED = "verified"
    """Evidence supports the claim."""

    REFUTED = "refuted"
    """Evidence contradicts the claim."""

    UNVERIFIABLE = "unverifiable"
    """No evidence found — claim cannot be checked with available sources."""

    UNCERTAIN = "uncertain"
    """Evidence is ambiguous or conflicting."""


class ConfidenceLevel(str, Enum):
    """
    Traffic-light confidence indicator.

    Deliberately uses BLUE instead of GREEN to avoid implying certainty —
    even a "verified" claim might be wrong if the reference sources are
    incomplete. BLUE signals "likely correct given available evidence."
    """

    RED = "red"
    """Likely hallucination. Confidence score < 0.4."""

    YELLOW = "yellow"
    """Uncertain. Confidence score in [0.4, 0.7)."""

    BLUE = "blue"
    """Likely correct. Confidence score >= 0.7."""


class VerificationResult(BaseModel):
    """Aggregated verification outcome for a single claim."""

    claim_id: str
    status: VerificationStatus
    confidence_level: ConfidenceLevel
    confidence_score: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    verifier_results: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-verifier sub-results keyed by verifier name.",
    )
    explanation: str = ""


# ======================================================================
# Individual verifiers
# ======================================================================

# Regex for standard legal citation formats:
#   Party1 v. Party2, <volume> <reporter> <page> (<court> <year>)
LEGAL_CITATION_RE = re.compile(
    r"(?P<case_name>\w+\s+v\.\s+[^,]+),?\s*"
    r"(?P<volume>\d{1,3})\s+"
    r"(?P<reporter>U\.S\.|S\.\s*Ct\.|L\.\s*Ed\.\s*\d*[a-z]*|"
    r"F\.\d\w*|F\.\s*Supp\.\s*\d*[a-z]*)"
    r"\.?\s*"
    r"(?P<page>\d+)\s*"
    r"\((?P<court_year>[^)]+)\)"
)


class CitationVerifier:
    """
    Verify legal citation claims by checking whether the cited authority
    exists and whether the attributed holding is accurate.

    In production this would:
        1. Parse the citation into its components (case name, reporter,
           volume, page, year).
        2. Query the CourtListener API (or Westlaw/LexisNexis) for the case.
        3. Retrieve the opinion text and run NLI to check the attributed
           holding against the actual holding.

    For the demo, a hardcoded set of known-valid citations is used.
    """

    # Known-valid citations and their actual holdings (simplified).
    KNOWN_CITATIONS: dict[str, dict[str, str]] = {
        "347 U.S. 483": {
            "case_name": "Brown v. Board of Education",
            "holding": (
                "Separate educational facilities are inherently unequal, "
                "violating the Equal Protection Clause of the Fourteenth Amendment."
            ),
            "year": "1954",
        },
        "410 U.S. 113": {
            "case_name": "Roe v. Wade",
            "holding": (
                "The Constitution protects a pregnant woman's liberty to choose "
                "to have an abortion without excessive government restriction."
            ),
            "year": "1973",
        },
        "384 U.S. 436": {
            "case_name": "Miranda v. Arizona",
            "holding": (
                "Detained criminal suspects must be informed of their "
                "constitutional right to an attorney and against self-incrimination."
            ),
            "year": "1966",
        },
        "556 U.S. 662": {
            "case_name": "Ashcroft v. Iqbal",
            "holding": (
                "To survive a motion to dismiss, a complaint must contain sufficient "
                "factual matter to state a claim to relief that is plausible on its face."
            ),
            "year": "2009",
        },
        "550 U.S. 544": {
            "case_name": "Bell Atlantic Corp. v. Twombly",
            "holding": (
                "A plaintiff's obligation to state the grounds of entitlement to "
                "relief requires more than labels and conclusions; a formulaic "
                "recitation of a cause of action's elements will not do."
            ),
            "year": "2007",
        },
    }

    def verify(self, claim: Claim) -> VerificationResult:
        """
        Verify a CITATION claim.

        Returns VERIFIED if the citation exists and the attributed holding is
        consistent, REFUTED if the citation is fabricated or the holding is
        misattributed, and UNVERIFIABLE if we cannot find the citation in our
        reference set.
        """
        citation_match = LEGAL_CITATION_RE.search(claim.text)

        if not citation_match:
            return VerificationResult(
                claim_id=claim.id,
                status=VerificationStatus.UNVERIFIABLE,
                confidence_level=ConfidenceLevel.YELLOW,
                confidence_score=0.5,
                evidence=["No parseable legal citation found in claim text."],
                verifier_results={"citation": {"parsed": False}},
                explanation="Could not parse a standard legal citation from the claim.",
            )

        volume = citation_match.group("volume")
        reporter = citation_match.group("reporter").replace(" ", "")
        page = citation_match.group("page")
        lookup_key = f"{volume} {citation_match.group('reporter')} {page}"

        # Normalize whitespace in the lookup key to match our dict keys.
        normalized_key = re.sub(r"\s+", " ", lookup_key).strip()

        # Try to find the citation in our known set.
        known = None
        for key, data in self.KNOWN_CITATIONS.items():
            if key in normalized_key or normalized_key in key:
                known = data
                break

        if known is None:
            # Citation not in our reference set — could be real but unknown,
            # or could be fabricated. In production we'd query CourtListener.
            return VerificationResult(
                claim_id=claim.id,
                status=VerificationStatus.REFUTED,
                confidence_level=ConfidenceLevel.RED,
                confidence_score=0.2,
                evidence=[
                    f"Citation '{normalized_key}' not found in reference database.",
                    "In production, this would trigger a CourtListener API lookup.",
                ],
                verifier_results={
                    "citation": {
                        "parsed": True,
                        "volume": volume,
                        "reporter": reporter,
                        "page": page,
                        "found_in_reference": False,
                    }
                },
                explanation=(
                    f"The citation '{normalized_key}' was not found in the reference "
                    "database. This may indicate a fabricated citation."
                ),
            )

        # Citation exists — now check whether the attributed holding matches.
        # In production this would use NLI; here we do keyword overlap after
        # filtering stopwords, so we measure substantive semantic similarity
        # rather than matching on "the", "of", "is", etc.
        stopwords = {
            "a", "an", "the", "of", "in", "to", "for", "and", "or", "is",
            "are", "was", "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "that", "this", "it", "its", "by", "on", "at",
            "with", "from", "as", "not", "but", "if", "no", "who", "which",
        }
        claim_lower = claim.text.lower()
        holding_keywords = {
            w.strip(".,;:()") for w in known["holding"].lower().split()
        } - stopwords
        claim_keywords = {
            w.strip(".,;:()") for w in claim_lower.split()
        } - stopwords
        keyword_overlap = len(holding_keywords & claim_keywords) / max(
            len(holding_keywords), 1
        )

        if keyword_overlap > 0.20:
            return VerificationResult(
                claim_id=claim.id,
                status=VerificationStatus.VERIFIED,
                confidence_level=ConfidenceLevel.BLUE,
                confidence_score=0.85,
                evidence=[
                    f"Citation found: {known['case_name']} ({known['year']}).",
                    f"Known holding: {known['holding']}",
                    f"Keyword overlap with claim: {keyword_overlap:.0%}.",
                ],
                verifier_results={
                    "citation": {
                        "parsed": True,
                        "found_in_reference": True,
                        "holding_match": keyword_overlap,
                    }
                },
                explanation=(
                    f"The citation refers to {known['case_name']} and the attributed "
                    "proposition is broadly consistent with the actual holding."
                ),
            )
        else:
            return VerificationResult(
                claim_id=claim.id,
                status=VerificationStatus.REFUTED,
                confidence_level=ConfidenceLevel.RED,
                confidence_score=0.25,
                evidence=[
                    f"Citation found: {known['case_name']} ({known['year']}).",
                    f"Known holding: {known['holding']}",
                    f"Attributed holding does NOT match (overlap: {keyword_overlap:.0%}).",
                ],
                verifier_results={
                    "citation": {
                        "parsed": True,
                        "found_in_reference": True,
                        "holding_match": keyword_overlap,
                    }
                },
                explanation=(
                    f"The citation to {known['case_name']} exists, but the holding "
                    "attributed in this claim does not match the actual holding."
                ),
            )


# Prompt used by NLIVerifier — included as a constant for transparency.
NLI_PROMPT = """\
You are a legal reasoning assistant. Given a CLAIM and a CONTEXT passage, \
determine whether the context ENTAILS, CONTRADICTS, or is NEUTRAL with \
respect to the claim.

CLAIM: {claim_text}

CONTEXT: {context_text}

Respond with exactly one of: ENTAILMENT, CONTRADICTION, NEUTRAL.
Then provide a one-sentence explanation.
"""


class NLIVerifier:
    """
    Natural Language Inference verifier.

    Checks whether a claim is entailed by, contradicted by, or neutral with
    respect to a reference knowledge base. In production this would retrieve
    passages from a vector store and run an LLM-based NLI check. The demo
    uses a small built-in reference corpus of known facts that allows us to
    detect specific misstatements.
    """

    # Reference facts the NLI verifier "knows" (simulating a retrieval step).
    # Each entry: (trigger_phrase, reference_text, entails: bool).
    REFERENCE_FACTS: list[tuple[str, str, bool]] = [
        (
            "brown v. board of education",
            "Brown v. Board of Education addressed racial segregation in public "
            "schools, holding that separate educational facilities are inherently "
            "unequal. It did not address public employee speech or whistleblower "
            "protections.",
            False,  # Any claim attributing speech/whistleblower holdings is wrong.
        ),
        (
            "ohio rev. code section 4113.52",
            "Ohio's Whistleblower Protection Act (Ohio Rev. Code Section 4113.52) "
            "was enacted on January 17, 1995.",
            True,  # The statute exists, but the date may differ.
        ),
        (
            "enacted on april 15, 1992",
            "Ohio Rev. Code Section 4113.52 was enacted on January 17, 1995, "
            "not April 15, 1992.",
            False,  # Wrong date.
        ),
        (
            "ashcroft v. iqbal",
            "Ashcroft v. Iqbal, 556 U.S. 662 (2009), established the plausibility "
            "standard for evaluating motions to dismiss under Rule 12(b)(6).",
            True,
        ),
        (
            "daubert v. merrell dow",
            "Daubert v. Merrell Dow Pharmaceuticals, Inc., 509 U.S. 579 (1993), "
            "established the standard for admissibility of expert testimony.",
            True,
        ),
    ]

    def verify(self, claim: Claim, context: str) -> VerificationResult:
        """
        Run NLI between *claim* and reference knowledge.

        In production:
            1. Retrieve top-k passages from a vector store relevant to the claim.
            2. For each passage, call the LLM with NLI_PROMPT.
            3. Aggregate across passages (majority vote).

        Demo: match claim against built-in reference facts.
        """
        claim_lower = claim.text.lower()

        # Search reference corpus for relevant facts.
        matched_refs: list[tuple[str, bool]] = []
        for trigger, ref_text, entails in self.REFERENCE_FACTS:
            if trigger in claim_lower:
                matched_refs.append((ref_text, entails))

        if not matched_refs:
            # No reference found — return neutral / uncertain.
            return VerificationResult(
                claim_id=claim.id,
                status=VerificationStatus.UNCERTAIN,
                confidence_level=ConfidenceLevel.YELLOW,
                confidence_score=0.5,
                evidence=["No matching reference found in NLI knowledge base."],
                verifier_results={"nli": {"label": "NEUTRAL", "score": 0.5}},
                explanation="NLI could not find relevant reference passages for this claim.",
            )

        # If any matched reference contradicts the claim, flag it.
        contradictions = [
            (ref, ent) for ref, ent in matched_refs if not ent
        ]
        supports = [(ref, ent) for ref, ent in matched_refs if ent]

        if contradictions:
            score = 0.15
            label = "CONTRADICTION"
            status = VerificationStatus.REFUTED
            evidence_texts = [
                f"NLI label: CONTRADICTION",
                f"Reference: {contradictions[0][0]}",
                "The claim conflicts with known reference material.",
            ]
        elif supports:
            score = 0.85
            label = "ENTAILMENT"
            status = VerificationStatus.VERIFIED
            evidence_texts = [
                f"NLI label: ENTAILMENT",
                f"Reference: {supports[0][0]}",
                "The claim is consistent with known reference material.",
            ]
        else:
            score = 0.5
            label = "NEUTRAL"
            status = VerificationStatus.UNCERTAIN
            evidence_texts = [
                f"NLI label: NEUTRAL",
                "Reference material is ambiguous.",
            ]

        return VerificationResult(
            claim_id=claim.id,
            status=status,
            confidence_level=self._score_to_level(score),
            confidence_score=round(score, 3),
            evidence=evidence_texts,
            verifier_results={"nli": {"label": label, "score": score}},
            explanation=f"NLI analysis yielded {label} against reference knowledge base.",
        )

    @staticmethod
    def _score_to_level(score: float) -> ConfidenceLevel:
        if score < 0.4:
            return ConfidenceLevel.RED
        elif score < 0.7:
            return ConfidenceLevel.YELLOW
        return ConfidenceLevel.BLUE


class ConsistencyVerifier:
    """
    Regeneration-consistency verifier.

    Tests whether an LLM can consistently reproduce the same claim across
    multiple independent regenerations. Claims that the model itself cannot
    reliably reproduce are more likely to be hallucinations.

    In production:
        1. Isolate the paragraph containing the claim.
        2. Prompt the LLM N times to regenerate that paragraph from scratch.
        3. Compute BERTScore between the original paragraph and each
           regeneration.
        4. Flag claims embedded in paragraphs with high variance.

    Demo: simulates BERTScore outputs with controlled randomness.
    """

    def verify(
        self, claim: Claim, document: str, n_samples: int = 3
    ) -> VerificationResult:
        """
        Check regeneration consistency for *claim* within *document*.

        Returns a VerificationResult where the confidence score reflects how
        consistently the model regenerates the section containing this claim.
        """
        # Locate the approximate paragraph containing the claim.
        claim_start = claim.source_span.get("start", 0)
        paragraph_start = document.rfind("\n\n", 0, claim_start)
        paragraph_start = paragraph_start + 2 if paragraph_start != -1 else 0
        paragraph_end = document.find("\n\n", claim_start)
        paragraph_end = paragraph_end if paragraph_end != -1 else len(document)
        original_paragraph = document[paragraph_start:paragraph_end]

        # In production, we'd call the LLM n_samples times and compute
        # BERTScore (precision, recall, F1) between the original and each
        # regeneration. Here we simulate the scores.
        #
        # The simulation is seeded by a hash of the claim text so results
        # are deterministic for the same claim across runs.
        #
        # Key insight: hallucinated claims containing fabricated specifics
        # (precise but invented numbers, dates, case names) tend to produce
        # high variance across regenerations because the model picks different
        # specifics each time. We model this by lowering the base score and
        # increasing noise for claims with "suspicious specificity" markers.
        seed = int(hashlib.md5(claim.text.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)

        # Detect suspicious specificity: fabricated claims often contain very
        # precise numbers (e.g., 78.3%), specific dates, or unusual case names
        # that a model would vary across regenerations.
        suspicious_patterns = [
            r"\d+\.\d+%",           # Suspiciously precise percentage
            r"enacted on \w+ \d+",  # Specific enactment date
            r"necessarily extend",  # Overly strong logical leap
        ]
        suspicion_level = sum(
            1 for pat in suspicious_patterns
            if re.search(pat, claim.text, re.IGNORECASE)
        )

        # Adjust base score and noise based on suspicion.
        # Higher suspicion -> lower base (model can't reproduce specifics
        # consistently) and higher noise (more variance across regenerations).
        if suspicion_level > 0:
            base = max(0.30, 0.70 - 0.25 * suspicion_level)
            noise_std = 0.20 + 0.06 * suspicion_level
        else:
            base = 0.78
            noise_std = 0.10

        bertscore_f1s: list[float] = []
        for _ in range(n_samples):
            noise = rng.gauss(0, noise_std)
            f1 = max(0.0, min(1.0, base + noise))
            bertscore_f1s.append(round(f1, 3))

        mean_f1 = sum(bertscore_f1s) / len(bertscore_f1s)
        variance = sum((f - mean_f1) ** 2 for f in bertscore_f1s) / len(bertscore_f1s)

        # High mean + low variance = consistent = likely not hallucinated.
        # Low mean or high variance = inconsistent = suspicious.
        consistency_score = mean_f1 * (1 - min(variance * 5, 0.5))
        consistency_score = round(max(0.0, min(1.0, consistency_score)), 3)

        if consistency_score >= 0.7:
            status = VerificationStatus.VERIFIED
            level = ConfidenceLevel.BLUE
        elif consistency_score >= 0.4:
            status = VerificationStatus.UNCERTAIN
            level = ConfidenceLevel.YELLOW
        else:
            status = VerificationStatus.REFUTED
            level = ConfidenceLevel.RED

        return VerificationResult(
            claim_id=claim.id,
            status=status,
            confidence_level=level,
            confidence_score=consistency_score,
            evidence=[
                f"BERTScore F1 across {n_samples} regenerations: {bertscore_f1s}",
                f"Mean F1: {mean_f1:.3f}, Variance: {variance:.4f}",
                f"Consistency score: {consistency_score}",
            ],
            verifier_results={
                "consistency": {
                    "bertscore_f1s": bertscore_f1s,
                    "mean_f1": mean_f1,
                    "variance": variance,
                    "consistency_score": consistency_score,
                }
            },
            explanation=(
                f"Regeneration consistency score: {consistency_score:.2f}. "
                f"{'Claim is consistently reproducible.' if consistency_score >= 0.7 else 'Claim shows variability across regenerations — possible hallucination.'}"
            ),
        )



# ======================================================================
# Aggregation pipeline
# ======================================================================

# Type-specific verifier weights.
# Each tuple: (citation_weight, nli_weight, consistency_weight)
_TYPE_WEIGHTS: dict[ClaimType, tuple[float, float, float]] = {
    ClaimType.CITATION: (0.6, 0.3, 0.1),
    ClaimType.FACTUAL: (0.1, 0.5, 0.4),
    ClaimType.REASONING: (0.0, 0.4, 0.6),
    ClaimType.STATISTICAL: (0.2, 0.4, 0.4),
}


class VerificationPipeline:
    """
    Orchestrates multi-source verification of extracted claims.

    Each claim is routed through applicable verifiers based on its type, and
    the individual verifier scores are aggregated using type-specific weights.
    """

    def __init__(self) -> None:
        self.citation_verifier = CitationVerifier()
        self.nli_verifier = NLIVerifier()
        self.consistency_verifier = ConsistencyVerifier()

    def verify_claim(
        self, claim: Claim, document: str, context: str = ""
    ) -> VerificationResult:
        """
        Verify a single claim using all applicable verifiers and aggregate.
        """
        citation_result = self.citation_verifier.verify(claim)
        nli_result = self.nli_verifier.verify(claim, context)
        consistency_result = self.consistency_verifier.verify(claim, document)

        return self._aggregate_results(
            claim, citation_result, nli_result, consistency_result
        )

    def verify_all(
        self,
        claims: list[Claim],
        document: str,
        context: str = "",
    ) -> list[VerificationResult]:
        """
        Verify all claims and return results with timing metadata.

        In production, claims could be verified in parallel with asyncio.
        """
        start = time.time()
        results: list[VerificationResult] = []

        for claim in claims:
            result = self.verify_claim(claim, document, context)
            results.append(result)

        elapsed = time.time() - start

        # Attach pipeline-level timing to the last result's metadata (or log it).
        if results:
            results[-1].verifier_results["_pipeline"] = {
                "total_claims": len(claims),
                "total_seconds": round(elapsed, 3),
            }

        return results

    def _aggregate_results(
        self,
        claim: Claim,
        citation_result: VerificationResult,
        nli_result: VerificationResult,
        consistency_result: VerificationResult,
    ) -> VerificationResult:
        """
        Compute a weighted confidence score from individual verifier results.
        """
        cw, nw, kw = _TYPE_WEIGHTS.get(
            claim.claim_type, (0.33, 0.34, 0.33)
        )

        weighted_score = (
            cw * citation_result.confidence_score
            + nw * nli_result.confidence_score
            + kw * consistency_result.confidence_score
        )
        weighted_score = round(max(0.0, min(1.0, weighted_score)), 3)

        # Determine overall status from weighted score.
        if weighted_score >= 0.7:
            status = VerificationStatus.VERIFIED
        elif weighted_score >= 0.4:
            status = VerificationStatus.UNCERTAIN
        else:
            status = VerificationStatus.REFUTED

        confidence_level = self._assign_confidence_level(weighted_score)

        # Merge evidence from all verifiers.
        all_evidence = (
            citation_result.evidence
            + nli_result.evidence
            + consistency_result.evidence
        )

        return VerificationResult(
            claim_id=claim.id,
            status=status,
            confidence_level=confidence_level,
            confidence_score=weighted_score,
            evidence=all_evidence,
            verifier_results={
                "citation": citation_result.verifier_results.get("citation", {}),
                "nli": nli_result.verifier_results.get("nli", {}),
                "consistency": consistency_result.verifier_results.get(
                    "consistency", {}
                ),
                "weights": {"citation": cw, "nli": nw, "consistency": kw},
            },
            explanation=(
                f"Weighted score: {weighted_score:.3f} "
                f"(citation={citation_result.confidence_score:.2f}*{cw}, "
                f"nli={nli_result.confidence_score:.2f}*{nw}, "
                f"consistency={consistency_result.confidence_score:.2f}*{kw}). "
                f"Status: {status.value}."
            ),
        )

    @staticmethod
    def _assign_confidence_level(score: float) -> ConfidenceLevel:
        """
        Map a continuous confidence score to a discrete level.

        RED    : score < 0.4  — likely hallucination
        YELLOW : 0.4 <= score < 0.7  — uncertain
        BLUE   : score >= 0.7  — likely correct
        """
        if score < 0.4:
            return ConfidenceLevel.RED
        elif score < 0.7:
            return ConfidenceLevel.YELLOW
        return ConfidenceLevel.BLUE
