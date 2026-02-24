# Detection Pipeline — Detailed Data Flow

The following diagram traces the complete data flow from raw legal document input through claim extraction, type-based routing, verification, and final annotated output. Each claim type follows a purpose-built verification path with primary and secondary checks. Confidence scores are mapped to a three-tier system: RED (likely hallucination), YELLOW (uncertain), and BLUE (likely accurate).

```mermaid
flowchart TB
    %% Step 1 — Document Input & Chunking
    A["Raw Legal Document (5-20 pages)"] --> B["Chunker 3000 char chunks 500 char overlap"]
    B --> C1["Chunk 1"]
    B --> C2["Chunk 2"]
    B --> CN["Chunk N"]

    %% Step 2 — Claim Extraction
    C1 --> D["LLM Claim Extractor"]
    C2 --> D
    CN --> D
    D --> E["Typed Claim Objects (text, type, source_span, chunk_id, raw_context)"]

    %% Step 3 — Deduplication
    E --> F["Deduplication (cross-chunk boundary semantic matching)"]
    F --> G["Unique Claims Set"]

    %% Step 4 — Claim Router
    G --> H{"Claim Router (by ClaimType)"}

    %% CITATION path
    H -->|"CITATION"| CIT1["Citation Parser (extract case name, volume, reporter, page)"]
    CIT1 --> CIT2["CourtListener DB Lookup"]
    CIT2 --> CIT3{"Case Found?"}
    CIT3 -->|"Yes"| CIT4["NLI Entailment vs. Actual Holding"]
    CIT3 -->|"No"| CIT5["Flag: Fabricated Citation (score = 0.0)"]
    CIT4 --> CIT6["Citation Verification Result"]
    CIT5 --> CIT6

    %% FACTUAL path
    H -->|"FACTUAL"| FACT1["RAG Retrieval (knowledge base + source documents)"]
    FACT1 --> FACT2["NLI Entailment Check"]
    FACT2 --> FACT3{"Entailment Score > 0.5?"}
    FACT3 -->|"Yes"| FACT4["Accept with Entailment Score"]
    FACT3 -->|"No / Borderline"| FACT5["Self-Consistency Check (N=5)"]
    FACT5 --> FACT4
    FACT4 --> FACT6["Factual Verification Result"]

    %% REASONING path
    H -->|"REASONING"| REAS1["Self-Consistency Check (Primary) N=5 Regenerations"]
    REAS1 --> REAS2["BERTScore Variance Analysis"]
    REAS2 --> REAS3{"Variance < Threshold?"}
    REAS3 -->|"Consistent"| REAS4["NLI Entailment (Secondary) vs. Source Premises"]
    REAS3 -->|"Inconsistent"| REAS5["Flag: Unstable Reasoning"]
    REAS4 --> REAS6["Reasoning Verification Result"]
    REAS5 --> REAS6

    %% STATISTICAL path
    H -->|"STATISTICAL"| STAT1["Source Retrieval (authoritative data repositories)"]
    STAT1 --> STAT2["NLI Entailment vs. Source Data"]
    STAT2 --> STAT3["Self-Consistency Check (N=3)"]
    STAT3 --> STAT4["Statistical Verification Result"]

    %% Step 9 — Weighted Aggregation
    CIT6 --> AGG["Weighted Aggregation (weights by claim type)"]
    FACT6 --> AGG
    REAS6 --> AGG
    STAT4 --> AGG

    AGG --> SCORE["Confidence Score (0.0 to 1.0)"]

    SCORE --> TIER{"Confidence Mapping"}
    TIER -->|"score < 0.4"| RED["RED Likely Hallucination"]
    TIER -->|"0.4 <= score < 0.7"| YELLOW["YELLOW Uncertain — Needs Review"]
    TIER -->|"score >= 0.7"| BLUE["BLUE Likely Accurate"]

    %% Step 10 — Output
    RED --> OUT["Annotated Document Output"]
    YELLOW --> OUT
    BLUE --> OUT

    OUT --> OUT1["Flagged Claims with Inline Annotations"]
    OUT --> OUT2["Evidence Trails (sources, scores, reasoning)"]
    OUT --> OUT3["Severity Ranking (ordered by confidence ascending)"]

    %% Styles
    style A fill:#e0e0e0,stroke:#666,color:#333
    style RED fill:#ff6666,stroke:#cc0000,color:#fff
    style YELLOW fill:#ffcc00,stroke:#cc9900,color:#333
    style BLUE fill:#6699ff,stroke:#3366cc,color:#fff
    style CIT5 fill:#ff9999,stroke:#cc0000,color:#333
    style REAS5 fill:#ff9999,stroke:#cc0000,color:#333
    style OUT fill:#d4edda,stroke:#28a745,color:#333
    style OUT1 fill:#f8f9fa,stroke:#666,color:#333
    style OUT2 fill:#f8f9fa,stroke:#666,color:#333
    style OUT3 fill:#f8f9fa,stroke:#666,color:#333
```
