# Architecture Overview — Multi-Stage Claim Verification Pipeline

The following diagram illustrates the end-to-end architecture of the hallucination detection system. A legal document enters the pipeline, is decomposed into typed atomic claims, and each claim is routed through parallel verification strategies selected by claim type. Results are aggregated into a weighted confidence score and surfaced to the user with full evidence trails.

```mermaid
flowchart TB
    %% Stage 1 — Document Input & Chunking
    A["Document Input (Legal Brief / Memo)"] --> B["Chunking & Claim Extraction (LLM-based)"]
    B --> C["Typed Atomic Claims"]

    %% Stage 2 — Claim Classification
    C --> D{"Claim Type Classifier"}
    D -->|CITATION| E1["Citation Claims"]
    D -->|FACTUAL| E2["Factual Claims"]
    D -->|REASONING| E3["Reasoning Claims"]
    D -->|STATISTICAL| E4["Statistical Claims"]

    %% Stage 3 — Parallel Verification
    subgraph PV ["Stage 3: Parallel Verification"]
        direction TB

        subgraph PATH_A ["Citation DB Lookup"]
            F1["CourtListener API Query"] --> F2{"Existence Check"}
            F2 -->|Found| F3["Holding Verification"]
            F2 -->|Not Found| F4["Flag as Fabricated"]
            F3 --> F5["Citation Result"]
            F4 --> F5
        end

        subgraph PATH_B ["NLI Entailment Check"]
            G1["RAG Retrieval (Source Documents)"] --> G2["Source vs. Claim Comparison"]
            G2 --> G3["Entailment Score (0 to 1)"]
            G3 --> G4["Entailment Result"]
        end

        subgraph PATH_C ["Self-Consistency Check"]
            H1["N Regenerations (temperature > 0)"] --> H2["BERTScore Comparison"]
            H2 --> H3["Variance Score"]
            H3 --> H4["Consistency Result"]
        end
    end

    E1 --> F1
    E2 --> G1
    E3 --> H1
    E4 --> G1

    E1 -.->|secondary| G1
    E3 -.->|secondary| G1
    E4 -.->|secondary| H1

    %% Stage 4 — Confidence Aggregation
    F5 --> I["Confidence Aggregation (Weighted by Claim Type)"]
    G4 --> I
    H4 --> I
    I --> J{"Confidence Level"}
    J -->|"score < 0.4"| K1["RED (Likely Hallucination)"]
    J -->|"0.4 <= score < 0.7"| K2["YELLOW (Uncertain)"]
    J -->|"score >= 0.7"| K3["BLUE (Likely Accurate)"]

    %% Stage 5 — Output
    K1 --> L["Severity Classification"]
    K2 --> L
    K3 --> L
    L --> M["User-Facing Output Annotated Document + Evidence Trails"]

    %% Notes
    N1["NOTE: Type-specific routing - CITATION claims prioritize Citation DB Lookup - FACTUAL claims prioritize NLI Entailment - REASONING claims prioritize Self-Consistency - STATISTICAL claims use NLI + Consistency Dashed arrows = secondary verification paths"]

    style N1 fill:#ffffcc,stroke:#cccc00,color:#333
    style K1 fill:#ff6666,stroke:#cc0000,color:#fff
    style K2 fill:#ffcc00,stroke:#cc9900,color:#333
    style K3 fill:#6699ff,stroke:#3366cc,color:#fff
    style PV fill:#f0f4ff,stroke:#aaa
```
