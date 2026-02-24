# Hallucination Taxonomy — Three-Axis Classification

The following diagram presents a three-axis taxonomy for classifying hallucinations in legal AI writing. Each hallucination is characterized along three independent dimensions: the underlying **mechanism** (root cause), the observable **manifestation** (what the user sees), and the **impact** (severity in a legal context). Connections between axes show the most common causal relationships, illustrating how specific root causes tend to produce particular observable patterns and severity levels.

```mermaid
flowchart LR
    ROOT["Hallucination Taxonomy"]

    ROOT --- AX1
    ROOT --- AX2
    ROOT --- AX3

    %% Axis 1 — Mechanism
    subgraph AX1 ["Axis 1: Mechanism (Root Cause)"]
        direction TB
        M1["Parametric Confabulation"]
        M2["Retrieval Failure"]
        M3["Context Window Degradation"]
        M4["Compositional Hallucination"]
        M5["Temporal Confusion"]
    end

    %% Axis 2 — Manifestation
    subgraph AX2 ["Axis 2: Manifestation (Observable Type)"]
        direction TB
        O1["Fabricated Citations"]
        O2["Misattribution"]
        O3["Factual Errors"]
        O4["Logical Contradictions"]
        O5["Unsupported Inferences"]
        O6["Statistical Fabrication"]
    end

    %% Axis 3 — Impact
    subgraph AX3 ["Axis 3: Impact (Severity)"]
        direction TB
        I1["CRITICAL Fabricated authority, Inverted holdings"]
        I2["HIGH Temporal errors, Party confusion"]
        I3["MEDIUM Unsupported reasoning"]
        I4["LOW Stylistic confabulation"]
    end

    %% Mechanism -> Manifestation connections
    M1 -->|"generates"| O1
    M1 -->|"generates"| O6
    M2 -->|"causes"| O3
    M2 -->|"causes"| O2
    M3 -->|"produces"| O4
    M3 -->|"produces"| O5
    M4 -->|"creates"| O2
    M4 -->|"creates"| O5
    M5 -->|"leads to"| O3
    M5 -->|"leads to"| O2

    %% Manifestation -> Impact connections
    O1 -->|"severity"| I1
    O2 -->|"severity"| I2
    O3 -->|"severity"| I2
    O4 -->|"severity"| I1
    O5 -->|"severity"| I3
    O6 -->|"severity"| I2

    style AX1 fill:#e6f0ff,stroke:#336699,color:#333
    style AX2 fill:#fff4e6,stroke:#cc8800,color:#333
    style AX3 fill:#ffe6e6,stroke:#cc3333,color:#333
    style I1 fill:#ff6666,stroke:#cc0000,color:#fff
    style I2 fill:#ff9966,stroke:#cc6600,color:#fff
    style I3 fill:#ffcc66,stroke:#cc9900,color:#333
    style I4 fill:#ffffcc,stroke:#cccc00,color:#333
    style ROOT fill:#333,stroke:#000,color:#fff
```
