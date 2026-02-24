# Supporting Materials

Proof-of-concept code and architecture diagrams for the hallucination detection pipeline described in the research proposal.

These are meant to supplement the proposal — they demonstrate the pipeline architecture and show how the pieces fit together, but the proposal stands on its own without them.

## Code

A working demo of the multi-stage claim verification pipeline. Fully mocked — no API keys needed.

| File                            | What it does                                                                                                                                             |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `code/claim_extractor.py`       | Breaks a legal document into typed atomic claims (CITATION, FACTUAL, REASONING, STATISTICAL) with severity estimates                                     |
| `code/verification_pipeline.py` | Runs each claim through type-appropriate verifiers (citation DB lookup, NLI entailment, self-consistency) and aggregates into RED/YELLOW/BLUE confidence |
| `code/evaluation_harness.py`    | Computes precision, recall, critical recall, severity-weighted scores, and generates reports                                                             |
| `code/sample_run.py`            | End-to-end demo: a mock legal brief with 5 planted hallucinations run through the full pipeline                                                          |

### Running the demo

```bash
cd code
pip install -r requirements.txt
python sample_run.py
```

The demo processes a ~900-word appellate brief excerpt with 5 planted hallucinations of different types and severities. It catches 4 out of 5 (100% precision, 80% recall, 100% critical recall). The one miss — a reasoning-level non-sequitur — is a realistic limitation of automated detection.

## Diagrams

Mermaid diagrams that render in any Markdown viewer (GitHub, VS Code, etc.).

| File                                 | What it shows                                                                                    |
| ------------------------------------ | ------------------------------------------------------------------------------------------------ |
| `diagrams/architecture-overview.md`  | The 5-stage pipeline end-to-end, including type-specific routing and parallel verification paths |
| `diagrams/detection-pipeline.md`     | Detailed data flow from raw document to flagged output, with confidence threshold mapping        |
| `diagrams/hallucination-taxonomy.md` | The three-axis classification (mechanism × manifestation × impact) with connections between axes |
