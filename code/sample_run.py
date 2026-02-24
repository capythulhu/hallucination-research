"""
End-to-End Demo: Legal Hallucination Detection

Runs the full pipeline on a mock appellate brief excerpt containing five
planted hallucinations of varying type and severity. Demonstrates claim
extraction, multi-source verification, and evaluation against ground truth.

Usage:
    python sample_run.py
"""

from __future__ import annotations

import sys
import time

from claim_extractor import Claim, ClaimExtractor, ClaimType, SeverityLevel
from verification_pipeline import (
    ConfidenceLevel,
    VerificationPipeline,
    VerificationResult,
    VerificationStatus,
)
from evaluation_harness import EvaluationHarness, EvaluationResult, HallucinationLabel


# ======================================================================
# Mock appellate brief excerpt (~900 words)
#
# Contains 5 planted hallucinations, annotated with inline comments below.
# ======================================================================

LEGAL_DOCUMENT = """\
IN THE UNITED STATES COURT OF APPEALS
FOR THE SIXTH CIRCUIT

No. 24-3847

JENNIFER LAWSON, individually and on behalf of all others similarly situated,
    Plaintiff-Appellant,
v.
MIDWEST REGIONAL HEALTH SYSTEMS, INC.,
    Defendant-Appellee.

BRIEF OF PLAINTIFF-APPELLANT

STATEMENT OF THE CASE

Plaintiff Jennifer Lawson was employed as a registered nurse by Defendant \
Midwest Regional Health Systems, Inc. ("MRHS") from March 2018 through her \
termination on September 12, 2023. During her employment, Lawson repeatedly \
reported patient safety violations to her supervisors, including inadequate \
staffing ratios in the intensive care unit and failures to follow medication \
administration protocols established by the Joint Commission.

On August 3, 2023, Lawson filed a formal complaint with the Ohio Department \
of Health regarding these violations. Five weeks later, MRHS terminated her \
employment, citing "performance deficiencies" that had never previously \
appeared in her personnel file. Lawson filed this action under 42 U.S.C. \
Section 1983 and Ohio Rev. Code Section 4113.52, alleging retaliation for \
protected whistleblowing activity.

ARGUMENT

I. THE DISTRICT COURT ERRED IN GRANTING SUMMARY JUDGMENT ON THE SECTION 1983 \
RETALIATION CLAIM.

A. Standard of Review

This Court reviews a district court's grant of summary judgment de novo. \
Morrison v. National Healthcare Corp., 487 F.3d 295 (6th Cir. 2011). Under \
this standard, the Court must view all facts and reasonable inferences in the \
light most favorable to the non-moving party. Ashcroft v. Iqbal, 556 U.S. 662 \
(2009).

B. Lawson Established a Prima Facie Case of Retaliation

To establish a prima facie case of First Amendment retaliation under Section \
1983, a plaintiff must show: (1) she engaged in constitutionally protected \
conduct; (2) an adverse action was taken against her; and (3) a causal \
connection exists between the protected conduct and the adverse action. The \
Supreme Court established this framework in Brown v. Board of Education, 347 \
U.S. 483 (1954), holding that public employees who report safety violations \
are engaged in speech on matters of public concern that is entitled to First \
Amendment protection.

The temporal proximity between Lawson's complaint and her termination — a mere \
five weeks — is itself sufficient to establish the causal connection element. \
This Court has consistently held that temporal proximity of less than three \
months creates a strong inference of retaliation.

C. The Whistleblower Protection Act Provides Additional Statutory Protection

Ohio's Whistleblower Protection Act, Ohio Rev. Code Section 4113.52, was \
enacted on April 15, 1992, to provide robust protections for employees who \
report violations of state or federal law. The statute prohibits employers from \
taking any disciplinary or retaliatory action against an employee who makes a \
good-faith report of a violation to an appropriate authority.

The legislative history demonstrates that the General Assembly intended this \
statute to complement, not supplant, federal whistleblower protections. \
Therefore, the protections available under Section 4113.52 necessarily extend \
to all forms of employer retaliation, including constructive discharge, \
demotion, and reduction in hours. This conclusion follows because the statute \
uses the broad phrase "disciplinary or retaliatory action," and any limiting \
interpretation would contradict the legislature's expressed remedial purpose.

D. Statistical Evidence Supports Systemic Retaliation

Discovery in this case revealed a troubling pattern of retaliation at MRHS. \
Internal records show that approximately 78.3% of employees who filed external \
safety complaints were terminated or constructively discharged within six months \
of their complaints. This rate dramatically exceeds the national average \
turnover rate for healthcare workers and suggests that MRHS maintained an \
institutional practice of punishing whistleblowers.

Furthermore, MRHS's own human resources data demonstrates that Lawson's \
performance evaluations were uniformly positive prior to her complaint. Her \
most recent evaluation, completed just two months before her termination, rated \
her as "exceeds expectations" in all categories. The purported "performance \
deficiencies" cited in her termination letter had no basis in her personnel \
file.

II. THE DISTRICT COURT ABUSED ITS DISCRETION IN EXCLUDING EXPERT TESTIMONY.

The district court excluded the testimony of Dr. Rachel Chen, an organizational \
psychologist who was prepared to testify regarding patterns of workplace \
retaliation in healthcare settings. The exclusion of Dr. Chen's testimony was \
error because her methodology — a comparative analysis of termination rates \
among whistleblowers versus non-whistleblowers — satisfies the reliability \
requirements of Daubert v. Merrell Dow Pharmaceuticals, Inc., 509 U.S. 579 \
(1993).

CONCLUSION

For the foregoing reasons, this Court should reverse the district court's \
grant of summary judgment and remand for trial on all claims.

Respectfully submitted,

JAMES P. HARTWELL
Counsel for Plaintiff-Appellant
"""


# ======================================================================
# Planted hallucinations — ground truth labels.
#
# We build these manually to match claims the extractor will find.
# The IDs will be patched after extraction to align with actual claim IDs.
# ======================================================================

PLANTED_HALLUCINATIONS = [
    {
        "description": (
            "CITATION/CRITICAL: Fabricated case — 'Morrison v. National Healthcare Corp., "
            "487 F.3d 295 (6th Cir. 2011)' does not exist."
        ),
        "type": ClaimType.CITATION,
        "severity": SeverityLevel.CRITICAL,
        "marker": "Morrison v. National Healthcare Corp.",
    },
    {
        "description": (
            "CITATION/HIGH: Brown v. Board of Education cited for a proposition it never "
            "held — it addressed school segregation, not public-employee speech."
        ),
        "type": ClaimType.CITATION,
        "severity": SeverityLevel.HIGH,
        "marker": "Brown v. Board of Education",
    },
    {
        "description": (
            "FACTUAL/HIGH: Ohio Rev. Code Section 4113.52 was enacted on January 17, 1995, "
            "not April 15, 1992 as stated."
        ),
        "type": ClaimType.FACTUAL,
        "severity": SeverityLevel.HIGH,
        "marker": "enacted on April 15, 1992",
    },
    {
        "description": (
            "REASONING/MEDIUM: Non-sequitur — the conclusion that protections 'necessarily "
            "extend to all forms' does not follow from the premise about broad statutory "
            "language alone."
        ),
        "type": ClaimType.REASONING,
        "severity": SeverityLevel.MEDIUM,
        "marker": "necessarily extend",
    },
    {
        "description": (
            "STATISTICAL/LOW: The '78.3%' termination rate for whistleblowers is fabricated — "
            "no such statistic exists in public records."
        ),
        "type": ClaimType.STATISTICAL,
        "severity": SeverityLevel.LOW,
        "marker": "78.3%",
    },
]


def _match_claim_to_hallucination(
    claim: Claim, hallucination: dict
) -> bool:
    """Check whether an extracted claim corresponds to a planted hallucination."""
    return hallucination["marker"].lower() in claim.text.lower()


def _color_for_level(level: ConfidenceLevel) -> str:
    """Return a rich color tag for a confidence level."""
    return {
        ConfidenceLevel.RED: "red",
        ConfidenceLevel.YELLOW: "yellow",
        ConfidenceLevel.BLUE: "blue",
    }.get(level, "white")


def main() -> None:
    """Run the full hallucination detection pipeline."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich import box
    except ImportError:
        print(
            "The 'rich' library is required for formatted output.\n"
            "Install it with: pip install rich\n"
            "Falling back to plain output.\n"
        )
        _main_plain()
        return

    console = Console()

    # ------------------------------------------------------------------
    # Step 1: Claim Extraction
    # ------------------------------------------------------------------
    console.print()
    console.print(
        Panel(
            "[bold]Step 1: Claim Extraction[/bold]\n"
            "Splitting document into chunks and extracting typed, atomic claims.",
            border_style="cyan",
        )
    )

    extractor = ClaimExtractor()
    extraction_start = time.time()
    result = extractor.extract(LEGAL_DOCUMENT, document_id="lawson-v-mrhs-brief")
    extraction_time = time.time() - extraction_start

    console.print(
        f"\n  Extracted [bold]{len(result.claims)}[/bold] claims "
        f"from [bold]{result.chunk_count}[/bold] chunk(s) "
        f"in {extraction_time:.2f}s.\n"
    )

    claims_table = Table(
        title="Extracted Claims",
        box=box.SIMPLE_HEAVY,
        show_lines=True,
        width=110,
    )
    claims_table.add_column("#", style="dim", width=3)
    claims_table.add_column("Type", width=12)
    claims_table.add_column("Severity", width=10)
    claims_table.add_column("Claim Text", width=75, overflow="fold")

    for i, claim in enumerate(result.claims, 1):
        severity_color = {
            SeverityLevel.CRITICAL: "red",
            SeverityLevel.HIGH: "yellow",
            SeverityLevel.MEDIUM: "cyan",
            SeverityLevel.LOW: "dim",
        }.get(claim.severity, "white")

        claims_table.add_row(
            str(i),
            claim.claim_type.value.upper(),
            f"[{severity_color}]{claim.severity.value.upper()}[/]",
            claim.text[:200] + ("..." if len(claim.text) > 200 else ""),
        )

    console.print(claims_table)

    # ------------------------------------------------------------------
    # Step 2: Verification
    # ------------------------------------------------------------------
    console.print()
    console.print(
        Panel(
            "[bold]Step 2: Multi-Source Verification[/bold]\n"
            "Running citation lookup, NLI, and consistency checks on each claim.",
            border_style="cyan",
        )
    )

    pipeline = VerificationPipeline()
    verification_start = time.time()
    verification_results = pipeline.verify_all(
        result.claims, LEGAL_DOCUMENT, context=LEGAL_DOCUMENT
    )
    verification_time = time.time() - verification_start

    console.print(f"\n  Verified {len(verification_results)} claims in {verification_time:.2f}s.\n")

    verification_table = Table(
        title="Verification Results",
        box=box.SIMPLE_HEAVY,
        show_lines=True,
        width=110,
    )
    verification_table.add_column("#", style="dim", width=3)
    verification_table.add_column("Status", width=14)
    verification_table.add_column("Confidence", width=12)
    verification_table.add_column("Score", width=7)
    verification_table.add_column("Explanation", width=65, overflow="fold")

    for i, vr in enumerate(verification_results, 1):
        color = _color_for_level(vr.confidence_level)
        verification_table.add_row(
            str(i),
            vr.status.value.upper(),
            f"[bold {color}]{vr.confidence_level.value.upper()}[/]",
            f"{vr.confidence_score:.3f}",
            vr.explanation[:150] + ("..." if len(vr.explanation) > 150 else ""),
        )

    console.print(verification_table)

    # ------------------------------------------------------------------
    # Step 3: Build ground truth and evaluate
    # ------------------------------------------------------------------
    console.print()
    console.print(
        Panel(
            "[bold]Step 3: Evaluation Against Ground Truth[/bold]\n"
            "Comparing pipeline output to known planted hallucinations.",
            border_style="cyan",
        )
    )

    # Build ground truth labels by matching extracted claims to planted hallucinations.
    ground_truth: list[HallucinationLabel] = []
    matched_hallucinations: dict[str, dict] = {}

    for claim in result.claims:
        is_hallucination = False
        severity = claim.severity
        claim_type = claim.claim_type
        matched_plant = None

        for plant in PLANTED_HALLUCINATIONS:
            if _match_claim_to_hallucination(claim, plant):
                is_hallucination = True
                severity = plant["severity"]
                claim_type = plant["type"]
                matched_plant = plant
                break

        label = HallucinationLabel(
            claim_id=claim.id,
            is_hallucination=is_hallucination,
            severity=severity,
            claim_type=claim_type,
            annotator_id="ground-truth-author",
        )
        ground_truth.append(label)

        if matched_plant is not None:
            matched_hallucinations[claim.id] = matched_plant

    harness = EvaluationHarness()
    total_time = extraction_time + verification_time
    eval_result = harness.compute_metrics(
        verification_results, ground_truth, latency_seconds=total_time
    )

    harness.print_report(eval_result)

    # ------------------------------------------------------------------
    # Step 4: Hallucination Detection Summary
    # ------------------------------------------------------------------
    console.print()
    summary_table = Table(
        title="Planted Hallucination Detection Summary",
        box=box.DOUBLE_EDGE,
        show_lines=True,
        width=110,
    )
    summary_table.add_column("#", style="dim", width=3)
    summary_table.add_column("Type/Severity", width=20)
    summary_table.add_column("Description", width=55, overflow="fold")
    summary_table.add_column("Detected?", width=12, justify="center")

    for i, plant in enumerate(PLANTED_HALLUCINATIONS, 1):
        # Find the claim that matched this hallucination.
        detected = False
        for claim in result.claims:
            if _match_claim_to_hallucination(claim, plant):
                # Check if the verifier flagged it as RED.
                for vr in verification_results:
                    if vr.claim_id == claim.id:
                        detected = vr.confidence_level == ConfidenceLevel.RED
                        break
                break

        det_str = (
            "[bold green]CAUGHT[/]" if detected else "[bold red]MISSED[/]"
        )
        summary_table.add_row(
            str(i),
            f"{plant['type'].value.upper()} / {plant['severity'].value.upper()}",
            plant["description"],
            det_str,
        )

    console.print(summary_table)

    # Final tally.
    caught_count = 0
    for plant in PLANTED_HALLUCINATIONS:
        for claim in result.claims:
            if _match_claim_to_hallucination(claim, plant):
                for vr in verification_results:
                    if vr.claim_id == claim.id and vr.confidence_level == ConfidenceLevel.RED:
                        caught_count += 1
                break

    console.print()
    console.print(
        Panel(
            f"[bold]Result: {caught_count}/{len(PLANTED_HALLUCINATIONS)} "
            f"planted hallucinations detected.[/bold]\n\n"
            f"Precision: {eval_result.precision:.2%} | "
            f"Recall: {eval_result.recall:.2%} | "
            f"F1: {eval_result.f1:.2%}\n"
            f"Critical Recall: {eval_result.critical_recall:.2%} | "
            f"Severity-Weighted Score: {eval_result.severity_weighted_score:.2%}",
            title="Pipeline Summary",
            border_style="bold cyan",
        )
    )
    console.print()


def _main_plain() -> None:
    """Fallback plain-text output when rich is not installed."""
    print("=" * 70)
    print("  Legal Hallucination Detection Pipeline — Demo Run")
    print("=" * 70)

    extractor = ClaimExtractor()
    result = extractor.extract(LEGAL_DOCUMENT, document_id="lawson-v-mrhs-brief")

    print(f"\nExtracted {len(result.claims)} claims from {result.chunk_count} chunk(s).\n")
    for i, claim in enumerate(result.claims, 1):
        print(f"  [{i}] {claim.claim_type.value.upper():12s} | {claim.severity.value.upper():8s} | {claim.text[:80]}...")

    pipeline = VerificationPipeline()
    verification_results = pipeline.verify_all(
        result.claims, LEGAL_DOCUMENT, context=LEGAL_DOCUMENT
    )

    print(f"\nVerification complete. {len(verification_results)} results.\n")
    for i, vr in enumerate(verification_results, 1):
        print(
            f"  [{i}] {vr.confidence_level.value.upper():6s} ({vr.confidence_score:.3f}) "
            f"| {vr.status.value.upper():14s} | {vr.explanation[:60]}..."
        )

    ground_truth: list[HallucinationLabel] = []
    for claim in result.claims:
        is_hall = any(
            _match_claim_to_hallucination(claim, p) for p in PLANTED_HALLUCINATIONS
        )
        ground_truth.append(
            HallucinationLabel(
                claim_id=claim.id,
                is_hallucination=is_hall,
                severity=claim.severity,
                claim_type=claim.claim_type,
            )
        )

    harness = EvaluationHarness()
    eval_result = harness.compute_metrics(verification_results, ground_truth)
    print("\n" + harness.generate_report(eval_result))

    print("\n--- Planted Hallucination Summary ---")
    for i, plant in enumerate(PLANTED_HALLUCINATIONS, 1):
        detected = False
        for claim in result.claims:
            if _match_claim_to_hallucination(claim, plant):
                for vr in verification_results:
                    if vr.claim_id == claim.id:
                        detected = vr.confidence_level == ConfidenceLevel.RED
                        break
                break
        status = "CAUGHT" if detected else "MISSED"
        print(f"  [{i}] {status:6s} | {plant['description'][:70]}...")


if __name__ == "__main__":
    main()
