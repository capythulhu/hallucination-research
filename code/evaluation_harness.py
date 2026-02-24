"""
Evaluation Harness for Legal Hallucination Detection

Computes precision, recall, F1, severity-weighted scores, and per-type
breakdowns against human-annotated ground truth. Designed to answer the
question: "How well does the hallucination detector perform, and — critically —
does it catch the most dangerous hallucinations?"
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from pydantic import BaseModel, Field

from claim_extractor import ClaimType, SeverityLevel
from verification_pipeline import ConfidenceLevel, VerificationResult


class HallucinationLabel(BaseModel):
    """Human-annotated ground-truth label for a single claim."""

    claim_id: str
    is_hallucination: bool
    severity: SeverityLevel
    claim_type: ClaimType
    annotator_id: str | None = None


class EvaluationResult(BaseModel):
    """Comprehensive evaluation metrics for a hallucination detection run."""

    precision: float = Field(
        description="TP / (TP + FP) — of the claims flagged as hallucinations, what fraction actually are?"
    )
    recall: float = Field(
        description="TP / (TP + FN) — of the actual hallucinations, what fraction were caught?"
    )
    f1: float = Field(
        description="Harmonic mean of precision and recall."
    )
    severity_weighted_score: float = Field(
        description="F1-like metric where TP/FP/FN are weighted by claim severity."
    )
    critical_recall: float = Field(
        description="Recall computed only on CRITICAL-severity claims."
    )
    document_detection_rate: float = Field(
        description="Fraction of documents where at least one hallucination was correctly identified."
    )
    per_type_metrics: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="Precision, recall, F1 broken down by ClaimType.",
    )
    latency_stats: dict[str, float] = Field(
        default_factory=dict,
        description="Timing statistics: mean, median, p95 per claim (seconds).",
    )
    cost_estimate: float = Field(
        default=0.0,
        description="Estimated API cost in USD for the full run.",
    )


class EvaluationHarness:
    """
    Computes evaluation metrics for a hallucination detection system.

    The harness treats RED-confidence predictions as "hallucination predicted"
    and everything else as "not hallucination." This binary threshold is
    deliberately conservative: we want high precision on what we flag as
    dangerous, rather than flooding attorneys with false alarms.
    """

    DEFAULT_SEVERITY_WEIGHTS: dict[SeverityLevel, float] = {
        SeverityLevel.CRITICAL: 10.0,
        SeverityLevel.HIGH: 5.0,
        SeverityLevel.MEDIUM: 2.0,
        SeverityLevel.LOW: 1.0,
    }

    def __init__(
        self,
        severity_weights: dict[SeverityLevel, float] | None = None,
    ) -> None:
        self.severity_weights = severity_weights or self.DEFAULT_SEVERITY_WEIGHTS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_metrics(
        self,
        predictions: list[VerificationResult],
        ground_truth: list[HallucinationLabel],
        latency_seconds: float | None = None,
    ) -> EvaluationResult:
        """
        Compute all evaluation metrics.

        Args:
            predictions: Verification results from the pipeline.
            ground_truth: Human-annotated labels for the same claim IDs.
            latency_seconds: Optional total pipeline latency for cost estimation.

        Returns:
            An :class:`EvaluationResult` with all computed metrics.
        """
        gt_map = {label.claim_id: label for label in ground_truth}
        pred_map = {pred.claim_id: pred for pred in predictions}

        # Align predictions with ground truth.
        aligned_pairs: list[tuple[VerificationResult, HallucinationLabel]] = []
        for claim_id, label in gt_map.items():
            if claim_id in pred_map:
                aligned_pairs.append((pred_map[claim_id], label))

        p, r, f = self._claim_level_metrics(aligned_pairs)
        sws = self._severity_weighted_score(aligned_pairs)
        cr = self._critical_recall(aligned_pairs)
        ddr = self._document_detection_rate(aligned_pairs)
        ptm = self._per_type_metrics(aligned_pairs)

        latency_stats = {}
        if latency_seconds is not None and len(predictions) > 0:
            per_claim = latency_seconds / len(predictions)
            latency_stats = {
                "total_seconds": round(latency_seconds, 3),
                "per_claim_seconds": round(per_claim, 3),
                "claims_evaluated": len(predictions),
            }

        # Rough cost estimate: ~$0.003 per 1K input tokens, assume ~800 tokens/claim.
        cost = len(predictions) * 0.003 * 0.8

        return EvaluationResult(
            precision=p,
            recall=r,
            f1=f,
            severity_weighted_score=sws,
            critical_recall=cr,
            document_detection_rate=ddr,
            per_type_metrics=ptm,
            latency_stats=latency_stats,
            cost_estimate=round(cost, 4),
        )

    # ------------------------------------------------------------------
    # Metric implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _claim_level_metrics(
        pairs: list[tuple[VerificationResult, HallucinationLabel]],
    ) -> tuple[float, float, float]:
        """
        Compute precision, recall, and F1.

        A prediction counts as "hallucination predicted" if its confidence
        level is RED.
        """
        tp = fp = fn = 0

        for pred, label in pairs:
            predicted_hallucination = pred.confidence_level == ConfidenceLevel.RED
            actual_hallucination = label.is_hallucination

            if predicted_hallucination and actual_hallucination:
                tp += 1
            elif predicted_hallucination and not actual_hallucination:
                fp += 1
            elif not predicted_hallucination and actual_hallucination:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return round(precision, 4), round(recall, 4), round(f1, 4)

    def _severity_weighted_score(
        self,
        pairs: list[tuple[VerificationResult, HallucinationLabel]],
    ) -> float:
        """
        Compute a severity-weighted F1 analogue.

        Instead of counting each TP/FP/FN as 1, we weight them by the
        severity of the underlying claim. This means catching a CRITICAL
        hallucination counts 10x more than catching a LOW one.
        """
        w_tp = w_fp = w_fn = 0.0

        for pred, label in pairs:
            weight = self.severity_weights.get(label.severity, 1.0)
            predicted = pred.confidence_level == ConfidenceLevel.RED
            actual = label.is_hallucination

            if predicted and actual:
                w_tp += weight
            elif predicted and not actual:
                w_fp += weight
            elif not predicted and actual:
                w_fn += weight

        w_precision = w_tp / (w_tp + w_fp) if (w_tp + w_fp) > 0 else 0.0
        w_recall = w_tp / (w_tp + w_fn) if (w_tp + w_fn) > 0 else 0.0
        w_f1 = (
            2 * w_precision * w_recall / (w_precision + w_recall)
            if (w_precision + w_recall) > 0
            else 0.0
        )

        return round(w_f1, 4)

    @staticmethod
    def _critical_recall(
        pairs: list[tuple[VerificationResult, HallucinationLabel]],
    ) -> float:
        """
        Recall computed exclusively on CRITICAL-severity claims.

        This is the single most important metric: failing to catch a
        fabricated case citation in a brief filed with the court can result
        in Rule 11 sanctions and malpractice liability.
        """
        critical_actual = 0
        critical_caught = 0

        for pred, label in pairs:
            if label.severity == SeverityLevel.CRITICAL and label.is_hallucination:
                critical_actual += 1
                if pred.confidence_level == ConfidenceLevel.RED:
                    critical_caught += 1

        return round(
            critical_caught / critical_actual if critical_actual > 0 else 0.0, 4
        )

    @staticmethod
    def _document_detection_rate(
        pairs: list[tuple[VerificationResult, HallucinationLabel]],
    ) -> float:
        """
        Fraction of documents where at least one hallucination was detected.

        For this single-document demo, the rate is either 0.0 or 1.0. In a
        multi-document evaluation, this metric captures whether the system
        provides *any* useful signal per document, even if it misses some
        individual claims.
        """
        has_hallucination = any(label.is_hallucination for _, label in pairs)
        caught_any = any(
            pred.confidence_level == ConfidenceLevel.RED and label.is_hallucination
            for pred, label in pairs
        )

        if not has_hallucination:
            return 1.0  # No hallucinations to catch — trivially "detected."

        return 1.0 if caught_any else 0.0

    @staticmethod
    def _per_type_metrics(
        pairs: list[tuple[VerificationResult, HallucinationLabel]],
    ) -> dict[str, dict[str, float]]:
        """
        Break down precision, recall, and F1 by claim type.
        """
        type_buckets: dict[str, list[tuple[VerificationResult, HallucinationLabel]]] = (
            defaultdict(list)
        )

        for pred, label in pairs:
            type_buckets[label.claim_type.value].append((pred, label))

        results: dict[str, dict[str, float]] = {}
        for claim_type, bucket in type_buckets.items():
            tp = fp = fn = 0
            for pred, label in bucket:
                predicted = pred.confidence_level == ConfidenceLevel.RED
                actual = label.is_hallucination

                if predicted and actual:
                    tp += 1
                elif predicted and not actual:
                    fp += 1
                elif not predicted and actual:
                    fn += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            results[claim_type] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "count": len(bucket),
            }

        return results

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def generate_report(self, result: EvaluationResult) -> str:
        """Generate a markdown-formatted evaluation report."""
        lines: list[str] = [
            "# Hallucination Detection Evaluation Report",
            "",
            "## Overall Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Precision | {result.precision:.2%} |",
            f"| Recall | {result.recall:.2%} |",
            f"| F1 Score | {result.f1:.2%} |",
            f"| Severity-Weighted Score | {result.severity_weighted_score:.2%} |",
            f"| Critical Recall | {result.critical_recall:.2%} |",
            f"| Document Detection Rate | {result.document_detection_rate:.2%} |",
            "",
            "## Per-Type Breakdown",
            "",
            "| Claim Type | Precision | Recall | F1 | Count |",
            "|------------|-----------|--------|-----|-------|",
        ]

        for ctype, metrics in result.per_type_metrics.items():
            lines.append(
                f"| {ctype.upper()} | {metrics['precision']:.2%} | "
                f"{metrics['recall']:.2%} | {metrics['f1']:.2%} | "
                f"{int(metrics['count'])} |"
            )

        lines.extend([
            "",
            "## Latency",
            "",
        ])
        if result.latency_stats:
            for key, val in result.latency_stats.items():
                lines.append(f"- **{key}**: {val}")
        else:
            lines.append("- No latency data available.")

        lines.extend([
            "",
            f"## Estimated Cost: ${result.cost_estimate:.4f}",
            "",
            "---",
            "*Report generated by the DevSignal Hallucination Detection evaluation harness.*",
        ])

        return "\n".join(lines)

    def print_report(self, result: EvaluationResult) -> None:
        """Print a rich-formatted evaluation report to the terminal."""
        try:
            from rich.console import Console
            from rich.table import Table
            from rich.panel import Panel
            from rich import box
        except ImportError:
            # Graceful fallback if rich is not installed.
            print(self.generate_report(result))
            return

        console = Console()

        # --- Overall metrics table ---
        overall = Table(
            title="Overall Metrics",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
        )
        overall.add_column("Metric", style="bold")
        overall.add_column("Value", justify="right")

        overall.add_row("Precision", f"{result.precision:.2%}")
        overall.add_row("Recall", f"{result.recall:.2%}")
        overall.add_row("F1 Score", f"{result.f1:.2%}")
        overall.add_row("Severity-Weighted Score", f"{result.severity_weighted_score:.2%}")
        overall.add_row(
            "Critical Recall",
            f"[bold {'green' if result.critical_recall >= 0.8 else 'red'}]"
            f"{result.critical_recall:.2%}[/]",
        )
        overall.add_row("Document Detection Rate", f"{result.document_detection_rate:.2%}")

        console.print()
        console.print(overall)

        # --- Per-type breakdown ---
        per_type = Table(
            title="Per-Type Breakdown",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
        )
        per_type.add_column("Claim Type", style="bold")
        per_type.add_column("Precision", justify="right")
        per_type.add_column("Recall", justify="right")
        per_type.add_column("F1", justify="right")
        per_type.add_column("Count", justify="right")

        for ctype, metrics in result.per_type_metrics.items():
            per_type.add_row(
                ctype.upper(),
                f"{metrics['precision']:.2%}",
                f"{metrics['recall']:.2%}",
                f"{metrics['f1']:.2%}",
                str(int(metrics["count"])),
            )

        console.print()
        console.print(per_type)

        # --- Cost & latency panel ---
        latency_text = ""
        if result.latency_stats:
            latency_text = " | ".join(
                f"{k}: {v}" for k, v in result.latency_stats.items()
            )
        else:
            latency_text = "No latency data."

        console.print()
        console.print(
            Panel(
                f"Estimated Cost: ${result.cost_estimate:.4f}\n{latency_text}",
                title="Pipeline Stats",
                border_style="dim",
            )
        )
