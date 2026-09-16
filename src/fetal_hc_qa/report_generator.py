"""
report_generator.py

ReportGenerator (Milestone 2 UML) -- implements UC-5 (View Report). Renders
the Per-Case Report and Batch Evaluation Summary screens specified in
Milestone 2's Detailed Specifications, using real pipeline output (not a
mock-up) as PNG images plus machine-readable JSON/CSV.
"""
import os
import json
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data_types import CaseReport, BatchSummaryReport, Tier

TIER_COLORS = {
    Tier.NO_FLAG: "#2E7D3A",
    Tier.REVIEW: "#C97A1B",
    Tier.RE_MEASURE: "#B23A2E",
    Tier.QUALITY_GATED: "#5B4E8C",
    Tier.UNMEASURABLE: "#888888",
}
TIER_LABELS = {
    Tier.NO_FLAG: "NO FLAG (≤ 3.0%, Sarris et al., 2012)",
    Tier.REVIEW: "REVIEW (3.0-4.9%, Sarris et al., 2012)",
    Tier.RE_MEASURE: "RE-MEASURE (> 4.9%, Sarris et al., 2012)",
    Tier.QUALITY_GATED: "QUALITY-GATED (plane confidence below threshold)",
    Tier.UNMEASURABLE: "UNMEASURABLE (no head boundary found)",
}


class ReportGenerator:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.cases_dir = os.path.join(output_dir, "cases")
        os.makedirs(self.cases_dir, exist_ok=True)

    def _dice(self, predicted_mask: np.ndarray, ground_truth_mask: np.ndarray) -> float:
        pred = (predicted_mask > 0).astype(np.uint8)
        gt = cv2.resize(ground_truth_mask, (pred.shape[1], pred.shape[0]),
                         interpolation=cv2.INTER_NEAREST)
        gt = (gt > 0).astype(np.uint8)
        intersection = np.sum(pred & gt)
        total = np.sum(pred) + np.sum(gt)
        if total == 0:
            return 1.0
        return float(2.0 * intersection / total)

    def generate_case_report(self, case, result, mask, preprocessed_image) -> CaseReport:
        dice = None
        if case.annotation_mask is not None and result.ai_hc_mm is not None:
            dice = self._dice(mask, case.annotation_mask)

        overlay_path = os.path.join(self.cases_dir, f"{case.case_id}_report.png")
        self._render_case_figure(case, result, mask, preprocessed_image, dice, overlay_path)

        report = CaseReport(
            case_id=case.case_id,
            image_overlay_path=overlay_path,
            sonographer_hc_mm=case.sonographer_hc_mm,
            ai_hc_mm=result.ai_hc_mm,
            pct_diff=result.pct_diff,
            ga_day_delta=result.ga_day_delta,
            confidence=result.confidence,
            tier=result.tier,
            dice=dice,
        )
        json_path = os.path.join(self.cases_dir, f"{case.case_id}_report.json")
        with open(json_path, "w") as f:
            json.dump({
                "case_id": report.case_id,
                "sonographer_hc_mm": report.sonographer_hc_mm,
                "ai_hc_mm": report.ai_hc_mm,
                "pct_diff": report.pct_diff,
                "ga_day_delta": report.ga_day_delta,
                "confidence": report.confidence,
                "tier": report.tier.value,
                "dice": report.dice,
            }, f, indent=2)
        return report

    def _render_case_figure(self, case, result, mask, preprocessed_image, dice, out_path):
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), dpi=150,
                                  gridspec_kw={"width_ratios": [1, 1.1]})
        ax_img, ax_txt = axes

        ax_img.imshow(preprocessed_image, cmap="gray")
        contours, _ = cv2.findContours((mask > 0).astype(np.uint8),
                                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            c = c.reshape(-1, 2)
            ax_img.plot(c[:, 0], c[:, 1], color="#C97A1B", linewidth=1.6, linestyle="--",
                        label="AI boundary")
        ax_img.set_title(f"{case.case_id}", fontsize=10)
        ax_img.axis("off")

        ax_txt.axis("off")
        lines = [
            ("Sonographer-Recorded HC", f"{result.sonographer_hc_mm:.1f} mm"),
            ("AI-Measured HC", f"{result.ai_hc_mm:.1f} mm" if result.ai_hc_mm else "n/a"),
            ("Percent Difference", f"{result.pct_diff:.2f}%" if result.pct_diff is not None else "n/a"),
            ("GA-Day Equivalent Δ",
             f"{result.ga_day_delta:.2f} days"
             if result.ga_day_delta is not None and not np.isnan(result.ga_day_delta)
             else "n/a"),
            ("Plane-Quality Confidence", f"{result.confidence:.2f} / 1.00"),
            ("Dice (vs. reference mask)", f"{dice:.3f}" if dice is not None else "n/a"),
        ]
        y = 0.95
        for label, val in lines:
            ax_txt.text(0.0, y, label, fontsize=9, va="top")
            ax_txt.text(1.0, y, val, fontsize=9, va="top", ha="right", weight="bold")
            y -= 0.13

        color = TIER_COLORS[result.tier]
        ax_txt.add_patch(plt.Rectangle((0, y - 0.10), 1.0, 0.16, transform=ax_txt.transAxes,
                                        facecolor=color, alpha=0.15, edgecolor=color))
        ax_txt.text(0.5, y - 0.02, TIER_LABELS[result.tier], fontsize=8.3, ha="center",
                    va="top", color=color, weight="bold", wrap=True, transform=ax_txt.transAxes)

        plt.tight_layout()
        plt.savefig(out_path, facecolor="white")
        plt.close(fig)

    def generate_batch_summary(self, case_reports) -> BatchSummaryReport:
        dices = [r.dice for r in case_reports if r.dice is not None]
        diffs = [abs(r.ai_hc_mm - r.sonographer_hc_mm) for r in case_reports if r.ai_hc_mm is not None]
        tier_counts = {}
        for r in case_reports:
            tier_counts[r.tier.value] = tier_counts.get(r.tier.value, 0) + 1

        summary = BatchSummaryReport(
            n_cases=len(case_reports),
            mean_dice=float(np.mean(dices)) if dices else None,
            mean_abs_diff_mm=float(np.mean(diffs)) if diffs else None,
            tier_counts=tier_counts,
            case_reports=case_reports,
        )
        self._render_batch_figure(summary)
        self._write_batch_csv(summary)
        return summary

    def _render_batch_figure(self, summary: BatchSummaryReport):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=150,
                                  gridspec_kw={"width_ratios": [1, 1.3]})
        ax_stats, ax_bar = axes

        ax_stats.axis("off")
        ax_stats.set_title(f"Batch Evaluation Summary (n = {summary.n_cases})", fontsize=11, loc="left")
        stats_lines = [
            ("Mean Dice Coefficient", f"{summary.mean_dice:.3f}" if summary.mean_dice is not None else "n/a"),
            ("Mean Absolute Difference",
             f"{summary.mean_abs_diff_mm:.2f} mm" if summary.mean_abs_diff_mm is not None else "n/a"),
        ]
        y = 0.75
        for label, val in stats_lines:
            ax_stats.text(0.0, y, label, fontsize=10)
            ax_stats.text(0.0, y - 0.12, val, fontsize=16, weight="bold", color="#1F3A5F")
            y -= 0.32

        tiers_order = [Tier.NO_FLAG, Tier.REVIEW, Tier.RE_MEASURE, Tier.QUALITY_GATED, Tier.UNMEASURABLE]
        counts = [summary.tier_counts.get(t.value, 0) for t in tiers_order]
        colors = [TIER_COLORS[t] for t in tiers_order]
        labels = ["No Flag", "Review", "Re-measure", "Quality-\nGated", "Unmeas-\nurable"]
        bars = ax_bar.bar(labels, counts, color=colors)
        for b, c in zip(bars, counts):
            ax_bar.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, str(c),
                        ha="center", fontsize=9, weight="bold")
        ax_bar.set_title("Discrepancy Tier Distribution", fontsize=10)
        ax_bar.spines[["top", "right"]].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "batch_summary.png"), facecolor="white")
        plt.close(fig)

    def _write_batch_csv(self, summary: BatchSummaryReport):
        import csv
        path = os.path.join(self.output_dir, "batch_summary.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["case_id", "sonographer_hc_mm", "ai_hc_mm", "pct_diff", "ga_day_delta",
                              "confidence", "dice", "tier"])
            for r in summary.case_reports:
                writer.writerow([r.case_id, r.sonographer_hc_mm, r.ai_hc_mm, r.pct_diff,
                                  r.ga_day_delta, r.confidence, r.dice, r.tier.value])
