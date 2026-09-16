#!/usr/bin/env python3
"""
run_pipeline.py

Command-line entry point: runs the full fetal_hc_qa pipeline end to end on
the synthetic HC18-schema dataset in data/sample_dataset/, using
PipelineOrchestrator exactly as designed in Milestone 2. Produces:
  reports/cases/<case_id>_report.png   (per-case visual report, UC-5)
  reports/cases/<case_id>_report.json  (per-case machine-readable report)
  reports/batch_summary.png            (batch evaluation summary figure)
  reports/batch_summary.csv            (full per-case results table)

Usage:
    python scripts/run_pipeline.py [--limit N]
"""
import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fetal_hc_qa.pipeline_orchestrator import PipelineOrchestrator  # noqa: E402
from fetal_hc_qa.data_types import Tier  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(ROOT, "data", "sample_dataset")
OUTPUT_DIR = os.path.join(ROOT, "reports")


def main():
    parser = argparse.ArgumentParser(description="Run the fetal_hc_qa QA pipeline.")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N cases (default: all).")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if not args.verbose else logging.DEBUG,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    orchestrator = PipelineOrchestrator(dataset_dir=DATASET_DIR, output_dir=OUTPUT_DIR)

    t0 = time.time()
    summary = orchestrator.run_batch(limit=args.limit)
    elapsed = time.time() - t0

    print("\n=== fetal_hc_qa batch run complete ===")
    print(f"Cases processed:        {summary.n_cases}")
    dice_line = f"{summary.mean_dice:.3f}" if summary.mean_dice is not None else "n/a"
    diff_line = f"{summary.mean_abs_diff_mm:.2f} mm" if summary.mean_abs_diff_mm is not None else "n/a"
    print(f"Mean Dice coefficient:  {dice_line}")
    print(f"Mean |AI - Sono.| diff: {diff_line}")
    print("Tier distribution:")
    for tier in Tier:
        print(f"    {tier.value:<15} {summary.tier_counts.get(tier.value, 0)}")
    print(f"Elapsed:                {elapsed:.2f} s")
    print(f"Reports written to:     {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
