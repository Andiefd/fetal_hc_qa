"""
profile_pipeline.py

Performance-metrics evidence for Milestone 4 (Components Testing / Module
Test Cases). Instruments the exact stage sequence PipelineOrchestrator.run_case()
executes (see pipeline_orchestrator.py) with wall-clock timing per stage, run
across every case in data/sample_dataset/, and reports per-stage mean/median/
max latency plus total end-to-end throughput.

This is a supplement to (not a replacement for) the correctness-focused
pytest suite in tests/: those 62 tests verify each module's *behavior* is
correct; this script verifies the pipeline's *performance* is reasonable for
a QA/second-reader tool meant to run alongside, not block, a sonographer's
workflow.
"""
import os
import statistics
import time

from fetal_hc_qa.dataset_loader import DatasetLoader
from fetal_hc_qa.preprocessor import ImagePreprocessor
from fetal_hc_qa.segmentation_model import HeadSegmentationModel
from fetal_hc_qa.measurement_extractor import MeasurementExtractor
from fetal_hc_qa.plane_quality_gate import PlaneQualityGate
from fetal_hc_qa.flagging_engine import DiscrepancyFlaggingEngine
from fetal_hc_qa.ga_day_converter import GADayConverter
from fetal_hc_qa.comparison_engine import ComparisonEngine
from fetal_hc_qa.report_generator import ReportGenerator

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sample_dataset")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
EVIDENCE_PATH = os.path.join(os.path.dirname(__file__), "..", "evidence", "performance_metrics.txt")
EVIDENCE_CSV = os.path.join(os.path.dirname(__file__), "..", "evidence", "performance_metrics.csv")

STAGES = [
    "load_case", "preprocess", "predict_mask", "extract_hc",
    "evaluate", "generate_case_report",
]


def main():
    loader = DatasetLoader(DATASET_DIR)
    preprocessor = ImagePreprocessor()
    segmentation_model = HeadSegmentationModel()
    measurement_extractor = MeasurementExtractor()
    comparison_engine = ComparisonEngine(
        gate=PlaneQualityGate(),
        flagger=DiscrepancyFlaggingEngine(),
        ga_converter=GADayConverter(),
    )
    report_generator = ReportGenerator(OUTPUT_DIR)

    timings = {stage: [] for stage in STAGES}
    case_ids = loader.list_case_ids()
    t_start = time.perf_counter()

    for case_id in case_ids:
        t0 = time.perf_counter()
        case = loader.load_case(case_id)
        t1 = time.perf_counter()
        timings["load_case"].append(t1 - t0)

        tensor = preprocessor.preprocess(case.image)
        t2 = time.perf_counter()
        timings["preprocess"].append(t2 - t1)

        mask = segmentation_model.predict_mask(tensor)
        t3 = time.perf_counter()
        timings["predict_mask"].append(t3 - t2)

        ai_hc_mm = None
        try:
            ai_hc_mm = measurement_extractor.extract_hc(mask, case.pixel_spacing_mm)
        except Exception:
            pass
        t4 = time.perf_counter()
        timings["extract_hc"].append(t4 - t3)

        confidence_image = tensor if tensor.shape == mask.shape else case.image
        result = comparison_engine.evaluate(case, ai_hc_mm, confidence_image, mask)
        t5 = time.perf_counter()
        timings["evaluate"].append(t5 - t4)

        report_generator.generate_case_report(case, result, mask, confidence_image)
        t6 = time.perf_counter()
        timings["generate_case_report"].append(t6 - t5)

    t_end = time.perf_counter()
    total_wall_s = t_end - t_start
    n = len(case_ids)

    lines = []
    lines.append(f"Performance profiling -- {n} cases, {DATASET_DIR}")
    lines.append(f"Total wall-clock time: {total_wall_s:.3f} s "
                 f"({n / total_wall_s:.2f} cases/sec, "
                 f"{1000 * total_wall_s / n:.1f} ms/case average end-to-end)")
    lines.append("")
    header = f"{'Stage':<22}{'Mean (ms)':>12}{'Median (ms)':>14}{'Max (ms)':>12}{'% of total':>12}"
    lines.append(header)
    lines.append("-" * len(header))

    stage_means_ms = {}
    for stage in STAGES:
        vals_ms = [v * 1000 for v in timings[stage]]
        mean_ms = statistics.mean(vals_ms)
        median_ms = statistics.median(vals_ms)
        max_ms = max(vals_ms)
        stage_means_ms[stage] = mean_ms
        pct = 100 * mean_ms / (1000 * total_wall_s / n)
        lines.append(f"{stage:<22}{mean_ms:>12.2f}{median_ms:>14.2f}{max_ms:>12.2f}{pct:>11.1f}%")

    report_text = "\n".join(lines)
    print(report_text)

    os.makedirs(os.path.dirname(EVIDENCE_PATH), exist_ok=True)
    with open(EVIDENCE_PATH, "w") as f:
        f.write(report_text + "\n")

    with open(EVIDENCE_CSV, "w") as f:
        f.write("stage,mean_ms,median_ms,max_ms\n")
        for stage in STAGES:
            vals_ms = [v * 1000 for v in timings[stage]]
            f.write(f"{stage},{statistics.mean(vals_ms):.4f},"
                    f"{statistics.median(vals_ms):.4f},{max(vals_ms):.4f}\n")

    print(f"\nSaved: {EVIDENCE_PATH}")
    print(f"Saved: {EVIDENCE_CSV}")


if __name__ == "__main__":
    main()
