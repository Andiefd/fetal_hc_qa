"""End-to-end integration tests for PipelineOrchestrator, run against a small
synthetic fixture dataset generated on the fly (independent of the 60-case
data/sample_dataset used for the full batch demonstration run)."""
import csv
import os

import cv2
import numpy as np
import pytest

from fetal_hc_qa.pipeline_orchestrator import PipelineOrchestrator
from fetal_hc_qa.data_types import Tier


def _draw_case(path, case_id, semi_a=80, semi_b=60, angle=15, noise_pct=0.0):
    size = 320
    img = np.random.default_rng(hash(case_id) % (2**32)).normal(
        loc=60, scale=18, size=(size, size)).astype(np.float32)
    img = cv2.GaussianBlur(img, (0, 0), sigmaX=1.2)
    center = (size // 2, size // 2)
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.ellipse(mask, center, (semi_a, semi_b), angle, 0, 360, 255, thickness=-1)
    ring = np.zeros_like(mask)
    cv2.ellipse(ring, center, (semi_a, semi_b), angle, 0, 360, 255, thickness=6)
    img = img + (mask.astype(np.float32) / 255.0) * 15.0
    img = img + (ring.astype(np.float32) / 255.0) * 90.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    from fetal_hc_qa.measurement_extractor import MeasurementExtractor
    true_perimeter_px = MeasurementExtractor.ramanujan_ellipse_perimeter(semi_a, semi_b)
    pixel_spacing_mm = 0.6
    true_hc_mm = true_perimeter_px * pixel_spacing_mm
    sonographer_hc_mm = true_hc_mm * (1 + noise_pct)

    img_name = f"{case_id}.png"
    cv2.imwrite(os.path.join(path, img_name), img)
    cv2.imwrite(os.path.join(path, img_name.replace(".png", "_Annotation.png")), mask)
    return {"filename": img_name, "pixel size(mm)": pixel_spacing_mm,
            "head circumference (mm)": round(sonographer_hc_mm, 2)}


@pytest.fixture
def small_dataset(tmp_path):
    rows = [
        _draw_case(tmp_path, "case_a", semi_a=80, semi_b=60, angle=10, noise_pct=0.0),
        _draw_case(tmp_path, "case_b", semi_a=90, semi_b=70, angle=45, noise_pct=0.0),
        _draw_case(tmp_path, "case_c", semi_a=70, semi_b=55, angle=100, noise_pct=0.08),  # forced large error
    ]
    csv_path = os.path.join(tmp_path, "training_set_pixel_size_and_HC.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "pixel size(mm)", "head circumference (mm)"])
        writer.writeheader()
        writer.writerows(rows)
    return str(tmp_path)


def test_run_case_produces_a_report(small_dataset, tmp_path):
    output_dir = os.path.join(str(tmp_path), "out")
    orchestrator = PipelineOrchestrator(dataset_dir=small_dataset, output_dir=output_dir)
    report = orchestrator.run_case("case_a.png")
    assert report is not None
    assert report.case_id == "case_a"
    assert report.ai_hc_mm is not None
    assert os.path.isfile(report.image_overlay_path)
    # A clean, undistorted synthetic ellipse should measure closely enough to
    # avoid a false RE_MEASURE flag.
    assert report.tier in (Tier.NO_FLAG, Tier.REVIEW)


def test_run_case_on_missing_case_returns_none(small_dataset, tmp_path):
    output_dir = os.path.join(str(tmp_path), "out")
    orchestrator = PipelineOrchestrator(dataset_dir=small_dataset, output_dir=output_dir)
    assert orchestrator.run_case("does_not_exist.png") is None


def test_run_batch_processes_every_case_and_flags_the_noisy_one(small_dataset, tmp_path):
    output_dir = os.path.join(str(tmp_path), "out")
    orchestrator = PipelineOrchestrator(dataset_dir=small_dataset, output_dir=output_dir)
    summary = orchestrator.run_batch()
    assert summary.n_cases == 3
    assert os.path.isfile(os.path.join(output_dir, "batch_summary.png"))
    assert os.path.isfile(os.path.join(output_dir, "batch_summary.csv"))
    # case_c was generated with an 8% forced measurement error, which exceeds
    # the 4.9% inter-observer limit -- it should land in RE_MEASURE.
    case_c_report = next(r for r in summary.case_reports if r.case_id == "case_c")
    assert case_c_report.tier is Tier.RE_MEASURE
    assert summary.mean_dice is not None and summary.mean_dice > 0.9
