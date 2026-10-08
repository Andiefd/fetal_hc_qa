"""Unit tests for ReportGenerator (UC-5: View Report) -- Per-Case Report and
Batch Evaluation Summary rendering, covering FR9 and FR10 from the
Milestone 3 Functional Requirements Mapping."""
import os
import json

import numpy as np
import pytest

from fetal_hc_qa.report_generator import ReportGenerator
from fetal_hc_qa.data_types import UltrasoundCase, DiscrepancyResult, CaseReport, Tier


def _case(case_id="001_HC", with_annotation=True):
    image = np.zeros((64, 64), dtype=np.uint8)
    annotation = None
    if with_annotation:
        annotation = np.zeros((64, 64), dtype=np.uint8)
        annotation[16:48, 16:48] = 255
    return UltrasoundCase(
        case_id=case_id,
        image=image,
        pixel_spacing_mm=0.6,
        sonographer_hc_mm=250.0,
        gestational_age_weeks=28.0,
        annotation_mask=annotation,
    )


def _result(tier=Tier.NO_FLAG, pct_diff=1.2, ai_hc_mm=253.0, confidence=0.91):
    return DiscrepancyResult(
        tier=tier,
        pct_diff=pct_diff,
        ga_day_delta=1.4,
        confidence=confidence,
        ai_hc_mm=ai_hc_mm,
        sonographer_hc_mm=250.0,
        reason="within intra-observer limit",
    )


def test_generate_case_report_writes_png_and_json(tmp_path):
    gen = ReportGenerator(str(tmp_path))
    case = _case()
    result = _result()
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[16:48, 16:48] = 255
    preprocessed = np.random.rand(64, 64).astype(np.float32)

    report = gen.generate_case_report(case, result, mask, preprocessed)

    assert os.path.exists(report.image_overlay_path)
    json_path = os.path.join(tmp_path, "cases", "001_HC_report.json")
    assert os.path.exists(json_path)

    with open(json_path) as f:
        payload = json.load(f)
    assert payload["case_id"] == "001_HC"
    assert payload["tier"] == "NO_FLAG"
    assert payload["ai_hc_mm"] == pytest.approx(253.0)


def test_generate_case_report_computes_dice_when_annotation_available(tmp_path):
    gen = ReportGenerator(str(tmp_path))
    case = _case(with_annotation=True)
    result = _result()
    # perfect overlap with the annotation mask -> Dice should be 1.0
    mask = case.annotation_mask.copy()
    preprocessed = np.zeros((64, 64), dtype=np.float32)

    report = gen.generate_case_report(case, result, mask, preprocessed)

    assert report.dice == pytest.approx(1.0)


def test_generate_case_report_dice_none_without_annotation(tmp_path):
    gen = ReportGenerator(str(tmp_path))
    case = _case(with_annotation=False)
    result = _result()
    mask = np.zeros((64, 64), dtype=np.uint8)
    preprocessed = np.zeros((64, 64), dtype=np.float32)

    report = gen.generate_case_report(case, result, mask, preprocessed)

    assert report.dice is None


def test_generate_case_report_handles_unmeasurable_case(tmp_path):
    """UC-3 exception path flowing through to the report: an UNMEASURABLE
    case has no ai_hc_mm/pct_diff, and the report must still render without
    raising."""
    gen = ReportGenerator(str(tmp_path))
    case = _case(with_annotation=False)
    result = DiscrepancyResult(
        tier=Tier.UNMEASURABLE,
        pct_diff=None,
        ga_day_delta=None,
        confidence=0.0,
        ai_hc_mm=None,
        sonographer_hc_mm=250.0,
        reason="no head boundary found",
    )
    mask = np.zeros((64, 64), dtype=np.uint8)
    preprocessed = np.zeros((64, 64), dtype=np.float32)

    report = gen.generate_case_report(case, result, mask, preprocessed)

    assert report.tier == Tier.UNMEASURABLE
    assert report.ai_hc_mm is None
    assert os.path.exists(report.image_overlay_path)


def _case_report(case_id, tier, dice=0.95, ai_hc_mm=253.0, sonographer_hc_mm=250.0):
    return CaseReport(
        case_id=case_id,
        image_overlay_path=f"/tmp/{case_id}.png",
        sonographer_hc_mm=sonographer_hc_mm,
        ai_hc_mm=ai_hc_mm,
        pct_diff=1.2,
        ga_day_delta=1.4,
        confidence=0.9,
        tier=tier,
        dice=dice,
    )


def test_generate_batch_summary_aggregates_stats_and_writes_files(tmp_path):
    gen = ReportGenerator(str(tmp_path))
    reports = [
        _case_report("001_HC", Tier.NO_FLAG, dice=0.98, ai_hc_mm=252.0, sonographer_hc_mm=250.0),
        _case_report("002_HC", Tier.REVIEW, dice=0.90, ai_hc_mm=260.0, sonographer_hc_mm=250.0),
        _case_report("003_HC", Tier.UNMEASURABLE, dice=None, ai_hc_mm=None, sonographer_hc_mm=250.0),
    ]

    summary = gen.generate_batch_summary(reports)

    assert summary.n_cases == 3
    assert summary.mean_dice == pytest.approx((0.98 + 0.90) / 2)
    assert summary.mean_abs_diff_mm == pytest.approx((2.0 + 10.0) / 2)
    assert summary.tier_counts == {"NO_FLAG": 1, "REVIEW": 1, "UNMEASURABLE": 1}

    assert os.path.exists(os.path.join(tmp_path, "batch_summary.png"))
    csv_path = os.path.join(tmp_path, "batch_summary.csv")
    assert os.path.exists(csv_path)
    with open(csv_path) as f:
        lines = f.readlines()
    assert len(lines) == 4  # header + 3 rows


def test_generate_batch_summary_handles_no_measurable_cases(tmp_path):
    gen = ReportGenerator(str(tmp_path))
    reports = [_case_report("001_HC", Tier.UNMEASURABLE, dice=None, ai_hc_mm=None)]

    summary = gen.generate_batch_summary(reports)

    assert summary.mean_dice is None
    assert summary.mean_abs_diff_mm is None
