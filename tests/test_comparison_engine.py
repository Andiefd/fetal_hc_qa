"""Unit tests for ComparisonEngine (UC-4: Compare & Flag Discrepancy),
including both exception paths (UNMEASURABLE, QUALITY_GATED)."""
import numpy as np
import pytest

from fetal_hc_qa.comparison_engine import ComparisonEngine
from fetal_hc_qa.data_types import UltrasoundCase, Tier


class _StubGate:
    """Duck-typed stand-in for PlaneQualityGate so evaluate()'s branching can
    be tested in isolation from real image processing."""
    def __init__(self, confidence, threshold=0.6):
        self._confidence = confidence
        self.threshold = threshold

    def assess_confidence(self, image, mask):
        return self._confidence


class _StubFlagger:
    def __init__(self, pct_diff, tier):
        self._pct_diff = pct_diff
        self._tier = tier

    def compute_pct_diff(self, ai_hc_mm, sonographer_hc_mm):
        return self._pct_diff

    def classify_tier(self, pct_diff):
        return self._tier


class _StubGAConverter:
    def to_ga_day_equivalent(self, ai_hc_mm, sonographer_hc_mm, gestational_age_weeks):
        return 1.23

    def estimate_ga_weeks_from_hc(self, hc_mm):
        return 28.0


def _make_case(ga_weeks=None):
    return UltrasoundCase(
        case_id="t1", image=np.zeros((10, 10), dtype=np.uint8), pixel_spacing_mm=0.5,
        sonographer_hc_mm=250.0, gestational_age_weeks=ga_weeks,
    )


def test_unmeasurable_when_no_ai_measurement():
    engine = ComparisonEngine(gate=_StubGate(0.9), flagger=_StubFlagger(1.0, Tier.NO_FLAG),
                               ga_converter=_StubGAConverter())
    result = engine.evaluate(_make_case(), ai_hc_mm=None, image_for_quality=np.zeros((10, 10)),
                              mask=np.zeros((10, 10)))
    assert result.tier is Tier.UNMEASURABLE
    assert result.pct_diff is None
    assert result.confidence == 0.0


def test_quality_gated_when_confidence_below_threshold():
    engine = ComparisonEngine(gate=_StubGate(0.3, threshold=0.6),
                               flagger=_StubFlagger(1.0, Tier.NO_FLAG),
                               ga_converter=_StubGAConverter())
    result = engine.evaluate(_make_case(), ai_hc_mm=252.0, image_for_quality=np.zeros((10, 10)),
                              mask=np.zeros((10, 10)))
    assert result.tier is Tier.QUALITY_GATED
    assert result.pct_diff is None
    assert result.confidence == pytest.approx(0.3)


def test_normal_path_uses_flagger_tier_and_ga_conversion():
    engine = ComparisonEngine(gate=_StubGate(0.9, threshold=0.6),
                               flagger=_StubFlagger(2.5, Tier.NO_FLAG),
                               ga_converter=_StubGAConverter())
    result = engine.evaluate(_make_case(ga_weeks=28.0), ai_hc_mm=256.0,
                              image_for_quality=np.zeros((10, 10)), mask=np.zeros((10, 10)))
    assert result.tier is Tier.NO_FLAG
    assert result.pct_diff == pytest.approx(2.5)
    assert result.ga_day_delta == pytest.approx(1.23)
    assert result.confidence == pytest.approx(0.9)


def test_normal_path_falls_back_to_estimated_ga_when_absent():
    engine = ComparisonEngine(gate=_StubGate(0.9, threshold=0.6),
                               flagger=_StubFlagger(6.0, Tier.RE_MEASURE),
                               ga_converter=_StubGAConverter())
    result = engine.evaluate(_make_case(ga_weeks=None), ai_hc_mm=270.0,
                              image_for_quality=np.zeros((10, 10)), mask=np.zeros((10, 10)))
    assert result.tier is Tier.RE_MEASURE
    assert result.ga_day_delta == pytest.approx(1.23)  # stub returns fixed value regardless of GA source
