"""Unit tests for DiscrepancyFlaggingEngine (Milestone 2 Algorithm 2)."""
import pytest

from fetal_hc_qa.flagging_engine import DiscrepancyFlaggingEngine
from fetal_hc_qa.data_types import Tier


@pytest.fixture
def engine():
    return DiscrepancyFlaggingEngine()


def test_compute_pct_diff_basic(engine):
    assert engine.compute_pct_diff(100.0, 100.0) == 0.0
    assert engine.compute_pct_diff(105.0, 100.0) == pytest.approx(5.0)
    assert engine.compute_pct_diff(95.0, 100.0) == pytest.approx(5.0)


def test_compute_pct_diff_zero_sonographer_raises(engine):
    with pytest.raises(ValueError):
        engine.compute_pct_diff(10.0, 0.0)


@pytest.mark.parametrize("pct_diff,expected_tier", [
    (0.0, Tier.NO_FLAG),
    (1.5, Tier.NO_FLAG),
    (3.0, Tier.NO_FLAG),          # exactly at the intra-observer boundary -> still NO_FLAG
    (3.0001, Tier.REVIEW),        # just over the boundary -> REVIEW
    (4.0, Tier.REVIEW),
    (4.9, Tier.REVIEW),           # exactly at the inter-observer boundary -> still REVIEW
    (4.9001, Tier.RE_MEASURE),    # just over the boundary -> RE_MEASURE
    (10.0, Tier.RE_MEASURE),
])
def test_classify_tier_boundaries(engine, pct_diff, expected_tier):
    assert engine.classify_tier(pct_diff) is expected_tier


def test_custom_limits_are_respected():
    engine = DiscrepancyFlaggingEngine(intra_limit=2.0, inter_limit=6.0)
    assert engine.classify_tier(2.0) is Tier.NO_FLAG
    assert engine.classify_tier(2.1) is Tier.REVIEW
    assert engine.classify_tier(6.1) is Tier.RE_MEASURE
