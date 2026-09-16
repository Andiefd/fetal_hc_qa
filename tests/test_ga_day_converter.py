"""Unit tests for GADayConverter (Milestone 2 Algorithm 4)."""
import pytest

from fetal_hc_qa.ga_day_converter import GADayConverter, GA_WEEKS, HC_MM_AT_GA


@pytest.fixture
def converter():
    return GADayConverter()


def test_velocity_is_higher_early_than_late(converter):
    early_rate = converter.hc_velocity_mm_per_day(22.0)
    late_rate = converter.hc_velocity_mm_per_day(34.0)
    # INTERGROWTH-21st HC growth velocity decelerates across gestation.
    assert early_rate > late_rate > 0


def test_zero_diff_gives_zero_day_equivalent(converter):
    delta = converter.to_ga_day_equivalent(250.0, 250.0, gestational_age_weeks=28.0)
    assert delta == pytest.approx(0.0)


def test_larger_hc_diff_gives_larger_day_equivalent(converter):
    small = converter.to_ga_day_equivalent(251.0, 250.0, gestational_age_weeks=28.0)
    large = converter.to_ga_day_equivalent(260.0, 250.0, gestational_age_weeks=28.0)
    assert large > small > 0


def test_estimate_ga_weeks_from_hc_matches_reference_table(converter):
    # At an exact reference-table HC, the inverse lookup should recover that
    # exact reference gestational age.
    for weeks, hc in zip(GA_WEEKS, HC_MM_AT_GA):
        estimated = converter.estimate_ga_weeks_from_hc(hc)
        assert estimated == pytest.approx(weeks, abs=0.01)


def test_estimate_ga_weeks_from_hc_interpolates(converter):
    midpoint_hc = (HC_MM_AT_GA[0] + HC_MM_AT_GA[1]) / 2.0
    estimated = converter.estimate_ga_weeks_from_hc(midpoint_hc)
    assert GA_WEEKS[0] < estimated < GA_WEEKS[1]
