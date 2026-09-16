"""Unit tests for MeasurementExtractor (Milestone 2 Algorithm 1)."""
import math

import numpy as np
import cv2
import pytest

from fetal_hc_qa.measurement_extractor import MeasurementExtractor, MeasurementExtractorError


@pytest.fixture
def extractor():
    return MeasurementExtractor()


def test_ramanujan_perimeter_matches_circle_formula(extractor):
    # A circle is an ellipse with a == b; perimeter should reduce to 2*pi*r.
    r = 50.0
    perimeter = extractor.ramanujan_ellipse_perimeter(r, r)
    assert perimeter == pytest.approx(2 * math.pi * r, rel=1e-6)


def test_ramanujan_perimeter_known_ellipse(extractor):
    # Reference value for semi-axes 50, 30 computed independently via the exact
    # elliptic-integral perimeter (scipy.special.ellipe), not via this codebase's
    # own Ramanujan approximation: 4*a*E(e^2) = 255.2700 mm. Ramanujan's
    # approximation should track this to within a small fraction of a percent.
    perimeter = extractor.ramanujan_ellipse_perimeter(50.0, 30.0)
    assert perimeter == pytest.approx(255.2700, abs=0.5)


def test_ramanujan_perimeter_is_symmetric_in_axes(extractor):
    # The formula only depends on (a+b) and (a-b)^2, so swapping a and b must
    # give the identical result -- this also documents why fit_ellipse()'s
    # (semi_a, semi_b) ordering (which follows cv2.fitEllipse's own axis
    # convention, not necessarily "x-axis first") never affects HC accuracy.
    assert extractor.ramanujan_ellipse_perimeter(50.0, 30.0) == pytest.approx(
        extractor.ramanujan_ellipse_perimeter(30.0, 50.0))


def test_ramanujan_perimeter_zero_axis_is_zero(extractor):
    assert extractor.ramanujan_ellipse_perimeter(0.0, 10.0) == 0.0


def test_px_to_mm_scales_linearly(extractor):
    assert extractor.px_to_mm(100.0, 0.5) == pytest.approx(50.0)
    assert extractor.px_to_mm(0.0, 0.5) == 0.0


def test_fit_ellipse_on_synthetic_disc(extractor):
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.ellipse(mask, (100, 100), (60, 40), 0, 0, 360, 255, thickness=-1)
    (cx, cy), (semi_a, semi_b), _ = extractor.fit_ellipse(mask)
    assert cx == pytest.approx(100, abs=2)
    assert cy == pytest.approx(100, abs=2)
    # cv2.fitEllipse's own (width, height) axis convention does not guarantee
    # which of the two returned semi-axes lines up with the drawing call's
    # x-radius vs. y-radius, so assert the *pair* of axis lengths matches
    # {60, 40} order-independently rather than a fixed position.
    assert sorted([semi_a, semi_b]) == pytest.approx(sorted([40, 60]), abs=3)


def test_fit_ellipse_on_empty_mask_raises(extractor):
    mask = np.zeros((200, 200), dtype=np.uint8)
    with pytest.raises(MeasurementExtractorError):
        extractor.fit_ellipse(mask)


def test_extract_hc_end_to_end_on_known_ellipse(extractor):
    mask = np.zeros((320, 320), dtype=np.uint8)
    cv2.ellipse(mask, (160, 160), (80, 60), 0, 0, 360, 255, thickness=-1)
    pixel_spacing_mm = 0.6
    hc_mm = extractor.extract_hc(mask, pixel_spacing_mm)
    expected_px = extractor.ramanujan_ellipse_perimeter(80, 60)
    assert hc_mm == pytest.approx(expected_px * pixel_spacing_mm, rel=0.05)


def test_extract_hc_returns_none_when_unmeasurable(extractor):
    mask = np.zeros((320, 320), dtype=np.uint8)
    assert extractor.extract_hc(mask, 0.6) is None
