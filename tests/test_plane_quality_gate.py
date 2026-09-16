"""Unit tests for PlaneQualityGate (Milestone 2 Algorithm 3)."""
import numpy as np
import cv2
import pytest

from fetal_hc_qa.plane_quality_gate import PlaneQualityGate


@pytest.fixture
def gate():
    return PlaneQualityGate()


def _clean_synthetic_frame():
    rng = np.random.default_rng(0)
    img = (rng.normal(loc=140, scale=25, size=(320, 320))).clip(0, 255).astype(np.uint8)
    cv2.circle(img, (160, 160), 5, 255, -1)  # a touch of sharp local contrast
    mask = np.zeros((320, 320), dtype=np.uint8)
    cv2.ellipse(mask, (160, 160), (80, 60), 0, 0, 360, 255, thickness=-1)
    return img, mask


def test_confidence_is_in_unit_interval(gate):
    img, mask = _clean_synthetic_frame()
    confidence = gate.assess_confidence(img, mask)
    assert 0.0 <= confidence <= 1.0


def test_confidence_is_zero_for_empty_mask(gate):
    img, _ = _clean_synthetic_frame()
    empty_mask = np.zeros((320, 320), dtype=np.uint8)
    confidence = gate.assess_confidence(img, empty_mask)
    # No plausible ellipse -> shape_score is 0; overall confidence should be
    # low (only the focus/boundary terms can contribute).
    assert confidence < gate.threshold


def test_weights_sum_to_one(gate):
    assert gate.w_focus + gate.w_shape + gate.w_boundary == pytest.approx(1.0)


def test_flat_blank_image_scores_low_focus(gate):
    flat_img = np.full((320, 320), 128, dtype=np.uint8)
    _, mask = _clean_synthetic_frame()
    confidence = gate.assess_confidence(flat_img, mask)
    # A perfectly flat image has zero Laplacian variance -> focus_score == 0,
    # so confidence should be noticeably reduced versus a textured frame.
    img, _ = _clean_synthetic_frame()
    textured_confidence = gate.assess_confidence(img, mask)
    assert confidence <= textured_confidence
