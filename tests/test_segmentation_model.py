"""Unit tests for HeadSegmentationModel (UC-3: Segment Head Boundary),
including its UC-3 exception path (no plausible head boundary found).

The model's implementation is a documented, disclosed classical-CV stand-in
for the Milestone 2 U-Net (see module docstring in segmentation_model.py);
these tests validate the *interface contract* -- given a plausible bright
head-boundary ring, predict_mask() returns a single connected, appropriately
sized foreground region, and given no plausible signal it returns an empty
mask -- rather than exact model behavior.
"""
import numpy as np
import cv2
import pytest

from fetal_hc_qa.segmentation_model import HeadSegmentationModel


def _ring_tensor(size=200, semi_a=70, semi_b=55, ring_thickness=6):
    """A [0,1] float32 tensor with a bright elliptical ring on a darker,
    speckled background -- a simplified stand-in for the synthetic dataset's
    ultrasound-style images (see scripts/generate_synthetic_dataset.py)."""
    rng = np.random.RandomState(7)
    base = rng.normal(loc=60, scale=10, size=(size, size)).astype(np.float32)

    center = (size // 2, size // 2)
    ring = np.zeros((size, size), dtype=np.uint8)
    cv2.ellipse(ring, center, (semi_a, semi_b), 0, 0, 360, 255, thickness=ring_thickness)

    img = base + (ring.astype(np.float32) / 255.0) * 150.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img.astype(np.float32) / 255.0, center, semi_a, semi_b


def test_predict_mask_returns_uint8_same_size_as_input():
    model = HeadSegmentationModel()
    tensor, *_ = _ring_tensor(size=150)

    mask = model.predict_mask(tensor)

    assert mask.shape == tensor.shape
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})


def test_predict_mask_finds_a_plausible_head_region_for_a_clear_ring():
    model = HeadSegmentationModel()
    tensor, center, semi_a, semi_b = _ring_tensor(size=200, semi_a=70, semi_b=55)

    mask = model.predict_mask(tensor)

    expected_area = np.pi * semi_a * semi_b
    mask_area = np.sum(mask > 0)

    # The classical-CV pipeline (close + fill + erode) will not reproduce the
    # ellipse's area exactly, but it should be in the right ballpark and the
    # mask's centroid should land near the true ellipse center.
    assert mask_area > 0.3 * expected_area
    assert mask_area < 1.8 * expected_area

    ys, xs = np.nonzero(mask)
    centroid = (float(np.mean(xs)), float(np.mean(ys)))
    assert centroid[0] == pytest.approx(center[0], abs=15)
    assert centroid[1] == pytest.approx(center[1], abs=15)


def test_predict_mask_returns_single_connected_region():
    model = HeadSegmentationModel()
    tensor, *_ = _ring_tensor(size=200)

    mask = model.predict_mask(tensor)

    n_labels, _ = cv2.connectedComponents((mask > 0).astype(np.uint8))
    # n_labels includes the background label, so a single foreground blob is 2.
    assert n_labels == 2


def test_predict_mask_on_all_zero_image_returns_empty_mask():
    """UC-3 exception path: a completely dark/no-signal frame (e.g. a
    corrupted or blank acquisition) -> empty mask, which downstream
    (MeasurementExtractor / ComparisonEngine) is treated as UNMEASURABLE
    rather than raising."""
    model = HeadSegmentationModel()
    tensor = np.zeros((200, 200), dtype=np.float32)

    mask = model.predict_mask(tensor)

    assert mask.shape == tensor.shape
    assert np.sum(mask) == 0


def test_predict_mask_on_uniform_nonzero_image_is_a_disclosed_limitation():
    """A perfectly uniform *non-zero* frame is a degenerate input Otsu
    thresholding cannot bisect meaningfully: with zero variance, the whole
    frame is classified as foreground and (being far above the 1%-of-frame
    floor) is returned as a single full-frame mask rather than triggering
    the empty-mask exception path. This is a real, disclosed limitation of
    the classical-CV segmentation backend (see segmentation_model.py module
    docstring) rather than a case this build claims to handle -- this test
    documents the actual behavior so it is not silently assumed away."""
    model = HeadSegmentationModel()
    tensor = np.full((200, 200), 0.24, dtype=np.float32)

    mask = model.predict_mask(tensor)

    assert np.sum(mask > 0) == tensor.size


def test_predict_mask_ignores_speckle_too_small_to_be_a_head():
    model = HeadSegmentationModel()
    img = np.random.RandomState(3).normal(loc=0.1, scale=0.02, size=(200, 200))
    img = np.clip(img, 0, 1)
    img8 = (img * 255).astype(np.uint8)
    # A handful of tiny bright specks, each far below the 1%-of-frame area floor.
    for (x, y) in [(20, 20), (150, 40), (80, 170)]:
        cv2.circle(img8, (x, y), 2, 255, -1)
    tensor = img8.astype(np.float32) / 255.0

    mask = model.predict_mask(tensor)

    assert np.sum(mask) == 0
