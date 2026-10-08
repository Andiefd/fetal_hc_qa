"""Unit tests for ImagePreprocessor (UC-2: Preprocess Image)."""
import numpy as np
import pytest

from fetal_hc_qa.preprocessor import ImagePreprocessor


def test_preprocess_resizes_to_configured_dimensions():
    pre = ImagePreprocessor(resize_dim=(320, 320))
    image = (np.random.rand(480, 640) * 255).astype(np.uint8)

    out = pre.preprocess(image)

    assert out.shape == (320, 320)


def test_preprocess_normalizes_to_unit_range_float32():
    pre = ImagePreprocessor(resize_dim=(100, 100))
    image = np.zeros((100, 100), dtype=np.uint8)
    image[:, :] = 255  # a solid white image should normalize to all 1.0

    out = pre.preprocess(image)

    assert out.dtype == np.float32
    assert np.max(out) <= 1.0
    assert np.min(out) >= 0.0
    assert np.allclose(out, 1.0)


def test_preprocess_zero_image_normalizes_to_zero():
    pre = ImagePreprocessor(resize_dim=(64, 64))
    image = np.zeros((64, 64), dtype=np.uint8)

    out = pre.preprocess(image)

    assert np.allclose(out, 0.0)


def test_preprocess_respects_non_default_resize_dim():
    pre = ImagePreprocessor(resize_dim=(128, 96))
    image = (np.random.rand(200, 200) * 255).astype(np.uint8)

    out = pre.preprocess(image)

    # cv2.resize takes (width, height); the returned array shape is (rows, cols)
    # i.e. (height, width).
    assert out.shape == (96, 128)


def test_augment_preserves_shape_and_stays_in_unit_range():
    pre = ImagePreprocessor()
    image = np.random.rand(50, 50).astype(np.float32)

    out = pre.augment(image)

    assert out.shape == image.shape
    assert np.max(out) <= 1.0
    assert np.min(out) >= 0.0


def test_augment_can_flip_horizontally(monkeypatch):
    pre = ImagePreprocessor()
    image = np.zeros((10, 10), dtype=np.float32)
    image[:, 0] = 1.0  # a distinctive left-edge column

    # Force the flip branch (rand() < 0.5) and zero jitter for a deterministic check.
    monkeypatch.setattr(np.random, "rand", lambda: 0.0)
    monkeypatch.setattr(np.random, "uniform", lambda lo, hi: 0.0)

    out = pre.augment(image)

    assert np.allclose(out[:, -1], 1.0)
    assert np.allclose(out[:, 0], 0.0)


def test_augment_can_skip_flip(monkeypatch):
    pre = ImagePreprocessor()
    image = np.zeros((10, 10), dtype=np.float32)
    image[:, 0] = 1.0

    monkeypatch.setattr(np.random, "rand", lambda: 0.99)
    monkeypatch.setattr(np.random, "uniform", lambda lo, hi: 0.0)

    out = pre.augment(image)

    assert np.allclose(out[:, 0], 1.0)


def test_augment_brightness_jitter_is_clipped_at_bounds(monkeypatch):
    pre = ImagePreprocessor()
    image = np.ones((10, 10), dtype=np.float32)  # already at the max

    monkeypatch.setattr(np.random, "rand", lambda: 0.99)  # skip flip
    monkeypatch.setattr(np.random, "uniform", lambda lo, hi: hi)  # push jitter to +0.05

    out = pre.augment(image)

    # 1.0 + 0.05 must clip back down to 1.0, not overflow.
    assert np.max(out) == pytest.approx(1.0)
