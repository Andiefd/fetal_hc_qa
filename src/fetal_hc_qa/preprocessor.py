"""
preprocessor.py

ImagePreprocessor (Milestone 2 UML). Implements UC-2 (Preprocess Image):
resize to the model's expected input dimensions, normalize intensity, and
(training-time only) augment. Kept backend-agnostic -- it returns a plain
normalized float32 ndarray, so it works unchanged whether the downstream
segmentation backend is the classical CV implementation used in this
Milestone 3 build or the PyTorch U-Net specified in Milestone 2 (a tensor
is just a named ndarray at this boundary).
"""
import cv2
import numpy as np


class ImagePreprocessor:
    def __init__(self, resize_dim=(320, 320)):
        self.resize_dim = resize_dim

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Resize + normalize to [0, 1] float32, per Milestone 2 System Logical
        Model's 'Preprocessed Tensor' data element."""
        resized = cv2.resize(image, self.resize_dim, interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32) / 255.0
        return normalized

    def augment(self, image: np.ndarray) -> np.ndarray:
        """Simple training-time augmentation (horizontal flip + mild brightness
        jitter). Not used at inference time; retained to satisfy the Milestone 2
        interface and for future model-training use."""
        out = image.copy()
        if np.random.rand() < 0.5:
            out = np.fliplr(out)
        jitter = np.random.uniform(-0.05, 0.05)
        out = np.clip(out + jitter, 0.0, 1.0)
        return out
