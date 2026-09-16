"""
plane_quality_gate.py

PlaneQualityGate (Milestone 2 UML) -- implements Milestone 2 Algorithm 3
(Plane-Quality Confidence Scoring): a weighted composite of three signals
so that a technically difficult scan is quality-gated rather than producing
a false discrepancy flag.

    confidence = 0.4*focus_score + 0.35*shape_score + 0.25*boundary_score

focus_score    -- variance of the Laplacian (a standard blur/focus proxy)
shape_score    -- 1 - normalized ellipse-fit residual (how cleanly the mask
                  boundary matches a true ellipse)
boundary_score -- mean image-gradient magnitude sampled along the fitted
                  ellipse boundary, normalized (stand-in for a trained
                  model's own softmax boundary confidence -- see
                  segmentation_model.py's module docstring for why this
                  pipeline's segmentation backend is classical CV rather
                  than a trained network in this Milestone 3 build)

The three weights and the 0.6 default threshold are the same initial design
defaults documented in Milestone 2, flagged there as pending empirical
tuning; see the Milestone 3 Implementation Plan for the (synthetic-data)
tuning evidence gathered in this build.
"""
import cv2
import numpy as np

from .measurement_extractor import MeasurementExtractor, MeasurementExtractorError

DEFAULT_THRESHOLD = 0.6


class PlaneQualityGate:
    def __init__(self, threshold: float = DEFAULT_THRESHOLD,
                 w_focus: float = 0.4, w_shape: float = 0.35, w_boundary: float = 0.25):
        self.threshold = threshold
        self.w_focus = w_focus
        self.w_shape = w_shape
        self.w_boundary = w_boundary
        self._extractor = MeasurementExtractor()

    def _focus_score(self, image: np.ndarray) -> float:
        img8 = (np.clip(image, 0, 1) * 255).astype(np.uint8) if image.dtype != np.uint8 else image
        variance = cv2.Laplacian(img8, cv2.CV_64F).var()
        # Empirically, clear synthetic/ultrasound-like frames land in the
        # 50-400 range; normalize and clamp to [0, 1].
        return float(np.clip(variance / 300.0, 0.0, 1.0))

    def _shape_score(self, mask: np.ndarray) -> float:
        try:
            (cx, cy), (a, b), angle = self._extractor.fit_ellipse(mask)
        except MeasurementExtractorError:
            return 0.0
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return 0.0
        contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)

        theta = np.deg2rad(angle)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        dx = contour[:, 0] - cx
        dy = contour[:, 1] - cy
        x_rot = dx * cos_t + dy * sin_t
        y_rot = -dx * sin_t + dy * cos_t
        radial = np.sqrt((x_rot / max(a, 1e-6)) ** 2 + (y_rot / max(b, 1e-6)) ** 2)
        residual = float(np.mean(np.abs(radial - 1.0)))
        return float(np.clip(1.0 - residual * 4.0, 0.0, 1.0))

    def _boundary_score(self, image: np.ndarray, mask: np.ndarray) -> float:
        img8 = (np.clip(image, 0, 1) * 255).astype(np.uint8) if image.dtype != np.uint8 else image
        gx = cv2.Sobel(img8, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img8, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(gx ** 2 + gy ** 2)

        boundary = cv2.morphologyEx(
            mask, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        )
        if boundary.sum() == 0:
            return 0.0
        boundary_gradient = gradient_mag[boundary > 0]
        mean_grad = float(np.mean(boundary_gradient)) if boundary_gradient.size else 0.0
        return float(np.clip(mean_grad / 120.0, 0.0, 1.0))

    def assess_confidence(self, image: np.ndarray, mask: np.ndarray) -> float:
        focus = self._focus_score(image)
        shape = self._shape_score(mask)
        boundary = self._boundary_score(image, mask)
        confidence = self.w_focus * focus + self.w_shape * shape + self.w_boundary * boundary
        return float(np.clip(confidence, 0.0, 1.0))
