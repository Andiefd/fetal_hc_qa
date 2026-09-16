"""
measurement_extractor.py

MeasurementExtractor (Milestone 2 UML) -- implements Milestone 2 Algorithm 1
(Segmentation Inference & Measurement Extraction): fit an ellipse to the
predicted mask, compute its perimeter via the Ramanujan approximation, and
convert from pixels to millimeters using the case's pixel spacing.
"""
import math
from typing import Optional, Tuple

import cv2
import numpy as np


class MeasurementExtractorError(Exception):
    """Raised when no plausible ellipse can be fit (UC-3 exception path)."""


class MeasurementExtractor:
    def fit_ellipse(self, mask: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
        """Fit an ellipse to the largest contour in a binary mask.
        Returns ((cx, cy), (semi_a_px, semi_b_px), angle_deg)."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise MeasurementExtractorError("No contour found in predicted mask")
        largest = max(contours, key=cv2.contourArea)
        if len(largest) < 5:
            raise MeasurementExtractorError("Contour too small to fit an ellipse")
        (cx, cy), (major, minor), angle = cv2.fitEllipse(largest)
        return (cx, cy), (major / 2.0, minor / 2.0), angle

    @staticmethod
    def ramanujan_ellipse_perimeter(semi_a: float, semi_b: float) -> float:
        """Ramanujan's second approximation for an ellipse's perimeter."""
        a, b = semi_a, semi_b
        if a == 0 or b == 0:
            return 0.0
        h = ((a - b) ** 2) / ((a + b) ** 2)
        return math.pi * (a + b) * (1 + (3 * h) / (10 + math.sqrt(4 - 3 * h)))

    @staticmethod
    def px_to_mm(px_length: float, pixel_spacing_mm: float) -> float:
        return px_length * pixel_spacing_mm

    def extract_hc(self, mask: np.ndarray, pixel_spacing_mm: float) -> Optional[float]:
        """Full Algorithm 1 pipeline: mask -> ellipse -> perimeter (px) -> HC (mm).
        Returns None if no plausible head boundary was found (UC-3 exception)."""
        try:
            _, (semi_a, semi_b), _ = self.fit_ellipse(mask)
        except MeasurementExtractorError:
            return None
        perimeter_px = self.ramanujan_ellipse_perimeter(semi_a, semi_b)
        return self.px_to_mm(perimeter_px, pixel_spacing_mm)
