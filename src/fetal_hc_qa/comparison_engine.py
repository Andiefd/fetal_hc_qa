"""
comparison_engine.py

ComparisonEngine (Milestone 2 UML) -- composes PlaneQualityGate and
DiscrepancyFlaggingEngine (Algorithm 3's `evaluate` function, verbatim from
Milestone 2), and layers on the GA-day-equivalent conversion (Algorithm 4).
Implements UC-4 (Compare & Flag Discrepancy), including both exception paths:
below-threshold plane quality -> QUALITY_GATED (not a false flag), and no AI
measurement available -> UNMEASURABLE.
"""
from typing import Optional

import numpy as np

from .data_types import DiscrepancyResult, Tier, UltrasoundCase
from .plane_quality_gate import PlaneQualityGate
from .flagging_engine import DiscrepancyFlaggingEngine
from .ga_day_converter import GADayConverter


class ComparisonEngine:
    def __init__(self, gate: PlaneQualityGate = None, flagger: DiscrepancyFlaggingEngine = None,
                 ga_converter: GADayConverter = None):
        self.gate = gate or PlaneQualityGate()
        self.flagger = flagger or DiscrepancyFlaggingEngine()
        self.ga_converter = ga_converter or GADayConverter()

    def evaluate(self, case: UltrasoundCase, ai_hc_mm: Optional[float],
                 image_for_quality: np.ndarray, mask: np.ndarray) -> DiscrepancyResult:
        if ai_hc_mm is None:
            return DiscrepancyResult(
                tier=Tier.UNMEASURABLE, pct_diff=None, ga_day_delta=None, confidence=0.0,
                ai_hc_mm=None, sonographer_hc_mm=case.sonographer_hc_mm,
                reason="Segmentation model did not find a plausible head boundary.",
            )

        confidence = self.gate.assess_confidence(image_for_quality, mask)
        if confidence < self.gate.threshold:
            return DiscrepancyResult(
                tier=Tier.QUALITY_GATED, pct_diff=None, ga_day_delta=None, confidence=confidence,
                ai_hc_mm=ai_hc_mm, sonographer_hc_mm=case.sonographer_hc_mm,
                reason=(f"Plane-quality confidence {confidence:.2f} below "
                        f"threshold {self.gate.threshold:.2f}."),
            )

        pct_diff = self.flagger.compute_pct_diff(ai_hc_mm, case.sonographer_hc_mm)
        tier = self.flagger.classify_tier(pct_diff)

        ga_weeks = case.gestational_age_weeks
        if ga_weeks is None:
            ga_weeks = self.ga_converter.estimate_ga_weeks_from_hc(case.sonographer_hc_mm)
        ga_day_delta = self.ga_converter.to_ga_day_equivalent(ai_hc_mm, case.sonographer_hc_mm, ga_weeks)

        return DiscrepancyResult(
            tier=tier, pct_diff=pct_diff, ga_day_delta=ga_day_delta, confidence=confidence,
            ai_hc_mm=ai_hc_mm, sonographer_hc_mm=case.sonographer_hc_mm,
        )
