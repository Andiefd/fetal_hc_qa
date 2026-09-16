"""
flagging_engine.py

DiscrepancyFlaggingEngine (Milestone 2 UML) -- implements Milestone 2
Algorithm 2 (Tiered Discrepancy Classification) using the exact Sarris et al.
(2012) 95% limits of agreement for head circumference:
    intra-observer: +/-3.0%  (+/-7.0 mm)
    inter-observer: +/-4.9%  (+/-12.1 mm)
"""
from .data_types import Tier

INTRA_LIMIT_PCT = 3.0
INTER_LIMIT_PCT = 4.9


class DiscrepancyFlaggingEngine:
    def __init__(self, intra_limit: float = INTRA_LIMIT_PCT, inter_limit: float = INTER_LIMIT_PCT):
        self.intra_limit = intra_limit
        self.inter_limit = inter_limit

    def compute_pct_diff(self, ai_hc_mm: float, sonographer_hc_mm: float) -> float:
        if sonographer_hc_mm == 0:
            raise ValueError("sonographer_hc_mm must be non-zero")
        return abs(ai_hc_mm - sonographer_hc_mm) / sonographer_hc_mm * 100.0

    def classify_tier(self, pct_diff: float) -> Tier:
        """Milestone 2 Algorithm 2, verbatim."""
        if pct_diff <= self.intra_limit:
            return Tier.NO_FLAG
        elif pct_diff <= self.inter_limit:
            return Tier.REVIEW
        else:
            return Tier.RE_MEASURE
