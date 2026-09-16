"""
data_types.py

Plain data (value) objects used throughout the pipeline, matching the
Milestone 2 UML class diagram exactly: UltrasoundCase, DiscrepancyResult,
CaseReport, and BatchSummaryReport. Kept dependency-free (dataclasses only)
so every other module can import them without a circular dependency.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np


class Tier(str, Enum):
    """Discrepancy tiers per Sarris et al. (2012) intra-/inter-observer limits."""
    NO_FLAG = "NO_FLAG"           # <= 3.0% (intra-observer 95% limit of agreement)
    REVIEW = "REVIEW"             # 3.0-4.9% (inter-observer 95% limit of agreement)
    RE_MEASURE = "RE_MEASURE"     # > 4.9% (exceeds published inter-observer limit)
    QUALITY_GATED = "QUALITY_GATED"  # plane-quality confidence below threshold; not classified
    UNMEASURABLE = "UNMEASURABLE"    # segmentation failed to find a head boundary


@dataclass
class UltrasoundCase:
    """One loaded case: an image plus its recorded measurement and metadata."""
    case_id: str
    image: np.ndarray                 # grayscale ndarray, as loaded from disk
    pixel_spacing_mm: float           # millimeters per pixel
    sonographer_hc_mm: float          # recorded HC value to check against
    gestational_age_weeks: Optional[float] = None
    annotation_mask: Optional[np.ndarray] = None  # ground-truth mask, if available (evaluation only)


@dataclass
class DiscrepancyResult:
    """Outcome of ComparisonEngine.evaluate() for one case."""
    tier: Tier
    pct_diff: Optional[float]
    ga_day_delta: Optional[float]
    confidence: float
    ai_hc_mm: Optional[float]
    sonographer_hc_mm: float
    reason: str = ""


@dataclass
class CaseReport:
    """Everything needed to render the Per-Case Report screen for one case."""
    case_id: str
    image_overlay_path: str
    sonographer_hc_mm: float
    ai_hc_mm: Optional[float]
    pct_diff: Optional[float]
    ga_day_delta: Optional[float]
    confidence: float
    tier: Tier
    dice: Optional[float] = None  # only computable when a ground-truth mask exists (evaluation)


@dataclass
class BatchSummaryReport:
    """Aggregate statistics across a full batch run."""
    n_cases: int
    mean_dice: Optional[float]
    mean_abs_diff_mm: Optional[float]
    tier_counts: dict = field(default_factory=dict)
    case_reports: list = field(default_factory=list)
