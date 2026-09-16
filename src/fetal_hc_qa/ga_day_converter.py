"""
ga_day_converter.py

GADayConverter -- implements Milestone 2 Algorithm 4 (Gestational-Age-Day
Equivalent Conversion). Converts an absolute HC difference (mm) into an
approximate number of gestational-age days, using the locally applicable HC
growth-velocity rate from the INTERGROWTH-21st international fetal growth
standards (International Fetal and Newborn Growth Consortium for the 21st
Century, 2023), rather than a single fixed mm/day constant, because HC
growth velocity decelerates across pregnancy (~1.7 mm/day near 20-24 weeks
down to ~0.9 mm/day near 32-36 weeks).
"""
import numpy as np

# INTERGROWTH-21st median HC (mm) at 20/24/28/32/36 weeks -- same reference
# table used by scripts/generate_synthetic_dataset.py and cited in Milestones
# 1-2. Values verified directly against the INTERGROWTH-21st data table.
GA_WEEKS = np.array([20, 24, 28, 32, 36], dtype=float)
HC_MM_AT_GA = np.array([172.5, 219.1, 260.4, 294.4, 319.4], dtype=float)


class GADayConverter:
    def hc_velocity_mm_per_day(self, gestational_age_weeks: float) -> float:
        """Local growth-rate estimate (mm/day) at a given gestational age,
        via finite differences over the INTERGROWTH-21st median HC curve."""
        ga_days = np.array([w * 7 for w in GA_WEEKS])
        rates = np.diff(HC_MM_AT_GA) / np.diff(ga_days)  # mm/day per interval
        midpoints = (ga_days[:-1] + ga_days[1:]) / 2.0
        query_days = gestational_age_weeks * 7
        return float(np.interp(query_days, midpoints, rates))

    def to_ga_day_equivalent(self, ai_hc_mm: float, sonographer_hc_mm: float,
                              gestational_age_weeks: float) -> float:
        """Milestone 2 Algorithm 4, verbatim."""
        hc_diff_mm = abs(ai_hc_mm - sonographer_hc_mm)
        growth_rate = self.hc_velocity_mm_per_day(gestational_age_weeks)
        if growth_rate <= 0:
            return float("nan")
        return hc_diff_mm / growth_rate

    def estimate_ga_weeks_from_hc(self, hc_mm: float) -> float:
        """Inverse lookup: approximate gestational age (weeks) for a given HC,
        used when a case has no separately-recorded gestational age (as with
        this synthetic dataset, which stores true GA internally but not in
        the HC18-format CSV, matching the real HC18 schema)."""
        return float(np.interp(hc_mm, HC_MM_AT_GA, GA_WEEKS))
