from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd


@dataclass
class ChangePoint:
    detected_date: date
    direction: str  # "up" or "down"
    statistic: float


class CusumDetector:
    """Bilateral CUSUM on smoothed new_followers rate."""

    def __init__(self, mu0: float, k: float, h: float) -> None:
        self.mu0 = mu0
        self.k = k
        self.h = h
        self._s_plus = 0.0
        self._s_minus = 0.0

    def calibrate(self, series: pd.Series) -> None:  # type: ignore[type-arg]
        """Estimate mu0, sigma from series; set k=0.5*sigma, h=4.5*sigma."""
        self.mu0 = float(series.mean())
        sigma = max(float(series.std()), 1e-3)  # guard against zero-variance series
        self.k = 0.5 * sigma
        self.h = 4.5 * sigma
        self._s_plus = 0.0
        self._s_minus = 0.0

    def update(self, x: float, t: date) -> ChangePoint | None:
        self._s_plus = max(0.0, self._s_plus + (x - self.mu0 - self.k))
        self._s_minus = max(0.0, self._s_minus - (x - self.mu0 + self.k))
        if self._s_plus >= self.h:
            cp = ChangePoint(detected_date=t, direction="up", statistic=self._s_plus)
            self._s_plus = 0.0
            self._s_minus = 0.0
            return cp
        if self._s_minus >= self.h:
            cp = ChangePoint(detected_date=t, direction="down", statistic=self._s_minus)
            self._s_plus = 0.0
            self._s_minus = 0.0
            return cp
        return None
