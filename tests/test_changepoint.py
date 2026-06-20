from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from growth_system.changepoint import CusumDetector, ChangePoint

AUDIENCE_PATH = "data/daily_audience.csv"
RUPTURE_DATE = date(2026, 5, 25)
TOLERANCE_DAYS = 10


def test_cusum_detects_rupture_in_real_data() -> None:
    """CUSUM should detect a changepoint within ±10 days of 2026-05-25.

    We run the detector only after the calibration period (first 60 days)
    and look for any 'up' changepoint near the known rupture date.
    """
    df = pd.read_csv(AUDIENCE_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    smooth = df["new_followers"].rolling(14, min_periods=1).mean()
    calib_n = 60
    calib_series = smooth.iloc[:calib_n]

    detector = CusumDetector(mu0=0.0, k=0.0, h=0.0)
    detector.calibrate(calib_series)

    changepoints: list[ChangePoint] = []
    # Only detect after calibration period to avoid detecting noise in calib data
    for i in range(calib_n, len(df)):
        cp = detector.update(float(smooth.iloc[i]), df["date"].iloc[i].date())
        if cp:
            changepoints.append(cp)

    assert len(changepoints) >= 1, "No changepoint detected in the post-calibration audience data."

    # Check that at least one 'up' changepoint falls within TOLERANCE_DAYS of RUPTURE_DATE
    up_cps = [cp for cp in changepoints if cp.direction == "up"]
    assert len(up_cps) >= 1, "No 'up' changepoint detected."

    near_rupture = [
        cp for cp in up_cps
        if abs((cp.detected_date - RUPTURE_DATE).days) <= TOLERANCE_DAYS
    ]
    assert len(near_rupture) >= 1, (
        f"No 'up' changepoint detected within {TOLERANCE_DAYS} days of {RUPTURE_DATE}. "
        f"Closest up changepoint: {min(up_cps, key=lambda c: abs((c.detected_date - RUPTURE_DATE).days))}"
    )


def test_cusum_no_false_alarm_on_flat_series() -> None:
    """CUSUM should not trigger on a flat series at the calibration mean."""
    # Slightly noisy flat series so sigma > 0
    rng = np.random.default_rng(0)
    flat = pd.Series(5.0 + rng.standard_normal(100) * 0.5)
    detector = CusumDetector(mu0=0.0, k=0.0, h=0.0)
    detector.calibrate(flat.iloc[:60])

    # Feed the same noise level — should not trigger since k/h calibrated to it
    changepoints = []
    base = date(2025, 1, 1)
    for i in range(60, 100):
        cp = detector.update(float(flat.iloc[i]), base + timedelta(days=i))
        if cp:
            changepoints.append(cp)

    # Allow at most 1 false alarm out of 40 observations
    assert len(changepoints) <= 1, f"Too many false alarms: {changepoints}"


def test_cusum_detects_up_shift() -> None:
    """CUSUM should detect an upward shift."""
    low = pd.Series([2.0] * 60)
    high_series = [20.0] * 40

    detector = CusumDetector(mu0=0.0, k=0.0, h=0.0)
    detector.calibrate(low)

    changepoints = []
    base = date(2025, 1, 1)
    for i, val in enumerate(high_series):
        cp = detector.update(val, base + timedelta(days=60 + i))
        if cp:
            changepoints.append(cp)

    assert len(changepoints) >= 1
    assert changepoints[0].direction == "up"
