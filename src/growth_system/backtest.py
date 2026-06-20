from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .archetypes import archetype_of, ALL_ARMS, ArmId
from .bandit import DiscountedThompsonBandit, SlidingWindowThompsonBandit
from .changepoint import CusumDetector
from .reward import BusinessReward, PostOutcome, ProxyLeadSource, RewardWeights
from .scorer import TailScorer
from .config import SystemConfig

REFERENCE_DATE = date(2026, 6, 20)


def parse_days_ago(raw: str) -> int:
    """Convert '1d', '2w', '3mo', '7yr' to days_ago integer."""
    raw = str(raw).strip()
    if raw.endswith('d'):
        return int(raw[:-1])
    elif raw.endswith('w'):
        return int(raw[:-1]) * 7
    elif raw.endswith('mo'):
        return int(raw[:-2]) * 30
    elif raw.endswith('yr'):
        return int(raw[:-2]) * 365
    return 0


def run_backtest(
    posts_path: str,
    audience_path: str,
    model_path: str,
    output_path: str = "backtest_results.png",
    seed: int = 42,
) -> dict[str, object]:
    posts_df = pd.read_csv(posts_path)
    audience_df = pd.read_csv(audience_path, parse_dates=["date"])

    # Sort by days_ago descending (oldest first)
    posts_df = posts_df.sort_values("days_ago", ascending=False).reset_index(drop=True)

    # Compute absolute publish dates
    posts_df["publish_date"] = posts_df["days_ago"].apply(
        lambda d: REFERENCE_DATE - timedelta(days=int(d))
    )

    # Fill missing text
    posts_df["text"] = posts_df["text"].fillna("")

    # Tail indicator (ground truth)
    eng_q80 = 98.2
    posts_df["in_tail"] = (posts_df["eng_score"] >= eng_q80).astype(float)

    # Compute archetype for each post
    posts_df["arm"] = posts_df["text"].apply(archetype_of)

    scorer = TailScorer(model_path)
    reward_fn = BusinessReward(RewardWeights(), ProxyLeadSource())

    # Calibrate CUSUM on first 60 days of audience data
    audience_df = audience_df.sort_values("date")
    smooth_followers = audience_df["new_followers"].rolling(14, min_periods=1).mean()
    calib_series = smooth_followers.iloc[:60]

    mu0 = float(calib_series.mean())
    sigma = float(calib_series.std()) + 1e-6
    k = 0.5 * sigma
    h = 4.5 * sigma

    detector = CusumDetector(mu0=mu0, k=k, h=h)
    changepoints_detected: list[date] = []

    # Run CUSUM over full audience series to find changepoints
    audience_df = audience_df.copy()
    audience_df["smooth_followers"] = smooth_followers
    for _, row in audience_df.iterrows():
        cp = detector.update(float(row["smooth_followers"]), row["date"].date())
        if cp:
            changepoints_detected.append(cp.detected_date)

    rng = np.random.default_rng(seed)

    def compute_reward(row: pd.Series) -> float:  # type: ignore[type-arg]
        outcome = PostOutcome(
            post_id=str(row["post_index"]),
            text=str(row["text"]),
            eng_score=float(row["eng_score"]),
            eng_q80=eng_q80,
        )
        return reward_fn.compute(outcome)

    rewards = posts_df.apply(compute_reward, axis=1).values

    # P0: static baseline — same actual rewards regardless of recommendation
    def policy_static(posts: pd.DataFrame) -> np.ndarray:  # type: ignore[type-arg]
        return rewards.copy()

    # P1: stationary Thompson sampling (gamma=1.0, no discount)
    def policy_stationary(
        posts: pd.DataFrame, seed_offset: int = 1
    ) -> np.ndarray:  # type: ignore[type-arg]
        bandit = DiscountedThompsonBandit(ALL_ARMS, gamma=1.0, seed=seed + seed_offset)
        cumrew = np.zeros(len(posts))
        for i, row in posts.iterrows():
            arm = row["arm"]
            r = rewards[int(i)]
            bandit.update(arm, r)
            cumrew[int(i)] = r
        return cumrew

    # P2: discounted Thompson sampling
    def policy_discounted(
        posts: pd.DataFrame, seed_offset: int = 2
    ) -> tuple[np.ndarray, list[dict[str, float]]]:  # type: ignore[type-arg]
        bandit = DiscountedThompsonBandit(ALL_ARMS, gamma=0.985, seed=seed + seed_offset)
        posteriors_over_time: list[dict[str, float]] = []
        cumrew = np.zeros(len(posts))
        for i, row in posts.iterrows():
            arm = row["arm"]
            r = rewards[int(i)]
            bandit.update(arm, r)
            cumrew[int(i)] = r
            posteriors_over_time.append(
                {
                    a: bandit.posterior()[a][0]
                    / (bandit.posterior()[a][0] + bandit.posterior()[a][1])
                    for a in ALL_ARMS
                }
            )
        return cumrew, posteriors_over_time

    r0 = policy_static(posts_df)
    r1 = policy_stationary(posts_df)
    r2_arr, posteriors = policy_discounted(posts_df)

    cum0 = np.cumsum(r0)
    cum1 = np.cumsum(r1)
    cum2 = np.cumsum(r2_arr)

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    dates = posts_df["publish_date"].values

    # Panel 1: cumulative reward
    ax = axes[0]
    ax.plot(dates, cum0, label="P0 — Statique (baseline)", color="gray", linestyle="--")
    ax.plot(dates, cum1, label="P1 — Thompson stationnaire", color="steelblue")
    ax.plot(dates, cum2, label="P2 — Thompson à escompte (γ=0.985)", color="crimson")
    for cp_date in changepoints_detected:
        ax.axvline(x=np.datetime64(cp_date), color="orange", linestyle=":", alpha=0.8)
    ax.set_title("Récompense cumulée — 3 politiques")
    ax.set_ylabel("Récompense cumulée")
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 2: regret P2 vs P0
    ax = axes[1]
    regret = cum0 - cum2
    ax.plot(dates, regret, color="purple")
    ax.axhline(0, color="black", linewidth=0.5)
    for cp_date in changepoints_detected:
        ax.axvline(
            x=np.datetime64(cp_date),
            color="orange",
            linestyle=":",
            alpha=0.8,
            label="CUSUM alarme",
        )
    ax.set_title("Regret cumulé (P0 − P2)")
    ax.set_ylabel("Regret")
    ax.grid(alpha=0.3)

    # Panel 3: posterior means by arm for P2
    ax = axes[2]
    for arm in ALL_ARMS:
        means = [p[arm] for p in posteriors]
        ax.plot(dates, means, label=arm)
    for cp_date in changepoints_detected:
        ax.axvline(x=np.datetime64(cp_date), color="orange", linestyle=":", alpha=0.8)
    ax.set_title("Probabilité posterior par archétype (P2)")
    ax.set_ylabel("θ moyen")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    summary: dict[str, object] = {
        "total_reward_P0": float(cum0[-1]),
        "total_reward_P1": float(cum1[-1]),
        "total_reward_P2": float(cum2[-1]),
        "regret_P2_vs_P0": float(cum0[-1] - cum2[-1]),
        "changepoints_detected": [str(d) for d in changepoints_detected],
        "p2_beats_p1": bool(cum2[-1] >= cum1[-1]),
        "p1_beats_p0": bool(cum1[-1] >= cum0[-1]),
    }
    return summary
