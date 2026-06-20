from __future__ import annotations

import numpy as np
import pytest

from growth_system.bandit import DiscountedThompsonBandit, SlidingWindowThompsonBandit
from growth_system.archetypes import ALL_ARMS


def simulate_nonstationary(
    bandit: DiscountedThompsonBandit,
    n_rounds: int = 140,
    env_seed: int = 0,
) -> float:
    """
    Non-stationary env (more extreme rates for clear signal):
    - First half: arm 'contrarian' wins (success_rate=0.9), others 0.1
    - Second half: arm 'data' wins (success_rate=0.9), others 0.1
    Returns total reward collected.
    """
    rng = np.random.default_rng(env_seed)
    total_reward = 0.0
    half = n_rounds // 2

    for i in range(n_rounds):
        arm = bandit.recommend()
        # Determine true success rate for this round
        if i < half:
            true_rate = 0.9 if arm == "contrarian" else 0.1
        else:
            true_rate = 0.9 if arm == "data" else 0.1
        reward = float(rng.random() < true_rate)
        bandit.update(arm, reward)
        total_reward += reward

    return total_reward


def test_discounted_beats_stationary_nonstationary_env() -> None:
    """Discounted bandit should accumulate more reward than stationary in non-stationary env."""
    N_TRIALS = 50
    n_rounds = 140

    disc_rewards = []
    stat_rewards = []

    for trial in range(N_TRIALS):
        # Same bandit internal seed, same env randomness — fair comparison
        disc_bandit = DiscountedThompsonBandit(ALL_ARMS, gamma=0.985, seed=trial)
        stat_bandit = DiscountedThompsonBandit(ALL_ARMS, gamma=1.0, seed=trial)

        env_seed = trial + 1000
        disc_rewards.append(simulate_nonstationary(disc_bandit, n_rounds, env_seed=env_seed))
        stat_rewards.append(simulate_nonstationary(stat_bandit, n_rounds, env_seed=env_seed))

    mean_disc = np.mean(disc_rewards)
    mean_stat = np.mean(stat_rewards)

    assert mean_disc > mean_stat, (
        f"Discounted ({mean_disc:.2f}) should beat stationary ({mean_stat:.2f}) "
        f"in a non-stationary environment over {N_TRIALS} trials."
    )


def test_bandit_recommend_returns_valid_arm() -> None:
    bandit = DiscountedThompsonBandit(ALL_ARMS, seed=0)
    for _ in range(10):
        arm = bandit.recommend()
        assert arm in ALL_ARMS


def test_bandit_state_dict_roundtrip() -> None:
    bandit = DiscountedThompsonBandit(ALL_ARMS, gamma=0.985, seed=1)
    # Do some updates
    bandit.update("contrarian", 1.0)
    bandit.update("data", 0.0)
    state = bandit.state_dict()

    bandit2 = DiscountedThompsonBandit(ALL_ARMS, gamma=0.985, seed=2)
    bandit2.load_state_dict(state)
    assert bandit2.posterior() == bandit.posterior()


def test_reset_arm() -> None:
    bandit = DiscountedThompsonBandit(ALL_ARMS, seed=0)
    for _ in range(10):
        bandit.update("contrarian", 1.0)
    alpha_before = bandit.posterior()["contrarian"][0]
    bandit.reset_arm("contrarian")
    alpha_after = bandit.posterior()["contrarian"][0]
    assert alpha_after < alpha_before


def test_sliding_window_bandit() -> None:
    bandit = SlidingWindowThompsonBandit(ALL_ARMS, window=10, seed=0)
    for i in range(15):
        bandit.update("contrarian", 1.0 if i % 2 == 0 else 0.0)
    arm = bandit.recommend()
    assert arm in ALL_ARMS
    post = bandit.posterior()
    assert "contrarian" in post
