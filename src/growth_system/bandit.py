from __future__ import annotations

from collections import deque

import numpy as np

from .archetypes import ArmId, ALL_ARMS


class DiscountedThompsonBandit:
    """Thompson sampling with geometric discounting for non-stationary bandits.

    gamma=0.985 gives half-life of ~46 posts (~6.5 weeks at 1 post/day),
    targeting the 6-10 week structural drift scale.
    """

    def __init__(
        self,
        arms: list[ArmId],
        gamma: float = 0.985,
        prior: tuple[float, float] = (1.0, 1.0),
        seed: int | None = None,
    ) -> None:
        self._arms = list(arms)
        self._gamma = gamma
        self._alpha0, self._beta0 = prior
        self._rng = np.random.default_rng(seed)
        self._alpha: dict[ArmId, float] = {a: self._alpha0 for a in arms}
        self._beta: dict[ArmId, float] = {a: self._beta0 for a in arms}

    def recommend(self) -> ArmId:
        samples = {a: self._rng.beta(self._alpha[a], self._beta[a]) for a in self._arms}
        return max(samples, key=lambda a: samples[a])

    def update(self, arm: ArmId, reward: float) -> None:
        # Apply discount to all arms before update
        for a in self._arms:
            self._alpha[a] = self._gamma * self._alpha[a] + self._alpha0
            self._beta[a] = self._gamma * self._beta[a] + self._beta0
        # Update the played arm
        self._alpha[arm] += reward
        self._beta[arm] += 1.0 - reward

    def posterior(self) -> dict[ArmId, tuple[float, float]]:
        return {a: (self._alpha[a], self._beta[a]) for a in self._arms}

    def reset_arm(self, arm: ArmId) -> None:
        """Accelerated forgetting after changepoint detection."""
        self._alpha[arm] = self._alpha0
        self._beta[arm] = self._beta0

    def state_dict(self) -> dict[str, object]:
        return {
            "alpha": dict(self._alpha),
            "beta": dict(self._beta),
            "gamma": self._gamma,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self._alpha = dict(state["alpha"])  # type: ignore[arg-type]
        self._beta = dict(state["beta"])  # type: ignore[arg-type]
        self._gamma = float(state["gamma"])  # type: ignore[arg-type]


class SlidingWindowThompsonBandit:
    """Thompson sampling with sliding window — more interpretable than discount."""

    def __init__(
        self,
        arms: list[ArmId],
        window: int = 35,
        prior: tuple[float, float] = (1.0, 1.0),
        seed: int | None = None,
    ) -> None:
        self._arms = list(arms)
        self._window = window
        self._alpha0, self._beta0 = prior
        self._rng = np.random.default_rng(seed)
        self._history: deque[tuple[ArmId, float]] = deque(maxlen=window)

    def _compute_posteriors(self) -> dict[ArmId, tuple[float, float]]:
        alpha: dict[ArmId, float] = {a: self._alpha0 for a in self._arms}
        beta: dict[ArmId, float] = {a: self._beta0 for a in self._arms}
        for arm, reward in self._history:
            alpha[arm] += reward
            beta[arm] += 1.0 - reward
        return {a: (alpha[a], beta[a]) for a in self._arms}

    def recommend(self) -> ArmId:
        posts = self._compute_posteriors()
        samples = {a: self._rng.beta(posts[a][0], posts[a][1]) for a in self._arms}
        return max(samples, key=lambda a: samples[a])

    def update(self, arm: ArmId, reward: float) -> None:
        self._history.append((arm, reward))

    def posterior(self) -> dict[ArmId, tuple[float, float]]:
        return self._compute_posteriors()

    def reset_arm(self, arm: ArmId) -> None:
        self._history = deque(
            [(a, r) for a, r in self._history if a != arm],
            maxlen=self._window,
        )
