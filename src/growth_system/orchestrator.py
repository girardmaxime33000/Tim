from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path

import pandas as pd

from .archetypes import ArmId, archetype_of, ALL_ARMS
from .bandit import DiscountedThompsonBandit
from .changepoint import CusumDetector, ChangePoint
from .reward import BusinessReward, PostOutcome, ProxyLeadSource
from .scorer import TailScorer
from .config import SystemConfig

logger = logging.getLogger(__name__)


@dataclass
class DecisionResult:
    recommended_arm: ArmId
    score: float | None  # None if no draft yet
    verdict: str | None
    changepoints: list[ChangePoint]


class GrowthSystem:
    def __init__(self, config: SystemConfig, model_path: str) -> None:
        self.config = config
        self.scorer = TailScorer(model_path)
        self.bandit = DiscountedThompsonBandit(
            arms=ALL_ARMS,
            gamma=config.bandit.gamma,
            prior=(config.bandit.prior_alpha, config.bandit.prior_beta),
            seed=config.seed,
        )
        self.detector = CusumDetector(mu0=0.0, k=0.0, h=0.0)
        self.reward_fn = BusinessReward(config.reward, ProxyLeadSource())
        self._changepoints: list[ChangePoint] = []

    def recommend(self) -> ArmId:
        return self.bandit.recommend()

    def score_draft(self, text: str) -> tuple[float, str]:
        return self.scorer.score(text), self.scorer.verdict(text)

    def record_outcome(
        self,
        text: str,
        eng_score: float,
        post_id: str,
        t: date,
        lambda_t: float,
    ) -> float:
        arm = archetype_of(text)
        outcome = PostOutcome(
            post_id=post_id,
            text=text,
            eng_score=eng_score,
            eng_q80=self.scorer.eng_q80,
        )
        r = self.reward_fn.compute(outcome)
        self.bandit.update(arm, r)
        cp = self.detector.update(lambda_t, t)
        if cp:
            logger.warning("Changepoint detected: %s — resetting bandit priors", cp)
            self._changepoints.append(cp)
            for a in ALL_ARMS:
                self.bandit.reset_arm(a)
        return r

    def calibrate_detector(self, series: pd.Series) -> None:  # type: ignore[type-arg]
        self.detector.calibrate(series)

    def save_state(self, path: str) -> None:
        state = {
            "bandit": self.bandit.state_dict(),
            "changepoints": [asdict(cp) for cp in self._changepoints],
        }
        with open(path, "w") as f:
            json.dump(state, f, default=str)

    def load_state(self, path: str) -> None:
        with open(path) as f:
            state = json.load(f)
        self.bandit.load_state_dict(state["bandit"])
