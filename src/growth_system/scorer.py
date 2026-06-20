from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .features import extract_features


@dataclass
class Contribution:
    feature: str
    value: float
    contribution: float  # signed contribution to log-odds


class TailScorer:
    def __init__(self, model_path: str) -> None:
        with open(model_path) as f:
            m = json.load(f)
        self._features: list[str] = m["features"]
        self._mean = np.array(m["mean"], dtype=float)
        self._scale = np.array(m["scale"], dtype=float)
        self._coef = np.array(m["coef"], dtype=float)
        self._intercept: float = m["intercept"]
        self.eng_q80: float = m["eng_q80"]
        self.base_rate: float = m["base_rate"]
        self.auc: float = m["auc"]

    def features(self, text: str) -> dict[str, float]:
        return extract_features(text)

    def score(self, text: str) -> float:
        feat = extract_features(text)
        x = np.array([feat[f] for f in self._features], dtype=float)
        x_std = (x - self._mean) / self._scale
        log_odds = self._intercept + float(np.dot(self._coef, x_std))
        return float(1.0 / (1.0 + math.exp(-log_odds)))

    def explain(self, text: str) -> list[Contribution]:
        feat = extract_features(text)
        x = np.array([feat[f] for f in self._features], dtype=float)
        x_std = (x - self._mean) / self._scale
        contribs = self._coef * x_std
        return [
            Contribution(
                feature=self._features[i],
                value=float(x[i]),
                contribution=float(contribs[i]),
            )
            for i in range(len(self._features))
        ]

    def verdict(self, text: str) -> Literal["publish", "optimize", "rework"]:
        p = self.score(text)
        if p >= 0.55:
            return "publish"
        elif p >= 0.30:
            return "optimize"
        else:
            return "rework"
