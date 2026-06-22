from __future__ import annotations

from pydantic import BaseModel


class RewardWeights(BaseModel):
    w_tail: float = 0.3
    w_lead: float = 0.6
    w_brand: float = 0.4


class BanditConfig(BaseModel):
    gamma: float = 0.985  # half-life ~6-10 weeks at 1 post/day
    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    window: int = 35  # sliding window alternative


class CusumConfig(BaseModel):
    k_sigma_factor: float = 0.5  # k = k_sigma_factor * sigma
    h_sigma_factor: float = 4.5  # h = h_sigma_factor * sigma
    calibration_days: int = 60


class ScorerConfig(BaseModel):
    publish_threshold: float = 0.55
    optimize_threshold: float = 0.30
    model_path: str = "data/scorer_model.json"
    eng_q80: float = 98.2


class SystemConfig(BaseModel):
    scorer: ScorerConfig = ScorerConfig()
    bandit: BanditConfig = BanditConfig()
    cusum: CusumConfig = CusumConfig()
    reward: RewardWeights = RewardWeights()
    seed: int = 42


# ---------------------------------------------------------------------------
# Attribution leads
# ---------------------------------------------------------------------------

DEFAULT_TYPE_WEIGHTS: dict[str, float] = {
    "Artiste": 1.0,
    "Artiste/galerie": 1.0,
    "Galerie": 1.0,
    "Partenaire": 0.9,
    "Partenaire artistique": 0.9,
    "Prestataire": 0.4,
    "Média": 0.5,
    "Autre": 0.3,
    "À vérifier": 0.5,
}

DEFAULT_ATTRIBUTION_WEIGHT_FALLBACK: float = 0.5
