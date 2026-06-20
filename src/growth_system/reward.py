from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .config import RewardWeights

logger = logging.getLogger(__name__)

_BAIT_RE = re.compile(
    r'(commentez|taguez|partagez si|likez|répondez|dites.moi en com)',
    re.IGNORECASE,
)
_OVERTAG_RE = re.compile(r'@\w+')


@dataclass
class PostOutcome:
    post_id: str
    text: str
    eng_score: float
    eng_q80: float  # threshold for tail membership
    tags_count: int = 0


class LeadSource(Protocol):
    def leads_for(self, post_id: str) -> float: ...


class ProxyLeadSource:
    """Default proxy: estimates lead signal from comment keywords."""

    _CTA_RE = re.compile(
        r'(rdv|calendly|DM|MP|contact|formulaire|audit|appel)',
        re.IGNORECASE,
    )

    def leads_for(self, post_id: str) -> float:
        logger.warning(
            "ProxyLeadSource in use for post %s — optimizing a proxy, not real CRM leads. "
            "Connect a real LeadSource for production use.",
            post_id,
        )
        return 0.0


class BusinessReward:
    """Compute reward aligned with business objective (leads), not vanity metrics."""

    def __init__(self, weights: RewardWeights, lead_source: LeadSource) -> None:
        self._w = weights
        self._lead_source = lead_source

    def compute(self, outcome: PostOutcome) -> float:
        tail_indicator = 1.0 if outcome.eng_score >= outcome.eng_q80 else 0.0
        lead_signal = float(
            np.clip(self._lead_source.leads_for(outcome.post_id), 0.0, 1.0)
        )
        brand_penalty = self._brand_penalty(outcome)

        raw = (
            self._w.w_tail * tail_indicator
            + self._w.w_lead * lead_signal
            - self._w.w_brand * brand_penalty
        )
        return float(np.clip(raw, 0.0, 1.0))

    def _brand_penalty(self, outcome: PostOutcome) -> float:
        penalty = 0.0
        if _BAIT_RE.search(outcome.text):
            penalty += 0.5
        overtags = len(_OVERTAG_RE.findall(outcome.text))
        if overtags > 15:
            penalty += 0.5
        return min(penalty, 1.0)
