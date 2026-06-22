from __future__ import annotations

import pytest

from growth_system.reward import BusinessReward, PostOutcome, ProxyLeadSource
from growth_system.config import RewardWeights

ENG_Q80 = 98.2


def make_reward() -> BusinessReward:
    return BusinessReward(RewardWeights(), ProxyLeadSource())


def test_bait_post_lower_reward_than_quality() -> None:
    """A post with bait CTA should have lower reward than a quality post with same engagement."""
    reward_fn = make_reward()

    quality_outcome = PostOutcome(
        post_id="quality",
        text="Le mythe de la croissance rapide\n\nVoici ce que j'ai appris en 10 ans.",
        eng_score=120.0,
        eng_q80=ENG_Q80,
    )

    bait_outcome = PostOutcome(
        post_id="bait",
        text="Commentez OUI si vous voulez recevoir le guide gratuit!",
        eng_score=120.0,
        eng_q80=ENG_Q80,
    )

    r_quality = reward_fn.compute(quality_outcome)
    r_bait = reward_fn.compute(bait_outcome)

    assert r_quality > r_bait, (
        f"Quality reward {r_quality:.3f} should be > bait reward {r_bait:.3f}"
    )


def test_tail_post_reward_positive() -> None:
    reward_fn = make_reward()
    outcome = PostOutcome(
        post_id="tail",
        text="Contenu de qualité sans bait.",
        eng_score=150.0,
        eng_q80=ENG_Q80,
    )
    r = reward_fn.compute(outcome)
    assert r > 0.0


def test_non_tail_post_reward_zero() -> None:
    reward_fn = make_reward()
    outcome = PostOutcome(
        post_id="low",
        text="Contenu ordinaire.",
        eng_score=10.0,
        eng_q80=ENG_Q80,
    )
    r = reward_fn.compute(outcome)
    # No tail, no leads, no bait — reward = 0
    assert r == pytest.approx(0.0, abs=0.01)


def test_reward_clipped_to_0_1() -> None:
    reward_fn = make_reward()
    outcome = PostOutcome(
        post_id="x",
        text="Test",
        eng_score=200.0,
        eng_q80=ENG_Q80,
    )
    r = reward_fn.compute(outcome)
    assert 0.0 <= r <= 1.0


def test_overtag_brand_penalty() -> None:
    """Post with > 15 @mentions should incur brand penalty."""
    reward_fn = make_reward()
    many_tags = " ".join(f"@user{i}" for i in range(20))
    outcome_tagged = PostOutcome(
        post_id="tagged",
        text=f"Super contenu! {many_tags}",
        eng_score=150.0,
        eng_q80=ENG_Q80,
    )
    outcome_clean = PostOutcome(
        post_id="clean",
        text="Super contenu!",
        eng_score=150.0,
        eng_q80=ENG_Q80,
    )
    r_tagged = reward_fn.compute(outcome_tagged)
    r_clean = reward_fn.compute(outcome_clean)
    assert r_clean > r_tagged
