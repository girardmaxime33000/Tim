from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

MODEL_PATH = "data/scorer_model.json"
POSTS_PATH = "data/posts.csv"
ENG_Q80 = 98.2


def test_scorer_loads() -> None:
    from growth_system.scorer import TailScorer
    scorer = TailScorer(MODEL_PATH)
    assert scorer.eng_q80 == pytest.approx(98.2, abs=0.1)
    assert 0.0 < scorer.auc <= 1.0


def test_scorer_output_range() -> None:
    from growth_system.scorer import TailScorer
    scorer = TailScorer(MODEL_PATH)
    texts = [
        "",
        "Le mythe de la croissance rapide\n\nEn réalité tout va bien.",
        "Bonjour tout le monde! Comment allez-vous aujourd'hui? #LinkedIn",
    ]
    for t in texts:
        s = scorer.score(t)
        assert 0.0 <= s <= 1.0, f"Score out of range: {s}"


def test_precision_at_top20pct() -> None:
    """Among top-20% scored posts, >= 55% should be actual tail posts."""
    from growth_system.scorer import TailScorer

    scorer = TailScorer(MODEL_PATH)
    df = pd.read_csv(POSTS_PATH)
    df["text"] = df["text"].fillna("")
    df["pred_score"] = df["text"].apply(scorer.score)
    df["in_tail"] = (df["eng_score"] >= ENG_Q80).astype(int)

    n = len(df)
    top_n = max(1, int(n * 0.20))
    top_df = df.nlargest(top_n, "pred_score")
    precision = top_df["in_tail"].mean()

    assert precision >= 0.55, (
        f"Precision@20% = {precision:.3f} < 0.55. "
        f"Top-20% has {top_df['in_tail'].sum()} tail posts out of {len(top_df)}."
    )


def test_verdict_publish() -> None:
    from growth_system.scorer import TailScorer
    scorer = TailScorer(MODEL_PATH)
    # A long text with many positive signals should score high
    text = (
        "Le mythe de la croissance rapide\n\n"
        + "Contenu riche avec beaucoup de détails. " * 30
        + "\n#LinkedInFrance #Croissance"
        + "\nQue pensez-vous de cette approche?"
    )
    verdict = scorer.verdict(text)
    assert verdict in ("publish", "optimize", "rework")


def test_explain_returns_all_features() -> None:
    from growth_system.scorer import TailScorer
    scorer = TailScorer(MODEL_PATH)
    contribs = scorer.explain("Test post avec quelques mots.")
    feature_names = {c.feature for c in contribs}
    assert "log_len" in feature_names
    assert "hook_contrarian" in feature_names
    assert len(contribs) == 8
