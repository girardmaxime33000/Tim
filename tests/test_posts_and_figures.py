"""Tests for GET /api/posts and GET /api/monitor/figures — brief extension jalon 3."""
from __future__ import annotations

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from growth_system.features import has_cta, has_link, hook_type_label, extract_features
from growth_system.archetypes import archetype_of
from growth_system.web.api import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Unit — new features.py helpers
# ---------------------------------------------------------------------------

class TestFeatureHelpers:
    def test_hook_type_contrarian(self) -> None:
        assert hook_type_label("Et si tout ce que vous saviez était faux ?") == "Contrarian"

    def test_hook_type_question(self) -> None:
        assert hook_type_label("Avez-vous déjà essayé cette méthode ?") == "Question"

    def test_hook_type_data(self) -> None:
        assert hook_type_label("3 raisons de changer de stratégie.") == "Data"

    def test_hook_type_statement(self) -> None:
        assert hook_type_label("La résilience est une compétence.") == "Statement"

    def test_hook_type_consistent_with_archetype(self) -> None:
        # hook_type_label and archetype_of must agree on every case
        texts = [
            "Et si tout changeait demain ?",
            "Pourquoi cette stratégie échoue ?",
            "Avez-vous déjà essayé ?",
            "3 erreurs classiques.",
            "La croissance demande du temps.",
        ]
        for text in texts:
            label = hook_type_label(text).lower()
            arm = archetype_of(text)
            assert label == arm, f"Mismatch for '{text}': hook_type={label}, archetype={arm}"

    def test_has_link_true(self) -> None:
        assert has_link("Voir ici : https://example.com") is True

    def test_has_link_false(self) -> None:
        assert has_link("Pas de lien ici.") is False

    def test_has_cta_true(self) -> None:
        assert has_cta("👉 Découvrez notre programme") is True

    def test_has_cta_commentez(self) -> None:
        assert has_cta("Commentez avec votre avis.") is True

    def test_has_cta_false(self) -> None:
        assert has_cta("Un post sans appel à l'action.") is False

    def test_emoji_count_via_extract_features(self) -> None:
        # emoji count must come from extract_features, not reimplemented
        text = "Bonjour 🎉🎊"
        feats = extract_features(text)
        assert feats["emoji"] == 2


# ---------------------------------------------------------------------------
# GET /api/posts
# ---------------------------------------------------------------------------

class TestPostsEndpoint:
    def test_posts_returns_200(self) -> None:
        r = client.get("/api/posts")
        assert r.status_code == 200

    def test_posts_schema(self) -> None:
        body = client.get("/api/posts").json()
        assert "posts" in body
        assert "total" in body
        assert "limit" in body
        assert "offset" in body

    def test_posts_count_matches_csv(self) -> None:
        from growth_system.web.api import POSTS_CSV
        if not POSTS_CSV.exists():
            pytest.skip("posts.csv not present")
        import pandas as pd
        n = len(pd.read_csv(POSTS_CSV))
        body = client.get("/api/posts?limit=9999").json()
        assert body["total"] == n

    def test_posts_row_schema(self) -> None:
        body = client.get("/api/posts?limit=1").json()
        if not body["posts"]:
            pytest.skip("no posts")
        row = body["posts"][0]
        for key in ("rank", "excerpt", "date", "format", "hook_type",
                    "likes", "comments", "shares", "eng_score",
                    "text_len", "emoji", "has_link", "has_cta", "permalink"):
            assert key in row, f"missing: {key}"

    def test_posts_excerpt_truncated(self) -> None:
        body = client.get("/api/posts?limit=50").json()
        for row in body["posts"]:
            assert len(row["excerpt"]) <= 160

    def test_posts_hook_type_valid(self) -> None:
        body = client.get("/api/posts?limit=50").json()
        valid = {"Contrarian", "Question", "Data", "Statement"}
        for row in body["posts"]:
            assert row["hook_type"] in valid

    def test_posts_hook_type_consistent_with_features(self) -> None:
        """Cross-check: hook_type in API must match hook_type_label() locally."""
        from growth_system.web.api import POSTS_CSV
        if not POSTS_CSV.exists():
            pytest.skip("posts.csv not present")
        import pandas as pd
        df = pd.read_csv(POSTS_CSV).head(20)
        body = client.get("/api/posts?limit=20&sort_by=date&order=desc").json()
        # Get the texts from the API response (excerpt may be truncated, so compare via features)
        # Instead, apply hook_type_label to full texts and compare with API's hook_type
        texts = df["text"].fillna("").tolist()
        api_hook_types = [row["hook_type"] for row in body["posts"]]
        local_hook_types = [hook_type_label(t) for t in texts[:len(api_hook_types)]]
        # Sort both by the same text ordering to compare
        api_excerpts = [row["excerpt"] for row in body["posts"]]
        for i, text in enumerate(texts):
            if text[:160] in api_excerpts:
                idx = api_excerpts.index(text[:160])
                assert api_hook_types[idx] == hook_type_label(text)

    def test_posts_pagination(self) -> None:
        all_body = client.get("/api/posts?limit=9999").json()
        total = all_body["total"]
        if total < 2:
            pytest.skip("not enough posts")
        page1 = client.get("/api/posts?limit=10&offset=0").json()
        page2 = client.get("/api/posts?limit=10&offset=10").json()
        assert len(page1["posts"]) <= 10
        assert len(page2["posts"]) <= 10
        # First row of page2 should differ from first of page1
        if page2["posts"]:
            assert page1["posts"][0]["permalink"] != page2["posts"][0]["permalink"]

    def test_posts_sort_by_eng_score_desc(self) -> None:
        body = client.get("/api/posts?sort_by=eng_score&order=desc&limit=10").json()
        scores = [r["eng_score"] for r in body["posts"]]
        assert scores == sorted(scores, reverse=True)

    def test_posts_sort_by_eng_score_asc(self) -> None:
        body = client.get("/api/posts?sort_by=eng_score&order=asc&limit=10").json()
        scores = [r["eng_score"] for r in body["posts"]]
        assert scores == sorted(scores)

    def test_posts_filter_format(self) -> None:
        body = client.get("/api/posts?format=Image&limit=50").json()
        for row in body["posts"]:
            assert row["format"] == "Image"

    def test_posts_filter_hook_type(self) -> None:
        body = client.get("/api/posts?hook_type=Statement&limit=50").json()
        for row in body["posts"]:
            assert row["hook_type"] == "Statement"

    def test_posts_filter_has_cta(self) -> None:
        body = client.get("/api/posts?has_cta_filter=true&limit=50").json()
        for row in body["posts"]:
            assert row["has_cta"] is True

    def test_posts_rank_starts_at_1(self) -> None:
        body = client.get("/api/posts?limit=5").json()
        if body["posts"]:
            assert body["posts"][0]["rank"] == 1


# ---------------------------------------------------------------------------
# GET /api/monitor/figures
# ---------------------------------------------------------------------------

class TestMonitorFigures:
    def test_figures_returns_200(self) -> None:
        r = client.get("/api/monitor/figures")
        assert r.status_code == 200

    def test_figures_keys(self) -> None:
        body = client.get("/api/monitor/figures").json()
        for key in ("growth", "distribution", "bandit", "cusum"):
            assert key in body, f"missing figure: {key}"

    def test_figures_valid_plotly_json(self) -> None:
        """Each figure must be deserializable and contain 'data' and 'layout'."""
        import plotly.io as pio
        body = client.get("/api/monitor/figures").json()
        for key, fig_json in body.items():
            fig = pio.from_json(fig_json)
            assert hasattr(fig, "data"), f"{key}: missing data"
            assert hasattr(fig, "layout"), f"{key}: missing layout"

    def test_growth_figure_has_traces(self) -> None:
        import plotly.io as pio
        body = client.get("/api/monitor/figures").json()
        fig = pio.from_json(body["growth"])
        assert len(fig.data) >= 2  # λ(t) + impressions

    def test_distribution_figure_has_histogram(self) -> None:
        import plotly.io as pio
        body = client.get("/api/monitor/figures").json()
        fig = pio.from_json(body["distribution"])
        types = [t.type for t in fig.data]
        assert "histogram" in types

    def test_cusum_figure_has_two_traces(self) -> None:
        import plotly.io as pio
        body = client.get("/api/monitor/figures").json()
        fig = pio.from_json(body["cusum"])
        assert len(fig.data) >= 2  # S+ and S-
