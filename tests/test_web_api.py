"""Tests for web API endpoints — Jalon 1.

Covers: /api/health, /api/profile, /api/score, /api/recommend, /api/update.
"""
from __future__ import annotations

import json
import pytest
from fastapi.testclient import TestClient

from growth_system.web.api import app, _bandit, STATE_JSON, LEADS_CSV

client = TestClient(app)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_DRAFT = (
    "Et si les galeries d'art avaient tort depuis 30 ans ?\n\n"
    "Voici ce que les données montrent sur le marché secondaire :\n"
    "- Artistes émergents : +340 % en 5 ans\n"
    "- Maisons classiques : stagnation\n\n"
    "Ce que ça change pour les collectionneurs ? Tout.\n"
    "#art #investissement"
)

SHORT_DRAFT = "Court"


# ---------------------------------------------------------------------------
# /api/health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_200(self) -> None:
        r = client.get("/api/health")
        assert r.status_code == 200

    def test_health_schema(self) -> None:
        r = client.get("/api/health")
        body = r.json()
        assert "status" in body
        assert "files" in body
        assert "posts_count" in body
        assert "model_loaded" in body

    def test_health_files_keys(self) -> None:
        r = client.get("/api/health")
        files = r.json()["files"]
        assert "posts.csv" in files
        assert "daily_audience.csv" in files
        assert "scorer_model.json" in files

    def test_health_status_ok_when_files_present(self) -> None:
        r = client.get("/api/health")
        body = r.json()
        # All data files should be present in this test environment
        if all(body["files"].values()):
            assert body["status"] == "ok"
            assert body["model_loaded"] is True


# ---------------------------------------------------------------------------
# /api/profile
# ---------------------------------------------------------------------------

class TestProfile:
    def test_profile_returns_200(self) -> None:
        r = client.get("/api/profile")
        assert r.status_code == 200

    def test_profile_schema(self) -> None:
        r = client.get("/api/profile")
        body = r.json()
        assert "posts_count" in body
        assert "tail_rate" in body
        assert "top_archetypes" in body

    def test_profile_posts_count_positive(self) -> None:
        r = client.get("/api/profile")
        assert r.json()["posts_count"] > 0

    def test_profile_tail_rate_in_range(self) -> None:
        r = client.get("/api/profile")
        rate = r.json()["tail_rate"]
        assert 0.0 <= rate <= 1.0

    def test_profile_top_archetypes_arms(self) -> None:
        r = client.get("/api/profile")
        arms = r.json()["top_archetypes"]
        assert len(arms) > 0
        for v in arms.values():
            assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# /api/score
# ---------------------------------------------------------------------------

class TestScore:
    def test_score_returns_200(self) -> None:
        r = client.post("/api/score", json={"text": SAMPLE_DRAFT})
        assert r.status_code == 200

    def test_score_schema(self) -> None:
        r = client.post("/api/score", json={"text": SAMPLE_DRAFT})
        body = r.json()
        assert "p_queue" in body
        assert "verdict" in body
        assert "contributions" in body
        assert "suggestions" in body

    def test_score_p_queue_in_range(self) -> None:
        r = client.post("/api/score", json={"text": SAMPLE_DRAFT})
        p = r.json()["p_queue"]
        assert 0.0 <= p <= 1.0

    def test_score_verdict_valid(self) -> None:
        r = client.post("/api/score", json={"text": SAMPLE_DRAFT})
        assert r.json()["verdict"] in {"publish", "optimize", "rework"}

    def test_score_contributions_count(self) -> None:
        r = client.post("/api/score", json={"text": SAMPLE_DRAFT})
        contribs = r.json()["contributions"]
        assert len(contribs) == 8  # 8 features

    def test_score_contributions_schema(self) -> None:
        r = client.post("/api/score", json={"text": SAMPLE_DRAFT})
        for c in r.json()["contributions"]:
            assert "feature" in c
            assert "value" in c
            assert "contribution" in c

    def test_score_suggestions_non_empty(self) -> None:
        r = client.post("/api/score", json={"text": SAMPLE_DRAFT})
        assert len(r.json()["suggestions"]) >= 1

    def test_score_empty_text_rejected(self) -> None:
        r = client.post("/api/score", json={"text": ""})
        assert r.status_code == 422

    def test_score_with_format(self) -> None:
        r = client.post("/api/score", json={"text": SAMPLE_DRAFT, "format": "image"})
        assert r.status_code == 200

    def test_score_contrarian_hook_higher_than_statement(self) -> None:
        # Same length, same structure — only the hook differs
        body = (
            "\n\nVoici 3 choses que personne ne vous dit sur le marché de l'art :\n"
            "- Les galeries traditionnelles perdent du terrain chaque année\n"
            "- Les artistes émergents génèrent 3× plus de rendement\n"
            "- La digitalisation change les règles du jeu\n\n"
            "Ce que vous devriez faire dès maintenant ? Diversifier.\n"
            "#art #investissement #collection"
        )
        contrarian = "Et si tout ce que vous saviez sur l'art était faux ?" + body
        statement = "Voici ce que j'ai appris sur le marché de l'art cette année." + body
        r_c = client.post("/api/score", json={"text": contrarian})
        r_s = client.post("/api/score", json={"text": statement})
        # Contrarian hook coefficient is positive → should score higher
        assert r_c.json()["p_queue"] >= r_s.json()["p_queue"]


# ---------------------------------------------------------------------------
# /api/recommend
# ---------------------------------------------------------------------------

class TestRecommend:
    def test_recommend_returns_200(self) -> None:
        r = client.get("/api/recommend")
        assert r.status_code == 200

    def test_recommend_schema(self) -> None:
        r = client.get("/api/recommend")
        body = r.json()
        assert "recommended_arm" in body
        assert "posteriors" in body

    def test_recommend_valid_arm(self) -> None:
        r = client.get("/api/recommend")
        arm = r.json()["recommended_arm"]
        assert arm in {"contrarian", "data", "question", "statement"}

    def test_recommend_posteriors_all_arms(self) -> None:
        r = client.get("/api/recommend")
        posteriors = r.json()["posteriors"]
        for arm in ("contrarian", "data", "question", "statement"):
            assert arm in posteriors

    def test_recommend_posteriors_schema(self) -> None:
        r = client.get("/api/recommend")
        for arm, ab in r.json()["posteriors"].items():
            assert "alpha" in ab
            assert "beta" in ab
            assert "theta" in ab
            assert 0.0 <= ab["theta"] <= 1.0


# ---------------------------------------------------------------------------
# /api/update
# ---------------------------------------------------------------------------

class TestUpdate:
    def test_update_returns_200(self) -> None:
        r = client.post("/api/update", json={"arm": "contrarian", "tail": 1, "leads": 0})
        assert r.status_code == 200

    def test_update_schema(self) -> None:
        r = client.post("/api/update", json={"arm": "data", "tail": 0, "leads": 0})
        body = r.json()
        assert "reward" in body
        assert "bandit_updated" in body
        assert "posteriors" in body

    def test_update_reward_in_range(self) -> None:
        r = client.post("/api/update", json={"arm": "question", "tail": 1, "leads": 2})
        assert 0.0 <= r.json()["reward"] <= 1.0

    def test_update_invalid_arm_rejected(self) -> None:
        r = client.post("/api/update", json={"arm": "unknown_arm", "tail": 1, "leads": 0})
        assert r.status_code == 422

    def test_update_tail_out_of_range_rejected(self) -> None:
        r = client.post("/api/update", json={"arm": "contrarian", "tail": 5, "leads": 0})
        assert r.status_code == 422

    def test_update_reward_increases_with_leads(self) -> None:
        r0 = client.post("/api/update", json={"arm": "statement", "tail": 0, "leads": 0})
        r1 = client.post("/api/update", json={"arm": "statement", "tail": 0, "leads": 3})
        assert r1.json()["reward"] > r0.json()["reward"]

    def test_update_bandit_updated_true(self) -> None:
        r = client.post("/api/update", json={"arm": "data", "tail": 1, "leads": 1})
        assert r.json()["bandit_updated"] is True
