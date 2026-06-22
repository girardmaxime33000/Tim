"""Tests Priority 1 — bandit_live figure + leads coverage + leads by archetype."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from growth_system.web import api as api_mod
from growth_system.web.api import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# §1 — bandit_live figure
# ---------------------------------------------------------------------------

class TestBanditLiveFigure:
    def test_bandit_live_key_present(self) -> None:
        body = client.get("/api/monitor/figures").json()
        assert "bandit_live" in body, "missing key 'bandit_live'"

    def test_bandit_live_valid_plotly(self) -> None:
        import plotly.io as pio
        body = client.get("/api/monitor/figures").json()
        fig = pio.from_json(body["bandit_live"])
        assert hasattr(fig, "data")
        assert hasattr(fig, "layout")

    def test_bandit_live_four_traces(self) -> None:
        """One Beta curve per archetype."""
        import plotly.io as pio
        body = client.get("/api/monitor/figures").json()
        fig = pio.from_json(body["bandit_live"])
        assert len(fig.data) == 4

    def test_bandit_live_title_contains_maj(self) -> None:
        import plotly.io as pio
        body = client.get("/api/monitor/figures").json()
        fig = pio.from_json(body["bandit_live"])
        assert "EN DIRECT" in fig.layout.title.text

    def test_bandit_live_recommended_annotated(self) -> None:
        """Exactly one trace name contains '← recommandé'."""
        import plotly.io as pio
        body = client.get("/api/monitor/figures").json()
        fig = pio.from_json(body["bandit_live"])
        flagged = [t for t in fig.data if "recommandé" in (t.name or "")]
        assert len(flagged) == 1

    def test_bandit_live_reflects_state(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Modifying growth_state.json changes bandit_live figure."""
        state_path = tmp_path / "growth_state.json"
        # Force contrarian to dominate (alpha=50)
        state = {"bandit": {"alpha": {"contrarian": 50.0, "data": 1.0, "question": 1.0, "statement": 1.0},
                             "beta":  {"contrarian": 1.0,  "data": 10.0, "question": 10.0, "statement": 10.0},
                             "gamma": 0.985}}
        state_path.write_text(json.dumps(state))
        monkeypatch.setattr(api_mod, "STATE_JSON", state_path)
        monkeypatch.setattr(api_mod, "_bandit", None)  # force reload

        import plotly.io as pio
        body = client.get("/api/monitor/figures").json()
        fig = pio.from_json(body["bandit_live"])
        recommended = [t.name for t in fig.data if "recommandé" in (t.name or "")]
        assert len(recommended) == 1
        assert "contrarian" in recommended[0]

    def test_bandit_live_existing_keys_intact(self) -> None:
        """Adding bandit_live must not remove existing figure keys."""
        body = client.get("/api/monitor/figures").json()
        for key in ("growth", "distribution", "cusum", "bandit"):
            assert key in body, f"existing key '{key}' was removed"


# ---------------------------------------------------------------------------
# §2 — leads coverage + leads by archetype
# ---------------------------------------------------------------------------

class TestLeadsCoverage:
    def test_monitor_has_leads_coverage(self) -> None:
        body = client.get("/api/monitor").json()
        assert "leads_coverage" in body

    def test_leads_coverage_schema(self) -> None:
        body = client.get("/api/monitor").json()
        cov = body["leads_coverage"]
        assert "n_with_leads" in cov
        assert "n_total" in cov
        assert "ratio" in cov

    def test_leads_coverage_ratio_range(self) -> None:
        body = client.get("/api/monitor").json()
        r = body["leads_coverage"]["ratio"]
        assert 0.0 <= r <= 1.0

    def test_leads_coverage_ratio_correct(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """With 2 posts and 1 lead entry matching, ratio must be 0.5."""
        posts = pd.DataFrame({
            "text": ["Post A", "Post B"],
            "permalink": ["https://li/A", "https://li/B"],
            "eng_score": [50.0, 200.0],
            "days_ago": [10, 20],
            "format": ["Text only", "Text only"],
            "likes": [1, 2], "comments": [0, 0], "shares": [0, 0],
        })
        posts_path = tmp_path / "posts.csv"
        posts.to_csv(posts_path, index=False)

        leads = pd.DataFrame({"post_id": ["https://li/A"], "qualified_contacts": [3]})
        leads_path = tmp_path / "leads.csv"
        leads.to_csv(leads_path, index=False)

        monkeypatch.setattr(api_mod, "POSTS_CSV", posts_path)
        monkeypatch.setattr(api_mod, "LEADS_CSV", leads_path)

        body = client.get("/api/monitor").json()
        cov = body["leads_coverage"]
        assert cov["n_total"] == 2
        assert cov["n_with_leads"] == 1
        assert abs(cov["ratio"] - 0.5) < 0.01

    def test_leads_coverage_low_ratio_below_threshold(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """3 posts, 0 leads → ratio = 0.0 (below 0.5 threshold)."""
        posts = pd.DataFrame({
            "text": ["A", "B", "C"], "permalink": ["p1", "p2", "p3"],
            "eng_score": [10.0, 20.0, 30.0], "days_ago": [1, 2, 3],
            "format": ["Text only"] * 3, "likes": [0]*3, "comments": [0]*3, "shares": [0]*3,
        })
        posts_path = tmp_path / "posts.csv"
        posts.to_csv(posts_path, index=False)
        monkeypatch.setattr(api_mod, "POSTS_CSV", posts_path)
        monkeypatch.setattr(api_mod, "LEADS_CSV", tmp_path / "leads_none.csv")

        body = client.get("/api/monitor").json()
        assert body["leads_coverage"]["ratio"] < 0.5

    def test_leads_coverage_high_ratio_above_threshold(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """3 posts, 3 leads matching → ratio = 1.0 (above 0.5 threshold)."""
        posts = pd.DataFrame({
            "text": ["A", "B", "C"], "permalink": ["p1", "p2", "p3"],
            "eng_score": [10.0, 20.0, 30.0], "days_ago": [1, 2, 3],
            "format": ["Text only"] * 3, "likes": [0]*3, "comments": [0]*3, "shares": [0]*3,
        })
        posts_path = tmp_path / "posts.csv"
        posts.to_csv(posts_path, index=False)

        leads = pd.DataFrame({
            "post_id": ["p1", "p2", "p3"],
            "qualified_contacts": [1, 2, 3],
        })
        leads_path = tmp_path / "leads.csv"
        leads.to_csv(leads_path, index=False)

        monkeypatch.setattr(api_mod, "POSTS_CSV", posts_path)
        monkeypatch.setattr(api_mod, "LEADS_CSV", leads_path)

        body = client.get("/api/monitor").json()
        assert body["leads_coverage"]["ratio"] >= 0.5


class TestLeadsByArchetype:
    def test_monitor_has_leads_by_archetype(self) -> None:
        body = client.get("/api/monitor").json()
        assert "leads_by_archetype" in body

    def test_leads_by_archetype_schema(self) -> None:
        body = client.get("/api/monitor").json()
        for item in body["leads_by_archetype"]:
            assert "archetype" in item
            assert "tail_rate" in item
            assert "leads_total" in item
            assert "leads_per_post" in item

    def test_leads_by_archetype_all_four_arms(self) -> None:
        body = client.get("/api/monitor").json()
        arms = {item["archetype"] for item in body["leads_by_archetype"]}
        assert arms == {"contrarian", "data", "question", "statement"}

    def test_leads_by_archetype_correct_counts(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Statement post with 5 leads must appear in leads_by_archetype."""
        posts = pd.DataFrame({
            # Statement hook (no contrarian/question/data trigger)
            "text": ["La résilience est une compétence."],
            "permalink": ["https://li/S1"],
            "eng_score": [200.0],  # above ENG_Q80=98.2 → tail
            "days_ago": [5],
            "format": ["Text only"],
            "likes": [100], "comments": [30], "shares": [10],
        })
        posts_path = tmp_path / "posts.csv"
        posts.to_csv(posts_path, index=False)

        leads = pd.DataFrame({"post_id": ["https://li/S1"], "qualified_contacts": [5]})
        leads_path = tmp_path / "leads.csv"
        leads.to_csv(leads_path, index=False)

        monkeypatch.setattr(api_mod, "POSTS_CSV", posts_path)
        monkeypatch.setattr(api_mod, "LEADS_CSV", leads_path)

        body = client.get("/api/monitor").json()
        by_arm = {item["archetype"]: item for item in body["leads_by_archetype"]}
        stmt = by_arm["statement"]
        assert stmt["leads_total"] == 5
        assert stmt["tail_rate"] == 1.0  # 1 post, all in tail

    def test_leads_by_archetype_tail_rate_range(self) -> None:
        body = client.get("/api/monitor").json()
        for item in body["leads_by_archetype"]:
            assert 0.0 <= item["tail_rate"] <= 1.0

    def test_leads_by_archetype_leads_per_post_non_negative(self) -> None:
        body = client.get("/api/monitor").json()
        for item in body["leads_by_archetype"]:
            assert item["leads_per_post"] >= 0.0
