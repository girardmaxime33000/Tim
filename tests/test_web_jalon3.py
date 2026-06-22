"""Tests for /api/monitor, /api/leads, /api/backtest — Jalon 3."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from growth_system.web.api import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# /api/monitor
# ---------------------------------------------------------------------------

class TestMonitor:
    def test_monitor_returns_200(self) -> None:
        r = client.get("/api/monitor")
        assert r.status_code == 200

    def test_monitor_schema(self) -> None:
        body = client.get("/api/monitor").json()
        for key in ("dates", "new_followers_smooth", "impressions",
                    "alarms", "retrain_recommended", "totals"):
            assert key in body, f"missing key: {key}"

    def test_monitor_dates_non_empty(self) -> None:
        body = client.get("/api/monitor").json()
        assert len(body["dates"]) > 0

    def test_monitor_series_same_length(self) -> None:
        body = client.get("/api/monitor").json()
        n = len(body["dates"])
        assert len(body["new_followers_smooth"]) == n
        assert len(body["impressions"]) == n

    def test_monitor_alarms_schema(self) -> None:
        body = client.get("/api/monitor").json()
        for alarm in body["alarms"]:
            assert "date" in alarm
            assert "direction" in alarm
            assert alarm["direction"] in ("up", "down")
            assert "statistic" in alarm

    def test_monitor_retrain_is_bool(self) -> None:
        body = client.get("/api/monitor").json()
        assert isinstance(body["retrain_recommended"], bool)

    def test_monitor_totals_schema(self) -> None:
        body = client.get("/api/monitor").json()
        totals = body["totals"]
        assert "impressions" in totals
        assert "new_followers" in totals
        assert "avg_daily_impressions" in totals

    def test_monitor_smooth_values_positive(self) -> None:
        body = client.get("/api/monitor").json()
        assert all(v >= 0 for v in body["new_followers_smooth"])


# ---------------------------------------------------------------------------
# /api/leads  GET
# ---------------------------------------------------------------------------

class TestLeadsGet:
    def test_leads_get_returns_200(self) -> None:
        r = client.get("/api/leads")
        assert r.status_code == 200

    def test_leads_get_schema(self) -> None:
        body = client.get("/api/leads").json()
        assert "leads" in body
        assert "total_leads" in body

    def test_leads_total_non_negative(self) -> None:
        body = client.get("/api/leads").json()
        assert body["total_leads"] >= 0

    def test_leads_items_schema(self) -> None:
        body = client.get("/api/leads").json()
        for item in body["leads"]:
            assert "post_id" in item
            assert "qualified_contacts" in item
            assert item["qualified_contacts"] >= 0


# ---------------------------------------------------------------------------
# /api/leads  POST
# ---------------------------------------------------------------------------

class TestLeadsPost:
    def test_leads_post_returns_200(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import growth_system.web.api as api_mod
        monkeypatch.setattr(api_mod, "LEADS_CSV", tmp_path / "leads.csv")
        monkeypatch.setattr(api_mod, "DATA_DIR", tmp_path)
        r = client.post("/api/leads", json={"post_id": "p1", "qualified_contacts": 3})
        assert r.status_code == 200

    def test_leads_post_schema(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import growth_system.web.api as api_mod
        monkeypatch.setattr(api_mod, "LEADS_CSV", tmp_path / "leads.csv")
        monkeypatch.setattr(api_mod, "DATA_DIR", tmp_path)
        body = client.post("/api/leads", json={"post_id": "p1", "qualified_contacts": 2}).json()
        assert "post_id" in body
        assert "qualified_contacts" in body
        assert "action" in body

    def test_leads_post_creates_entry(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import growth_system.web.api as api_mod
        leads_csv = tmp_path / "leads.csv"
        monkeypatch.setattr(api_mod, "LEADS_CSV", leads_csv)
        monkeypatch.setattr(api_mod, "DATA_DIR", tmp_path)
        body = client.post("/api/leads", json={"post_id": "mypost", "qualified_contacts": 5}).json()
        assert body["action"] == "created"
        assert leads_csv.exists()

    def test_leads_post_upserts(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import growth_system.web.api as api_mod
        monkeypatch.setattr(api_mod, "LEADS_CSV", tmp_path / "leads.csv")
        monkeypatch.setattr(api_mod, "DATA_DIR", tmp_path)
        client.post("/api/leads", json={"post_id": "p1", "qualified_contacts": 1})
        body = client.post("/api/leads", json={"post_id": "p1", "qualified_contacts": 4}).json()
        assert body["action"] == "updated"
        assert body["qualified_contacts"] == 4

    def test_leads_post_negative_rejected(self) -> None:
        r = client.post("/api/leads", json={"post_id": "p1", "qualified_contacts": -1})
        assert r.status_code == 422

    def test_leads_roundtrip(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import growth_system.web.api as api_mod
        monkeypatch.setattr(api_mod, "LEADS_CSV", tmp_path / "leads.csv")
        monkeypatch.setattr(api_mod, "DATA_DIR", tmp_path)
        client.post("/api/leads", json={"post_id": "post-A", "qualified_contacts": 7})
        # Upsert and read back via GET is handled by the same LEADS_CSV path
        df = pd.read_csv(tmp_path / "leads.csv")
        assert len(df) == 1
        assert int(df.iloc[0]["qualified_contacts"]) == 7


# ---------------------------------------------------------------------------
# /api/backtest
# ---------------------------------------------------------------------------

class TestBacktest:
    def test_backtest_returns_200(self) -> None:
        r = client.post("/api/backtest")
        assert r.status_code == 200

    def test_backtest_schema(self) -> None:
        body = client.post("/api/backtest").json()
        for key in ("total_reward_P0", "total_reward_P1", "total_reward_P2",
                    "regret_P2_vs_P0", "changepoints_detected",
                    "p2_beats_p1", "p1_beats_p0",
                    "dates", "cum_reward_P0", "cum_reward_P1", "cum_reward_P2",
                    "regret", "posteriors_over_time", "leads_csv_empty"):
            assert key in body, f"missing key: {key}"

    def test_backtest_rewards_positive(self) -> None:
        body = client.post("/api/backtest").json()
        assert body["total_reward_P0"] >= 0
        assert body["total_reward_P1"] >= 0
        assert body["total_reward_P2"] >= 0

    def test_backtest_series_non_empty(self) -> None:
        body = client.post("/api/backtest").json()
        assert len(body["dates"]) > 0
        assert len(body["cum_reward_P2"]) == len(body["dates"])

    def test_backtest_regret_consistent(self) -> None:
        body = client.post("/api/backtest").json()
        # regret = P0 - P2; final value should match scalar
        final_regret = body["regret"][-1]
        assert abs(final_regret - body["regret_P2_vs_P0"]) < 0.01

    def test_backtest_posteriors_arms(self) -> None:
        body = client.post("/api/backtest").json()
        if body["posteriors_over_time"]:
            first = body["posteriors_over_time"][0]
            for arm in ("contrarian", "data", "question", "statement"):
                assert arm in first

    def test_backtest_changepoints_list(self) -> None:
        body = client.post("/api/backtest").json()
        assert isinstance(body["changepoints_detected"], list)

    def test_backtest_plot_created(self) -> None:
        client.post("/api/backtest")
        from growth_system.web.api import DATA_DIR
        assert (DATA_DIR / "backtest_results.png").exists()

    def test_backtest_leads_empty_flag(self) -> None:
        body = client.post("/api/backtest").json()
        # In test environment leads.csv may or may not exist — flag should be bool
        assert isinstance(body["leads_csv_empty"], bool)
