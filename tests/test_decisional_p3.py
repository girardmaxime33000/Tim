"""Tests Priority 3 — score_log, rework_rate, CUSUM growth bands,
precision_trend figure."""
from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from growth_system.web import api as api_mod
from growth_system.web.api import app

client = TestClient(app)

_FIXTURE_CSV = (
    Path(__file__).parent / "fixtures"
    / "LI_POSTS_urn_li_fsd_profile_ACoAACkBeK0BOXez5sioKmu4nCwSCpGC4fqMteU_252.csv"
)


# ---------------------------------------------------------------------------
# §4 — score_log.csv
# ---------------------------------------------------------------------------

class TestScoreLog:
    def test_score_appends_to_log(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        log_path = tmp_path / "score_log.csv"
        monkeypatch.setattr(api_mod, "SCORE_LOG_CSV", log_path)
        monkeypatch.setattr(api_mod, "DATA_DIR", tmp_path)

        client.post("/api/score", json={"text": "La résilience est une compétence.", "format": "Text only"})
        assert log_path.exists()
        df = pd.read_csv(log_path)
        assert len(df) == 1

    def test_score_log_schema(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        log_path = tmp_path / "score_log.csv"
        monkeypatch.setattr(api_mod, "SCORE_LOG_CSV", log_path)
        monkeypatch.setattr(api_mod, "DATA_DIR", tmp_path)

        client.post("/api/score", json={"text": "Et si tout changeait ?", "format": "Image"})
        df = pd.read_csv(log_path)
        for col in ("timestamp", "text_excerpt", "p_queue", "verdict", "format"):
            assert col in df.columns, f"missing column: {col}"

    def test_score_log_excerpt_truncated(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        log_path = tmp_path / "score_log.csv"
        monkeypatch.setattr(api_mod, "SCORE_LOG_CSV", log_path)
        monkeypatch.setattr(api_mod, "DATA_DIR", tmp_path)

        long_text = "X" * 300
        client.post("/api/score", json={"text": long_text, "format": "Text only"})
        df = pd.read_csv(log_path)
        assert df.iloc[0]["text_excerpt"] == "X" * 160

    def test_score_log_accumulates(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        log_path = tmp_path / "score_log.csv"
        monkeypatch.setattr(api_mod, "SCORE_LOG_CSV", log_path)
        monkeypatch.setattr(api_mod, "DATA_DIR", tmp_path)

        for text in ["Post 1", "Post 2", "Post 3"]:
            client.post("/api/score", json={"text": text, "format": "Text only"})
        df = pd.read_csv(log_path)
        assert len(df) == 3

    def test_score_log_separate_from_posts_csv(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """score_log.csv must never be at the same path as posts.csv."""
        assert api_mod.SCORE_LOG_CSV != api_mod.POSTS_CSV

    def test_monitor_has_scoring_recent(self) -> None:
        body = client.get("/api/monitor").json()
        assert "scoring_recent" in body
        assert isinstance(body["scoring_recent"], list)

    def test_monitor_has_rework_rate(self) -> None:
        body = client.get("/api/monitor").json()
        assert "rework_rate_30d" in body
        assert 0.0 <= body["rework_rate_30d"] <= 1.0

    def test_rework_rate_correct(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """2 rework out of 4 in last 30d → 0.5."""
        import datetime
        log_path = tmp_path / "score_log.csv"
        now = datetime.datetime.now()
        rows = [
            {"timestamp": (now - datetime.timedelta(days=1)).isoformat(timespec="seconds"),
             "text_excerpt": "A", "p_queue": 0.1, "verdict": "rework", "format": "Text only"},
            {"timestamp": (now - datetime.timedelta(days=2)).isoformat(timespec="seconds"),
             "text_excerpt": "B", "p_queue": 0.2, "verdict": "rework", "format": "Text only"},
            {"timestamp": (now - datetime.timedelta(days=3)).isoformat(timespec="seconds"),
             "text_excerpt": "C", "p_queue": 0.6, "verdict": "publish", "format": "Text only"},
            {"timestamp": (now - datetime.timedelta(days=4)).isoformat(timespec="seconds"),
             "text_excerpt": "D", "p_queue": 0.4, "verdict": "optimize", "format": "Text only"},
        ]
        pd.DataFrame(rows).to_csv(log_path, index=False)
        monkeypatch.setattr(api_mod, "SCORE_LOG_CSV", log_path)

        body = client.get("/api/monitor").json()
        assert abs(body["rework_rate_30d"] - 0.5) < 0.01

    def test_scoring_recent_max_10(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """scoring_recent must return at most 10 rows."""
        import datetime
        log_path = tmp_path / "score_log.csv"
        now = datetime.datetime.now()
        rows = [
            {"timestamp": (now - datetime.timedelta(days=i)).isoformat(timespec="seconds"),
             "text_excerpt": f"T{i}", "p_queue": 0.3, "verdict": "optimize", "format": "Text only"}
            for i in range(15)
        ]
        pd.DataFrame(rows).to_csv(log_path, index=False)
        monkeypatch.setattr(api_mod, "SCORE_LOG_CSV", log_path)

        body = client.get("/api/monitor").json()
        assert len(body["scoring_recent"]) <= 10


# ---------------------------------------------------------------------------
# §5 — CUSUM growth bands
# ---------------------------------------------------------------------------

class TestGrowthBands:
    def test_growth_figure_present(self) -> None:
        body = client.get("/api/monitor/figures").json()
        assert "growth" in body

    def test_growth_figure_has_shapes_if_alarm_exists(self) -> None:
        """If the real dataset has CUSUM alarms, growth figure must have shapes."""
        import plotly.io as pio
        # Check if alarms detected on real data
        monitor = client.get("/api/monitor").json()
        if not monitor["alarms"]:
            pytest.skip("no alarms on this dataset")
        up_alarms = [a for a in monitor["alarms"] if a["direction"] == "up"]
        if not up_alarms:
            pytest.skip("no 'up' alarms — no bands expected")
        body = client.get("/api/monitor/figures").json()
        fig = pio.from_json(body["growth"])
        shapes = fig.layout.shapes or []
        assert len(shapes) > 0, "expected vrect bands for acceleration phases"


# ---------------------------------------------------------------------------
# §6 — precision_trend figure
# ---------------------------------------------------------------------------

class TestPrecisionTrend:
    def test_precision_trend_key_present(self) -> None:
        body = client.get("/api/monitor/figures").json()
        assert "precision_trend" in body

    def test_precision_trend_valid_plotly(self) -> None:
        import plotly.io as pio
        body = client.get("/api/monitor/figures").json()
        fig = pio.from_json(body["precision_trend"])
        assert hasattr(fig, "data")

    def test_precision_log_written_on_ingest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        if not _FIXTURE_CSV.exists():
            pytest.skip("fixture CSV not present")

        monkeypatch.setattr(api_mod, "POSTS_CSV", tmp_path / "posts.csv")
        monkeypatch.setattr(api_mod, "SNAPSHOT_DIR", tmp_path / "snap")
        monkeypatch.setattr(api_mod, "LAST_INGEST_JSON", tmp_path / "last_ingest.json")
        monkeypatch.setattr(api_mod, "PRECISION_LOG_CSV", tmp_path / "precision_log.csv")
        monkeypatch.setattr(api_mod, "DATA_DIR", tmp_path)
        monkeypatch.setattr(api_mod, "_scorer", None)

        csv_bytes = _FIXTURE_CSV.read_bytes()
        r = client.post(
            "/api/ingest",
            files={"file": ("export.csv", io.BytesIO(csv_bytes), "text/csv")},
            data={"dry_run": "false"},
        )
        assert r.status_code == 200
        pr_path = tmp_path / "precision_log.csv"
        assert pr_path.exists(), "precision_log.csv not created"
        df = pd.read_csv(pr_path)
        assert "precision_at_20" in df.columns
        assert len(df) == 1

    def test_precision_trend_with_two_points(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pr_path = tmp_path / "precision_log.csv"
        pd.DataFrame([
            {"timestamp": "2026-05-01T10:00:00", "precision_at_20": 0.62, "n_posts": 100},
            {"timestamp": "2026-06-01T10:00:00", "precision_at_20": 0.68, "n_posts": 133},
        ]).to_csv(pr_path, index=False)
        monkeypatch.setattr(api_mod, "PRECISION_LOG_CSV", pr_path)

        import plotly.io as pio
        body = client.get("/api/monitor/figures").json()
        fig = pio.from_json(body["precision_trend"])
        assert len(fig.data) == 1
        assert len(fig.data[0].y) == 2

    def test_precision_trend_single_point(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pr_path = tmp_path / "precision_log.csv"
        pd.DataFrame([
            {"timestamp": "2026-06-01T10:00:00", "precision_at_20": 0.654, "n_posts": 133},
        ]).to_csv(pr_path, index=False)
        monkeypatch.setattr(api_mod, "PRECISION_LOG_CSV", pr_path)

        import plotly.io as pio
        body = client.get("/api/monitor/figures").json()
        fig = pio.from_json(body["precision_trend"])
        assert len(fig.data) == 1

    def test_precision_log_separate_from_posts_csv(self) -> None:
        assert api_mod.PRECISION_LOG_CSV != api_mod.POSTS_CSV
        assert api_mod.PRECISION_LOG_CSV != api_mod.SCORE_LOG_CSV
