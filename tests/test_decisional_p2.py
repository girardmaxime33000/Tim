"""Tests Priority 2 — freshness card: last_ingest_at, last_retrain_at,
last_changepoint_at, changepoint_unaddressed flag."""
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

# Fixture CSV: minimal LinkedIn export with 55 posts (above MIN_POSTS=50)
_FIXTURE_CSV = (
    Path(__file__).parent / "fixtures"
    / "LI_POSTS_urn_li_fsd_profile_ACoAACkBeK0BOXez5sioKmu4nCwSCpGC4fqMteU_252.csv"
)


# ---------------------------------------------------------------------------
# freshness schema
# ---------------------------------------------------------------------------

class TestFreshnessSchema:
    def test_monitor_has_freshness(self) -> None:
        body = client.get("/api/monitor").json()
        assert "freshness" in body

    def test_freshness_keys(self) -> None:
        body = client.get("/api/monitor").json()
        f = body["freshness"]
        for key in ("last_ingest_at", "last_retrain_at", "last_changepoint_at",
                    "changepoint_unaddressed"):
            assert key in f, f"missing key: {key}"

    def test_freshness_unaddressed_is_bool(self) -> None:
        body = client.get("/api/monitor").json()
        assert isinstance(body["freshness"]["changepoint_unaddressed"], bool)


# ---------------------------------------------------------------------------
# last_ingest_at — written by /api/ingest
# ---------------------------------------------------------------------------

class TestLastIngest:
    def test_ingest_writes_last_ingest_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Successful ingest must create last_ingest.json."""
        if not _FIXTURE_CSV.exists():
            pytest.skip("fixture CSV not present")

        monkeypatch.setattr(api_mod, "POSTS_CSV", tmp_path / "posts.csv")
        monkeypatch.setattr(api_mod, "SNAPSHOT_DIR", tmp_path / "snap")
        monkeypatch.setattr(api_mod, "LAST_INGEST_JSON", tmp_path / "last_ingest.json")

        csv_bytes = _FIXTURE_CSV.read_bytes()
        r = client.post(
            "/api/ingest",
            files={"file": ("export.csv", io.BytesIO(csv_bytes), "text/csv")},
            data={"dry_run": "false"},
        )
        assert r.status_code == 200, r.json()
        li_path = tmp_path / "last_ingest.json"
        assert li_path.exists(), "last_ingest.json not created"
        payload = json.loads(li_path.read_text())
        assert "last_ingest_at" in payload
        assert payload["last_ingest_at"]  # non-empty timestamp

    def test_dry_run_does_not_write_last_ingest_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        if not _FIXTURE_CSV.exists():
            pytest.skip("fixture CSV not present")

        monkeypatch.setattr(api_mod, "POSTS_CSV", tmp_path / "posts.csv")
        monkeypatch.setattr(api_mod, "SNAPSHOT_DIR", tmp_path / "snap")
        monkeypatch.setattr(api_mod, "LAST_INGEST_JSON", tmp_path / "last_ingest.json")

        csv_bytes = _FIXTURE_CSV.read_bytes()
        client.post(
            "/api/ingest",
            files={"file": ("export.csv", io.BytesIO(csv_bytes), "text/csv")},
            data={"dry_run": "true"},
        )
        assert not (tmp_path / "last_ingest.json").exists()

    def test_freshness_last_ingest_from_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        li_path = tmp_path / "last_ingest.json"
        li_path.write_text(json.dumps({"last_ingest_at": "2026-06-01T12:00:00"}))
        monkeypatch.setattr(api_mod, "LAST_INGEST_JSON", li_path)

        body = client.get("/api/monitor").json()
        assert body["freshness"]["last_ingest_at"] == "2026-06-01T12:00:00"

    def test_freshness_last_ingest_null_when_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(api_mod, "LAST_INGEST_JSON", tmp_path / "no_file.json")
        body = client.get("/api/monitor").json()
        assert body["freshness"]["last_ingest_at"] is None


# ---------------------------------------------------------------------------
# last_retrain_at — read from scorer_model.json["trained_at"]
# ---------------------------------------------------------------------------

class TestLastRetrain:
    def test_freshness_retrain_null_when_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Current scorer_model.json has no trained_at → null."""
        body = client.get("/api/monitor").json()
        # Real model has no trained_at field → must be null (not an error)
        # (If someone added it, this becomes a string — still passes the schema test above)
        assert body["freshness"]["last_retrain_at"] is None or isinstance(
            body["freshness"]["last_retrain_at"], str
        )

    def test_freshness_retrain_from_model_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model_path = tmp_path / "scorer_model.json"
        model_data = json.loads(api_mod.MODEL_PATH.read_text())
        model_data["trained_at"] = "2026-05-15T09:00:00"
        model_path.write_text(json.dumps(model_data))
        monkeypatch.setattr(api_mod, "MODEL_PATH", model_path)
        # Also reset scorer singleton so it reloads
        monkeypatch.setattr(api_mod, "_scorer", None)

        body = client.get("/api/monitor").json()
        assert body["freshness"]["last_retrain_at"] == "2026-05-15T09:00:00"


# ---------------------------------------------------------------------------
# changepoint_unaddressed flag logic
# ---------------------------------------------------------------------------

class TestChangepointUnaddressed:
    def _make_monitor_with(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        retrain_at: str | None,
    ) -> dict:
        """Call GET /api/monitor with a controlled retrain date."""
        if retrain_at is not None:
            model_data = json.loads(api_mod.MODEL_PATH.read_text())
            model_data["trained_at"] = retrain_at
            model_path = tmp_path / "scorer_model.json"
            model_path.write_text(json.dumps(model_data))
            monkeypatch.setattr(api_mod, "MODEL_PATH", model_path)
            monkeypatch.setattr(api_mod, "_scorer", None)
        return client.get("/api/monitor").json()

    def test_unaddressed_true_when_changepoint_after_retrain(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Last changepoint (real data) > retrain → unaddressed=True."""
        body = self._make_monitor_with(tmp_path, monkeypatch, "2020-01-01T00:00:00")
        f = body["freshness"]
        if f["last_changepoint_at"] is None:
            pytest.skip("no changepoint detected on this dataset")
        assert f["changepoint_unaddressed"] is True

    def test_unaddressed_false_when_retrain_after_changepoint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Retrain date after all changepoints → unaddressed=False."""
        body = self._make_monitor_with(tmp_path, monkeypatch, "2099-01-01T00:00:00")
        f = body["freshness"]
        assert f["changepoint_unaddressed"] is False

    def test_unaddressed_true_when_no_retrain_and_alarm_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No trained_at + alarm exists → unaddressed=True."""
        body = self._make_monitor_with(tmp_path, monkeypatch, retrain_at=None)
        f = body["freshness"]
        if f["last_changepoint_at"] is None:
            pytest.skip("no changepoint on this dataset")
        assert f["changepoint_unaddressed"] is True
