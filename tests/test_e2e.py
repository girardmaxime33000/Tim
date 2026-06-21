"""Test E2E léger — Jalon 7.

Démarre le serveur FastAPI via TestClient (pas de processus séparé),
et enchaîne le flux complet : ingest → score → recommend → update →
monitor → leads → backtest — tout via HTTP.

Toutes les écritures de fichiers sont redirigées vers tmp_path :
le vrai data/posts.csv n'est jamais touché.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

FIXTURE_CSV = (
    Path(__file__).parent
    / "fixtures"
    / "LI_POSTS_urn_li_fsd_profile_ACoAACkBeK0BOXez5sioKmu4nCwSCpGC4fqMteU_252.csv"
)
AUTHOR = "Timothée Roy"

SAMPLE_DRAFT = (
    "Et si les galeries d'art avaient tort depuis 30 ans ?\n\n"
    "Voici ce que les données montrent sur le marché secondaire :\n"
    "- Artistes émergents : +340 % en 5 ans\n"
    "- Maisons classiques : stagnation\n\n"
    "Ce que ça change pour les collectionneurs ? Tout.\n"
    "#art #investissement"
)


@pytest.fixture
def e2e_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """TestClient with all file paths redirected to tmp_path."""
    import growth_system.web.api as api_mod

    monkeypatch.setattr(api_mod, "POSTS_CSV", tmp_path / "posts.csv")
    monkeypatch.setattr(api_mod, "LEADS_CSV", tmp_path / "leads.csv")
    monkeypatch.setattr(api_mod, "STATE_JSON", tmp_path / "growth_state.json")
    monkeypatch.setattr(api_mod, "SNAPSHOT_DIR", tmp_path / "snapshots")
    monkeypatch.setattr(api_mod, "DATA_DIR", tmp_path)
    # Reset singletons so they don't carry state from other tests
    monkeypatch.setattr(api_mod, "_scorer", None)
    monkeypatch.setattr(api_mod, "_bandit", None)

    return TestClient(api_mod.app)


class TestE2EFlow:
    def test_health_before_ingest(self, e2e_client: TestClient) -> None:
        """API responds even without posts.csv — status degraded, not crash."""
        r = e2e_client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] in ("ok", "degraded")

    def test_ingest_fixture(self, e2e_client: TestClient) -> None:
        """Ingest the real fixture CSV and validate the summary."""
        content = FIXTURE_CSV.read_bytes()
        r = e2e_client.post(
            "/api/ingest",
            files={"file": ("export.csv", content, "text/csv")},
            data={"author": AUTHOR},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["n_posts"] >= 100
        assert body["n_new"] >= 100   # first ingest: all posts are new
        assert 0.0 <= body["tail_rate"] <= 1.0
        assert body["snapshot"] is None  # no prior file to snapshot

    def test_score_after_ingest(self, e2e_client: TestClient, tmp_path: Path) -> None:
        """Score a draft after ingest — model must be loaded."""
        # First ingest
        content = FIXTURE_CSV.read_bytes()
        e2e_client.post(
            "/api/ingest",
            files={"file": ("export.csv", content, "text/csv")},
            data={"author": AUTHOR},
        )
        # Now score
        r = e2e_client.post("/api/score", json={"text": SAMPLE_DRAFT})
        assert r.status_code == 200
        body = r.json()
        assert 0.0 <= body["p_queue"] <= 1.0
        assert body["verdict"] in ("publish", "optimize", "rework")
        assert len(body["contributions"]) == 8

    def test_recommend(self, e2e_client: TestClient) -> None:
        """Bandit recommends a valid arm."""
        r = e2e_client.get("/api/recommend")
        assert r.status_code == 200
        assert r.json()["recommended_arm"] in ("contrarian", "data", "question", "statement")

    def test_update_bandit(self, e2e_client: TestClient, tmp_path: Path) -> None:
        """Update bandit with a tail=1 result and verify state is persisted."""
        r = e2e_client.post(
            "/api/update",
            json={"arm": "contrarian", "tail": 1, "leads": 2},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["bandit_updated"] is True
        assert 0.0 <= body["reward"] <= 1.0
        # State file must exist after update
        state_file = tmp_path / "growth_state.json"
        assert state_file.exists()
        state = json.loads(state_file.read_text())
        assert "bandit" in state

    def test_second_ingest_merges(self, e2e_client: TestClient) -> None:
        """A second ingest of the same file must not shrink the corpus."""
        content = FIXTURE_CSV.read_bytes()
        r1 = e2e_client.post(
            "/api/ingest",
            files={"file": ("export.csv", content, "text/csv")},
            data={"author": AUTHOR},
        )
        n1 = r1.json()["n_posts"]
        r2 = e2e_client.post(
            "/api/ingest",
            files={"file": ("export.csv", content, "text/csv")},
            data={"author": AUTHOR},
        )
        assert r2.status_code == 200
        assert r2.json()["n_posts"] >= n1
        assert r2.json()["snapshot"] is not None  # snapshot created on second write

    def test_monitor(self, e2e_client: TestClient) -> None:
        """Monitor endpoint returns series and alarm list."""
        r = e2e_client.get("/api/monitor")
        assert r.status_code == 200
        body = r.json()
        assert len(body["dates"]) > 0
        assert isinstance(body["alarms"], list)

    def test_leads_crud(self, e2e_client: TestClient) -> None:
        """Create a lead, read it back, update it."""
        # Create
        r = e2e_client.post("/api/leads", json={"post_id": "post-e2e", "qualified_contacts": 3})
        assert r.status_code == 200
        assert r.json()["action"] == "created"

        # Read back
        r = e2e_client.get("/api/leads")
        items = r.json()["leads"]
        assert any(l["post_id"] == "post-e2e" and l["qualified_contacts"] == 3 for l in items)

        # Update
        r = e2e_client.post("/api/leads", json={"post_id": "post-e2e", "qualified_contacts": 7})
        assert r.json()["action"] == "updated"
        assert r.json()["qualified_contacts"] == 7

    def test_backtest_after_ingest(self, e2e_client: TestClient) -> None:
        """Full backtest after ingest returns valid curves."""
        content = FIXTURE_CSV.read_bytes()
        e2e_client.post(
            "/api/ingest",
            files={"file": ("export.csv", content, "text/csv")},
            data={"author": AUTHOR},
        )
        r = e2e_client.post("/api/backtest")
        assert r.status_code == 200
        body = r.json()
        assert body["total_reward_P0"] >= 0
        assert len(body["dates"]) > 0
        assert len(body["cum_reward_P2"]) == len(body["dates"])
        # Final regret must match scalar
        assert abs(body["regret"][-1] - body["regret_P2_vs_P0"]) < 0.02

    def test_full_workflow_sequence(self, e2e_client: TestClient) -> None:
        """Smoke test: all endpoints called in order without error."""
        content = FIXTURE_CSV.read_bytes()

        # 1. Ingest
        r = e2e_client.post("/api/ingest",
                             files={"file": ("export.csv", content, "text/csv")},
                             data={"author": AUTHOR})
        assert r.status_code == 200

        # 2. Health — should be ok now
        r = e2e_client.get("/api/health")
        assert r.json()["status"] == "ok"

        # 3. Profile
        r = e2e_client.get("/api/profile")
        assert r.json()["posts_count"] >= 100

        # 4. Score
        r = e2e_client.post("/api/score", json={"text": SAMPLE_DRAFT})
        assert r.json()["verdict"] in ("publish", "optimize", "rework")

        # 5. Recommend
        arm = e2e_client.get("/api/recommend").json()["recommended_arm"]
        assert arm in ("contrarian", "data", "question", "statement")

        # 6. Update
        r = e2e_client.post("/api/update", json={"arm": arm, "tail": 1, "leads": 1})
        assert r.json()["bandit_updated"] is True

        # 7. Monitor
        r = e2e_client.get("/api/monitor")
        assert len(r.json()["dates"]) > 0

        # 8. Lead upsert
        r = e2e_client.post("/api/leads", json={"post_id": "p-smoke", "qualified_contacts": 2})
        assert r.status_code == 200

        # 9. Backtest
        r = e2e_client.post("/api/backtest")
        assert r.status_code == 200
        assert r.json()["total_reward_P2"] >= 0
