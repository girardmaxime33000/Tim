"""Tests for POST /api/ingest — Jalon 2.

CRITICAL: tests never write to the real data/posts.csv.
All writes go to temp directories.
"""
from __future__ import annotations

import io
import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from growth_system.web.ingest import (
    IngestError,
    _parse_days_ago,
    load_raw,
    normalize,
    ingest_to_path,
)

# ---------------------------------------------------------------------------
# Fixture path
# ---------------------------------------------------------------------------

FIXTURE_CSV = (
    Path(__file__).parent
    / "fixtures"
    / "LI_POSTS_urn_li_fsd_profile_ACoAACkBeK0BOXez5sioKmu4nCwSCpGC4fqMteU_252.csv"
)

AUTHOR = "Timothée Roy"


# ---------------------------------------------------------------------------
# Unit: _parse_days_ago
# ---------------------------------------------------------------------------

class TestParseDaysAgo:
    def test_days(self) -> None:
        assert _parse_days_ago("1d") == 1
        assert _parse_days_ago("5d") == 5

    def test_weeks(self) -> None:
        assert _parse_days_ago("2w") == 14
        assert _parse_days_ago("3w") == 21

    def test_months(self) -> None:
        assert _parse_days_ago("1mo") == 30
        assert _parse_days_ago("4mo") == 120

    def test_years(self) -> None:
        assert _parse_days_ago("1yr") == 365
        assert _parse_days_ago("7yr") == 2555

    def test_nan(self) -> None:
        import math
        assert _parse_days_ago(float("nan")) is None

    def test_invalid(self) -> None:
        assert _parse_days_ago("yesterday") is None


# ---------------------------------------------------------------------------
# Unit: load_raw
# ---------------------------------------------------------------------------

class TestLoadRaw:
    def test_csv_loads(self) -> None:
        content = FIXTURE_CSV.read_bytes()
        df = load_raw(content, "export.csv")
        assert len(df) == 252

    def test_unsupported_extension(self) -> None:
        with pytest.raises(IngestError, match="non supporté"):
            load_raw(b"data", "file.txt")

    def test_json_loads(self) -> None:
        # Build a minimal JSON that can be round-tripped
        data = [{"permalink": "a", "text": "hello", "likes": 1,
                  "comments": 0, "shares": 0, "publishDate": "1d"}]
        content = json.dumps(data).encode()
        df = load_raw(content, "export.json")
        assert len(df) == 1


# ---------------------------------------------------------------------------
# Unit: normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    @pytest.fixture
    def raw(self) -> pd.DataFrame:
        return pd.read_csv(FIXTURE_CSV)

    def test_filters_author(self, raw: pd.DataFrame) -> None:
        df = normalize(raw, author_filter=AUTHOR)
        assert len(df) >= 100  # at least 100 posts for Timothée Roy

    def test_eng_score_computed(self, raw: pd.DataFrame) -> None:
        df = normalize(raw, author_filter=AUTHOR)
        row = df.iloc[0]
        expected = row["likes"] + 3 * row["comments"] + 2 * row["shares"]
        assert row["eng_score"] == expected

    def test_days_ago_present(self, raw: pd.DataFrame) -> None:
        df = normalize(raw, author_filter=AUTHOR)
        assert df["days_ago"].notna().any()
        assert (df["days_ago"] > 0).all()

    def test_format_column(self, raw: pd.DataFrame) -> None:
        df = normalize(raw, author_filter=AUTHOR)
        assert "format" in df.columns
        assert set(df["format"].unique()).issubset(
            {"Text only", "Image", "Video", "Document-Carousel"}
        )

    def test_no_duplicate_permalinks(self, raw: pd.DataFrame) -> None:
        df = normalize(raw, author_filter=AUTHOR)
        assert df["permalink"].nunique() == len(df)

    def test_output_columns(self, raw: pd.DataFrame) -> None:
        df = normalize(raw, author_filter=AUTHOR)
        for col in ("permalink", "text", "eng_score", "days_ago", "format"):
            assert col in df.columns

    def test_missing_required_column_raises(self) -> None:
        df = pd.DataFrame([{"text": "x", "likes": 1}])
        with pytest.raises(IngestError, match="Colonnes manquantes"):
            normalize(df)

    def test_too_few_posts_raises(self) -> None:
        # Build a minimal valid-schema df with only 5 rows
        rows = [
            {"permalink": f"p{i}", "text": "x" * 200, "likes": 0,
             "comments": 0, "shares": 0, "publishDate": "1d"}
            for i in range(5)
        ]
        df = pd.DataFrame(rows)
        with pytest.raises(IngestError, match="minimum requis"):
            normalize(df)

    def test_truncation_guard_raises(self) -> None:
        # 60 rows but all text < 10 chars (truncated)
        rows = [
            {"permalink": f"p{i}", "text": "hi", "likes": 0,
             "comments": 0, "shares": 0, "publishDate": "1d"}
            for i in range(60)
        ]
        df = pd.DataFrame(rows)
        with pytest.raises(IngestError, match="tronqué"):
            normalize(df)

    def test_deduplication(self) -> None:
        # Duplicate permalink should be deduplicated
        rows = [
            {"permalink": "same", "text": "x" * 500, "likes": 1,
             "comments": 0, "shares": 0, "publishDate": "1d"},
        ] * 60  # 60 identical rows
        df = pd.DataFrame(rows)
        # After dedup: 1 post → < MIN_POSTS → should raise
        with pytest.raises(IngestError, match="minimum requis"):
            normalize(df)


# ---------------------------------------------------------------------------
# Unit: ingest_to_path (never touches real posts.csv)
# ---------------------------------------------------------------------------

class TestIngestToPath:
    def test_writes_to_temp_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "posts.csv"
            snap_dir = Path(tmp) / "snapshots"
            content = FIXTURE_CSV.read_bytes()
            summary = ingest_to_path(
                content=content,
                filename="export.csv",
                dest=dest,
                snapshot_dir=snap_dir,
                author_filter=AUTHOR,
            )
            assert dest.exists()
            assert summary["n_posts"] >= 100
            assert summary["tail_rate"] >= 0.0
            assert summary["snapshot"] is None  # no prior file to snapshot

    def test_snapshot_created_on_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "posts.csv"
            snap_dir = Path(tmp) / "snapshots"
            content = FIXTURE_CSV.read_bytes()
            # First write
            ingest_to_path(content=content, filename="export.csv",
                           dest=dest, snapshot_dir=snap_dir, author_filter=AUTHOR)
            # Second write → should create snapshot
            summary = ingest_to_path(content=content, filename="export.csv",
                                      dest=dest, snapshot_dir=snap_dir, author_filter=AUTHOR)
            assert summary["snapshot"] is not None
            assert Path(summary["snapshot"]).exists()

    def test_atomic_write_no_tmp_left(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "posts.csv"
            content = FIXTURE_CSV.read_bytes()
            ingest_to_path(content=content, filename="export.csv",
                           dest=dest, author_filter=AUTHOR)
            tmp_file = dest.with_suffix(".tmp")
            assert not tmp_file.exists()

    def test_guardrail_error_does_not_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "posts.csv"
            # Tiny CSV — will fail MIN_POSTS guardrail
            tiny = pd.DataFrame([
                {"permalink": "x", "text": "hello", "likes": 0,
                 "comments": 0, "shares": 0, "publishDate": "1d"}
            ])
            content = tiny.to_csv(index=False).encode()
            with pytest.raises(IngestError):
                ingest_to_path(content=content, filename="bad.csv", dest=dest)
            assert not dest.exists()


# ---------------------------------------------------------------------------
# Integration: POST /api/ingest via TestClient
# ---------------------------------------------------------------------------

@pytest.fixture
def patched_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """TestClient with POSTS_CSV and SNAPSHOT_DIR redirected to tmp_path."""
    import growth_system.web.api as api_mod

    monkeypatch.setattr(api_mod, "POSTS_CSV", tmp_path / "posts.csv")
    monkeypatch.setattr(api_mod, "SNAPSHOT_DIR", tmp_path / "snapshots")

    from fastapi.testclient import TestClient
    return TestClient(api_mod.app)


class TestIngestEndpoint:
    def test_ingest_returns_200(self, patched_client: TestClient) -> None:
        content = FIXTURE_CSV.read_bytes()
        r = patched_client.post(
            "/api/ingest",
            files={"file": ("export.csv", content, "text/csv")},
            data={"author": AUTHOR},
        )
        assert r.status_code == 200

    def test_ingest_schema(self, patched_client: TestClient) -> None:
        content = FIXTURE_CSV.read_bytes()
        r = patched_client.post(
            "/api/ingest",
            files={"file": ("export.csv", content, "text/csv")},
            data={"author": AUTHOR},
        )
        body = r.json()
        for key in ("n_raw", "n_posts", "duplicates_removed",
                    "tail_rate", "date_range", "destination"):
            assert key in body, f"missing key: {key}"

    def test_ingest_n_posts(self, patched_client: TestClient) -> None:
        content = FIXTURE_CSV.read_bytes()
        r = patched_client.post(
            "/api/ingest",
            files={"file": ("export.csv", content, "text/csv")},
            data={"author": AUTHOR},
        )
        assert r.json()["n_posts"] >= 100

    def test_ingest_tail_rate_in_range(self, patched_client: TestClient) -> None:
        content = FIXTURE_CSV.read_bytes()
        r = patched_client.post(
            "/api/ingest",
            files={"file": ("export.csv", content, "text/csv")},
            data={"author": AUTHOR},
        )
        assert 0.0 <= r.json()["tail_rate"] <= 1.0

    def test_ingest_duplicate_removed(self, patched_client: TestClient) -> None:
        content = FIXTURE_CSV.read_bytes()
        r = patched_client.post(
            "/api/ingest",
            files={"file": ("export.csv", content, "text/csv")},
            data={"author": AUTHOR},
        )
        body = r.json()
        assert body["n_raw"] > body["n_posts"] or body["duplicates_removed"] >= 0

    def test_ingest_dry_run_does_not_write(
        self, patched_client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import growth_system.web.api as api_mod
        dest = tmp_path / "posts.csv"
        monkeypatch.setattr(api_mod, "POSTS_CSV", dest)
        content = FIXTURE_CSV.read_bytes()
        r = patched_client.post(
            "/api/ingest",
            files={"file": ("export.csv", content, "text/csv")},
            data={"author": AUTHOR, "dry_run": "true"},
        )
        assert r.status_code == 200
        assert not dest.exists()
        assert "dry-run" in r.json()["destination"]

    def test_ingest_bad_extension_422(self, patched_client: TestClient) -> None:
        r = patched_client.post(
            "/api/ingest",
            files={"file": ("export.txt", b"data", "text/plain")},
        )
        assert r.status_code == 422

    def test_ingest_too_few_posts_422(self, patched_client: TestClient) -> None:
        tiny = pd.DataFrame([
            {"permalink": "x", "text": "hello", "likes": 0,
             "comments": 0, "shares": 0, "publishDate": "1d"}
        ])
        content = tiny.to_csv(index=False).encode()
        r = patched_client.post(
            "/api/ingest",
            files={"file": ("tiny.csv", content, "text/csv")},
        )
        assert r.status_code == 422

    def test_ingest_writes_valid_csv(self, patched_client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import growth_system.web.api as api_mod
        dest = tmp_path / "posts.csv"
        monkeypatch.setattr(api_mod, "POSTS_CSV", dest)
        monkeypatch.setattr(api_mod, "SNAPSHOT_DIR", tmp_path / "snapshots")
        content = FIXTURE_CSV.read_bytes()
        patched_client.post(
            "/api/ingest",
            files={"file": ("export.csv", content, "text/csv")},
            data={"author": AUTHOR},
        )
        df = pd.read_csv(dest)
        assert "eng_score" in df.columns
        assert "days_ago" in df.columns
        assert "format" in df.columns
