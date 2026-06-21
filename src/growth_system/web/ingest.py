"""Normalisation pipeline for LinkedIn export files (CSV / JSON / XLSX).

Never writes to the real posts.csv — callers decide the destination path.
"""
from __future__ import annotations

import hashlib
import io
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_COLS = {"permalink", "text", "likes", "comments", "shares", "publishDate"}
MIN_POSTS = 50
MEDIAN_TEXT_LEN_MIN = 160  # below → suspect truncation

_PUBLISH_DATE_RE = re.compile(
    r"^(?P<n>\d+)\s*(?P<unit>d|w|mo|yr)$", re.IGNORECASE
)


def _parse_days_ago(raw: Any, ref: date | None = None) -> int | None:
    """Convert relative strings like '1d', '2w', '3mo', '7yr' to integer days."""
    if pd.isna(raw):
        return None
    s = str(raw).strip()
    m = _PUBLISH_DATE_RE.match(s)
    if not m:
        return None
    n = int(m.group("n"))
    unit = m.group("unit").lower()
    mapping = {"d": 1, "w": 7, "mo": 30, "yr": 365}
    return n * mapping[unit]


def _derive_format(row: pd.Series) -> str:
    if pd.notna(row.get("videoUrl")) and str(row.get("videoUrl")).strip():
        return "Video"
    if pd.notna(row.get("documentUrl")) and str(row.get("documentUrl")).strip():
        return "Document-Carousel"
    if pd.notna(row.get("images")) and str(row.get("images")).strip():
        return "Image"
    return "Text only"


def _permalink_or_hash(row: pd.Series) -> str:
    pl = row.get("permalink", "")
    if pd.notna(pl) and str(pl).strip():
        return str(pl).strip()
    text = str(row.get("text", ""))
    return "hash:" + hashlib.sha256(text.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class IngestError(ValueError):
    """Raised when a guardrail rejects the upload."""


def load_raw(content: bytes, filename: str) -> pd.DataFrame:
    """Parse bytes into a raw DataFrame based on file extension."""
    ext = Path(filename).suffix.lower()
    if ext == ".csv":
        return pd.read_csv(io.BytesIO(content))
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(io.BytesIO(content))
    if ext == ".json":
        return pd.read_json(io.BytesIO(content))
    raise IngestError(f"Format non supporté : {ext}. Acceptés : csv, xlsx, json.")


def normalize(
    raw: pd.DataFrame,
    author_filter: str | None = None,
) -> pd.DataFrame:
    """
    Normalize a raw LinkedIn export DataFrame into the posts.csv schema.

    Steps:
    1. Filter by author (fullName) if provided
    2. Validate guardrails (≥50 posts, no truncation, required columns)
    3. Deduplicate by permalink / text hash
    4. Compute eng_score, days_ago, format
    5. Return cleaned DataFrame

    Raises IngestError on guardrail violation.
    """
    # ── Step 1: required columns ────────────────────────────────────────────
    missing = REQUIRED_COLS - set(raw.columns)
    if missing:
        raise IngestError(f"Colonnes manquantes : {sorted(missing)}")

    df = raw.copy()

    # ── Step 2: author filter ───────────────────────────────────────────────
    if author_filter and "fullName" in df.columns:
        df = df[df["fullName"] == author_filter].copy()

    # ── Step 3: drop rows with empty text ───────────────────────────────────
    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.strip() != ""].copy()

    # ── Step 4: deduplication by permalink / hash ───────────────────────────
    df["_dedup_key"] = df.apply(_permalink_or_hash, axis=1)
    df = df.drop_duplicates(subset=["_dedup_key"]).copy()

    # ── Step 5: minimum posts guardrail ─────────────────────────────────────
    if len(df) < MIN_POSTS:
        raise IngestError(
            f"Seulement {len(df)} posts après filtrage — minimum requis : {MIN_POSTS}. "
            "Le scoreur serait invalidé avec aussi peu de données."
        )

    # ── Step 6: truncation guardrail ────────────────────────────────────────
    median_len = df["text"].str.len().median()
    if median_len < MEDIAN_TEXT_LEN_MIN:
        raise IngestError(
            f"Texte suspect : médiane {median_len:.0f} chars < {MEDIAN_TEXT_LEN_MIN}. "
            "L'export semble tronqué (utilisez le format CSV, pas XLSX)."
        )

    # ── Step 7: eng_score ───────────────────────────────────────────────────
    for col in ("likes", "comments", "shares"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    df["eng_score"] = df["likes"] + 3 * df["comments"] + 2 * df["shares"]

    # ── Step 8: days_ago ────────────────────────────────────────────────────
    df["days_ago"] = df["publishDate"].apply(_parse_days_ago)

    # ── Step 9: format ──────────────────────────────────────────────────────
    df["format"] = df.apply(_derive_format, axis=1)

    # ── Step 10: permalink ──────────────────────────────────────────────────
    df["permalink"] = df["_dedup_key"]

    # ── Final: select output columns ────────────────────────────────────────
    out_cols = ["permalink", "text", "likes", "comments", "shares",
                "eng_score", "days_ago", "format"]
    # Keep optional columns if present
    for opt in ("fullName", "images", "videoUrl", "documentUrl"):
        if opt in df.columns:
            out_cols.append(opt)

    return df[out_cols].reset_index(drop=True)


def ingest_to_path(
    content: bytes,
    filename: str,
    dest: Path,
    snapshot_dir: Path | None = None,
    author_filter: str | None = None,
) -> dict[str, Any]:
    """
    Full ingest pipeline: parse → normalize → snapshot existing → atomic write.

    Returns a summary dict (n_posts, date_range, tail_rate, duplicates_removed).
    Never writes dest unless all guardrails pass.
    """
    raw = load_raw(content, filename)
    n_raw = len(raw)

    normalized = normalize(raw, author_filter=author_filter)
    n_out = len(normalized)

    # Snapshot existing file before overwrite
    snapshotted: str | None = None
    if dest.exists() and snapshot_dir is not None:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snap_name = f"{dest.stem}_{date.today().isoformat()}.csv"
        snap_path = snapshot_dir / snap_name
        import shutil
        shutil.copy2(dest, snap_path)
        snapshotted = str(snap_path)

    # Atomic write: tmp → rename
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    normalized.to_csv(tmp, index=False)
    tmp.rename(dest)

    # Summary stats
    tail_rate = float((normalized["eng_score"] >= 98.2).mean())
    days = normalized["days_ago"].dropna()
    date_range = (
        f"{int(days.min())}–{int(days.max())} jours" if len(days) else "N/A"
    )

    return {
        "n_raw": n_raw,
        "n_posts": n_out,
        "duplicates_removed": n_raw - n_out,
        "tail_rate": round(tail_rate, 3),
        "date_range": date_range,
        "snapshot": snapshotted,
        "destination": str(dest),
    }
