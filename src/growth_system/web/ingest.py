"""Normalisation pipeline for LinkedIn export files (CSV / JSON / XLSX).

Never writes to the real posts.csv — callers decide the destination path.
"""
from __future__ import annotations

import hashlib
import io
import re
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_COLS = {"permalink", "text", "likes", "comments", "shares", "publishDate"}
MIN_POSTS = 50
ENG_Q80 = 98.2  # frozen from scorer_model.json; must not be recalculated on import
MEDIAN_TEXT_LEN_MIN = 300   # export XLSX tronque autour de 160 chars — seuil conservateur
TRUNCATION_ELLIPSIS_RE = re.compile(r"\.\.\.\s*$")
TRUNCATION_FRAC_MAX = 0.20  # > 20 % de posts finissant par "…" → suspect

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


def _check_truncation(df: pd.DataFrame) -> None:
    """Raise IngestError if the text column looks truncated."""
    texts = df["text"].astype(str)
    median_len = texts.str.len().median()
    if median_len < MEDIAN_TEXT_LEN_MIN:
        raise IngestError(
            f"Texte suspect : médiane {median_len:.0f} chars < {MEDIAN_TEXT_LEN_MIN}. "
            "L'export semble tronqué (préférez le format CSV, pas XLSX)."
        )
    ellipsis_frac = texts.apply(lambda t: bool(TRUNCATION_ELLIPSIS_RE.search(t))).mean()
    if ellipsis_frac > TRUNCATION_FRAC_MAX:
        raise IngestError(
            f"{ellipsis_frac:.0%} des posts finissent par '…' — export tronqué détecté. "
            "Scrollez jusqu'en bas de la page avant d'exporter."
        )


def normalize(
    raw: pd.DataFrame,
    author_filter: str | None = None,
    existing: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Normalize a raw LinkedIn export DataFrame into the posts.csv schema.

    If `existing` is provided, the result is the merge of existing posts and
    the new export: existing rows are kept, new rows are appended, duplicates
    (by permalink/hash) are removed. The corpus can only grow, never shrink.

    Steps:
    1. Validate required columns
    2. Filter by author (fullName) if provided
    3. Drop empty-text rows
    4. Deduplicate by permalink / text hash
    5. MIN_POSTS guardrail (on new rows; merged result always ≥ existing)
    6. Truncation guardrail (on new rows)
    7. Compute eng_score, days_ago, format
    8. Merge with existing (union, existing wins on conflict)

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
    # Posts without text are inscoreable and unclassifiable by archetype.
    # They are intentionally excluded rather than kept as unlabelled negatives.
    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.strip() != ""].copy()

    # ── Step 4: deduplication within import ────────────────────────────────
    df["_dedup_key"] = df.apply(_permalink_or_hash, axis=1)
    df = df.drop_duplicates(subset=["_dedup_key"]).copy()

    # ── Step 5: minimum posts guardrail (on import, before merge) ───────────
    if len(df) < MIN_POSTS:
        raise IngestError(
            f"Seulement {len(df)} posts après filtrage — minimum requis : {MIN_POSTS}. "
            "Le scoreur serait invalidé avec aussi peu de données."
        )

    # ── Step 6: truncation guardrail ────────────────────────────────────────
    _check_truncation(df)

    # ── Step 7: eng_score, days_ago, format ─────────────────────────────────
    for col in ("likes", "comments", "shares"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    df["eng_score"] = df["likes"] + 3 * df["comments"] + 2 * df["shares"]
    df["days_ago"] = df["publishDate"].apply(_parse_days_ago)
    df["format"] = df.apply(_derive_format, axis=1)
    df["permalink"] = df["_dedup_key"]

    out_cols = ["permalink", "text", "likes", "comments", "shares",
                "eng_score", "days_ago", "format"]
    for opt in ("fullName", "images", "videoUrl", "documentUrl"):
        if opt in df.columns:
            out_cols.append(opt)
    df = df[out_cols].copy()

    # ── Step 8: merge with existing corpus ──────────────────────────────────
    if existing is not None and len(existing) > 0:
        # Union on permalink. On conflict, keep the row with higher eng_score:
        # LinkedIn engagement is monotonically non-decreasing, so the higher
        # value is always the more recent scrape. This prevents posts scraped
        # early (low likes) from being frozen below the queue threshold forever.
        existing_indexed = existing.set_index("permalink")
        df_indexed = df.set_index("permalink")

        # Start from existing; update rows where new export has higher eng_score
        shared = existing_indexed.index.intersection(df_indexed.index)
        for pk in shared:
            if df_indexed.loc[pk, "eng_score"] > existing_indexed.loc[pk, "eng_score"]:
                existing_indexed.loc[pk] = df_indexed.loc[pk]

        # Append genuinely new rows
        truly_new = df_indexed[~df_indexed.index.isin(existing_indexed.index)]
        merged = pd.concat([existing_indexed, truly_new]).reset_index()
        return merged

    return df.reset_index(drop=True)


def ingest_to_path(
    content: bytes,
    filename: str,
    dest: Path,
    snapshot_dir: Path | None = None,
    author_filter: str | None = None,
) -> dict[str, Any]:
    """
    Full ingest pipeline: parse → normalize (merge with existing) → snapshot → atomic write.

    Returns a summary dict. Never writes dest unless all guardrails pass.
    The corpus can only grow: a partial re-export never silently drops posts.
    """
    raw = load_raw(content, filename)
    n_raw = len(raw)

    # Load existing corpus for merge
    existing: pd.DataFrame | None = None
    n_existing = 0
    if dest.exists():
        existing = pd.read_csv(dest)
        n_existing = len(existing)

    normalized = normalize(raw, author_filter=author_filter, existing=existing)
    n_out = len(normalized)
    n_new = n_out - n_existing

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

    tail_rate = float((normalized["eng_score"] >= ENG_Q80).mean())
    days = normalized["days_ago"].dropna()
    date_range = (
        f"{int(days.min())}–{int(days.max())} jours" if len(days) else "N/A"
    )

    return {
        "n_raw": n_raw,
        "n_posts": n_out,
        "n_new": n_new,
        "duplicates_removed": n_raw - (n_out - n_existing),
        "tail_rate": round(tail_rate, 3),
        "date_range": date_range,
        "snapshot": snapshotted,
        "destination": str(dest),
    }
