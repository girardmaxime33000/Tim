"""FastAPI backend — local only (127.0.0.1).

Wraps the existing growth_system package. No new business logic here.
"""
from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from growth_system.archetypes import ALL_ARMS, ArmId, archetype_of
from growth_system.features import has_cta, has_link, hook_type_label, extract_features
from growth_system.web.ingest import ENG_Q80, IngestError, ingest_to_path, load_raw, normalize
from growth_system.bandit import DiscountedThompsonBandit
from growth_system.changepoint import CusumDetector
from growth_system.config import SystemConfig
from growth_system.orchestrator import GrowthSystem
from growth_system.reward import BusinessReward, PostOutcome, ProxyLeadSource, RewardWeights
from growth_system.scorer import TailScorer

# ---------------------------------------------------------------------------
# Paths (resolved relative to the repo root, not the module)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent.parent.parent  # src/growth_system/web -> repo root
DATA_DIR = _REPO_ROOT / "data"
MODEL_PATH = DATA_DIR / "scorer_model.json"
POSTS_CSV = DATA_DIR / "posts.csv"
AUDIENCE_CSV = DATA_DIR / "daily_audience.csv"
STATE_JSON = _REPO_ROOT / "growth_state.json"
LEADS_CSV = DATA_DIR / "leads.csv"

# ---------------------------------------------------------------------------
# Singletons (loaded once at startup)
# ---------------------------------------------------------------------------

_scorer: TailScorer | None = None
_bandit: DiscountedThompsonBandit | None = None


def _get_scorer() -> TailScorer:
    global _scorer
    if _scorer is None:
        _scorer = TailScorer(str(MODEL_PATH))
    return _scorer


def _get_bandit() -> DiscountedThompsonBandit:
    global _bandit
    if _bandit is None:
        _bandit = DiscountedThompsonBandit(ALL_ARMS, gamma=0.985, seed=42)
        if STATE_JSON.exists():
            with open(STATE_JSON) as f:
                state = json.load(f)
            if "bandit" in state:
                _bandit.load_state_dict(state["bandit"])
    return _bandit


def _save_bandit_state() -> None:
    bandit = _get_bandit()
    state: dict[str, Any] = {"bandit": bandit.state_dict()}
    tmp = STATE_JSON.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    tmp.rename(STATE_JSON)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

_STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Growth System API", version="0.1.0")

# Serve frontend static files (HTML/CSS/JS) at root
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def root() -> FileResponse:
        return FileResponse(str(_STATIC_DIR / "index.html"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000",
                   "http://localhost", "http://127.0.0.1",
                   "null"],  # for file:// during dev
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    files: dict[str, bool]
    posts_count: int | None
    audience_days: int | None
    model_loaded: bool


class ProfileResponse(BaseModel):
    posts_count: int
    last_post_days_ago: int | None
    last_updated: str | None
    tail_rate: float
    top_archetypes: dict[str, float]


class ScoreRequest(BaseModel):
    text: str = Field(..., min_length=1)
    format: str = "Text only"


class ContributionItem(BaseModel):
    feature: str
    value: float
    contribution: float


class ScoreResponse(BaseModel):
    p_queue: float
    verdict: str
    contributions: list[ContributionItem]
    suggestions: list[str]


class RecommendResponse(BaseModel):
    recommended_arm: str
    posteriors: dict[str, dict[str, float]]


class UpdateRequest(BaseModel):
    arm: str
    tail: int = Field(..., ge=0, le=1)
    leads: int = Field(default=0, ge=0)
    post_id: str | None = None


class UpdateResponse(BaseModel):
    reward: float
    bandit_updated: bool
    posteriors: dict[str, dict[str, float]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SUGGESTION_MAP: dict[str, str] = {
    "hook_contrarian": "Commencer par une accroche contrarian (Le mythe…, Et si…, Pourquoi…)",
    "hook_data": "Ouvrir avec un chiffre fort en première ligne",
    "hook_question": "Terminer la première ligne par une question",
    "hashtag": "Ajouter un hashtag pertinent",
    "question_body": "Inclure une question dans le corps pour stimuler les commentaires",
    "log_len": "Allonger le texte (les posts longs surperforment sur ce profil)",
    "emoji": "Réduire le nombre d'emojis (corrélation négative sur ce profil)",
    "tags": "Mentionner moins de personnes (le tag diminue le score ici)",
}


def _build_suggestions(contributions: list[Any], verdict: str) -> list[str]:
    """Return top-3 actionable suggestions based on negative contributions."""
    if verdict == "publish":
        return ["Post prêt à publier — aucune modification requise."]
    neg = sorted(
        [c for c in contributions if c.contribution < -0.05],
        key=lambda c: c.contribution,
    )
    pos_missing = [
        c for c in contributions
        if c.value == 0.0 and c.contribution >= 0 and c.feature.startswith("hook_")
    ]
    tips: list[str] = []
    for c in neg[:2]:
        if c.feature in _SUGGESTION_MAP:
            tips.append(_SUGGESTION_MAP[c.feature])
    for c in pos_missing[:1]:
        if c.feature in _SUGGESTION_MAP:
            tips.append(_SUGGESTION_MAP[c.feature])
    return tips or ["Retravailler l'accroche (première ligne) pour augmenter P(queue)."]


# ---------------------------------------------------------------------------
# Endpoints — Jalon 1
# ---------------------------------------------------------------------------


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Check presence and integrity of data files."""
    files = {
        "posts.csv": POSTS_CSV.exists(),
        "daily_audience.csv": AUDIENCE_CSV.exists(),
        "scorer_model.json": MODEL_PATH.exists(),
    }
    posts_count: int | None = None
    audience_days: int | None = None
    model_loaded = False

    if files["posts.csv"]:
        try:
            df = pd.read_csv(POSTS_CSV)
            posts_count = len(df)
        except Exception:
            pass

    if files["daily_audience.csv"]:
        try:
            df_a = pd.read_csv(AUDIENCE_CSV)
            audience_days = len(df_a)
        except Exception:
            pass

    if files["scorer_model.json"]:
        try:
            _get_scorer()
            model_loaded = True
        except Exception:
            pass

    status = "ok" if all(files.values()) and model_loaded else "degraded"
    return HealthResponse(
        status=status,
        files=files,
        posts_count=posts_count,
        audience_days=audience_days,
        model_loaded=model_loaded,
    )


@app.get("/api/profile", response_model=ProfileResponse)
def profile() -> ProfileResponse:
    """Summary stats about the post corpus."""
    if not POSTS_CSV.exists():
        return ProfileResponse(
            posts_count=0,
            last_post_days_ago=None,
            last_updated=None,
            tail_rate=0.0,
            top_archetypes={},
        )

    df = pd.read_csv(POSTS_CSV)
    df["text"] = df["text"].fillna("")

    tail_rate = float((df["eng_score"] >= 98.2).mean())
    last_post = int(df["days_ago"].min()) if "days_ago" in df.columns else None

    # Archetype distribution (tail rate per arm)
    df["arm"] = df["text"].apply(archetype_of)
    top = (
        df.groupby("arm")["eng_score"]
        .apply(lambda s: float((s >= 98.2).mean()))
        .to_dict()
    )

    mtime = POSTS_CSV.stat().st_mtime
    import datetime
    last_updated = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")

    return ProfileResponse(
        posts_count=len(df),
        last_post_days_ago=last_post,
        last_updated=last_updated,
        tail_rate=round(tail_rate, 3),
        top_archetypes={k: round(v, 3) for k, v in top.items()},
    )


@app.post("/api/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    """Score a draft text and return P(queue), verdict, contributions, suggestions."""
    scorer = _get_scorer()
    p = scorer.score(req.text)
    verdict = scorer.verdict(req.text)
    raw_contribs = scorer.explain(req.text)
    contributions = [
        ContributionItem(
            feature=c.feature,
            value=round(c.value, 4),
            contribution=round(c.contribution, 4),
        )
        for c in raw_contribs
    ]
    suggestions = _build_suggestions(raw_contribs, verdict)

    # Append to score_log.csv (separate file, never touches posts.csv / growth_state.json)
    import datetime as _dt, csv as _csv, tempfile as _tmp, shutil as _shutil
    _log_row = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "text_excerpt": req.text[:160],
        "p_queue": round(p, 4),
        "verdict": verdict,
        "format": req.format,
    }
    _cols = list(_log_row.keys())
    _rows: list[dict] = []
    if SCORE_LOG_CSV.exists():
        try:
            _rows = pd.read_csv(SCORE_LOG_CSV).to_dict("records")
        except Exception:
            pass
    _rows.append(_log_row)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _tmp.NamedTemporaryFile(mode="w", dir=DATA_DIR, suffix=".tmp",
                                 delete=False, newline="") as _tf:
        _w = _csv.DictWriter(_tf, fieldnames=_cols)
        _w.writeheader()
        _w.writerows(_rows)
        _tf_path = _tf.name
    _shutil.move(_tf_path, str(SCORE_LOG_CSV))

    return ScoreResponse(
        p_queue=round(p, 4),
        verdict=verdict,
        contributions=contributions,
        suggestions=suggestions,
    )


@app.get("/api/recommend", response_model=RecommendResponse)
def recommend() -> RecommendResponse:
    """Recommend the best archetype for the next post."""
    bandit = _get_bandit()
    arm = bandit.recommend()
    post = bandit.posterior()
    posteriors = {
        a: {
            "alpha": round(ab[0], 3),
            "beta": round(ab[1], 3),
            "theta": round(ab[0] / (ab[0] + ab[1]), 3),
        }
        for a, ab in post.items()
    }
    return RecommendResponse(recommended_arm=arm, posteriors=posteriors)


@app.post("/api/update", response_model=UpdateResponse)
def update(req: UpdateRequest) -> UpdateResponse:
    """Record a post outcome and update the bandit."""
    if req.arm not in ALL_ARMS:
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail=f"Unknown arm '{req.arm}'. Valid: {ALL_ARMS}")

    reward_weights = RewardWeights()
    # Normalise leads: 3 leads → 1.0 (saturates)
    lead_signal = min(req.leads / 3.0, 1.0)
    reward = float(
        reward_weights.w_tail * req.tail
        + reward_weights.w_lead * lead_signal
    )
    reward = max(0.0, min(1.0, reward))

    bandit = _get_bandit()
    bandit.update(req.arm, reward)  # type: ignore[arg-type]
    _save_bandit_state()

    # Persist to leads.csv if post_id provided
    if req.post_id and req.leads > 0:
        _append_lead(req.post_id, req.leads)

    post = bandit.posterior()
    posteriors = {
        a: {
            "alpha": round(ab[0], 3),
            "beta": round(ab[1], 3),
            "theta": round(ab[0] / (ab[0] + ab[1]), 3),
        }
        for a, ab in post.items()
    }
    return UpdateResponse(reward=round(reward, 4), bandit_updated=True, posteriors=posteriors)


SNAPSHOT_DIR = DATA_DIR / "snapshots"
LAST_INGEST_JSON = DATA_DIR / "last_ingest.json"
SCORE_LOG_CSV = DATA_DIR / "score_log.csv"
PRECISION_LOG_CSV = DATA_DIR / "precision_log.csv"


class IngestResponse(BaseModel):
    n_raw: int
    n_posts: int
    n_new: int
    duplicates_removed: int
    tail_rate: float
    date_range: str
    snapshot: str | None
    destination: str


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile = File(...),
    author: str | None = Form(default=None),
    dry_run: bool = Form(default=False),
) -> IngestResponse:
    """Ingest a LinkedIn export file and normalise it into posts.csv."""
    content = await file.read()
    filename = file.filename or "upload.csv"

    if dry_run:
        # Validate without writing
        try:
            raw = load_raw(content, filename)
            existing = pd.read_csv(POSTS_CSV) if POSTS_CSV.exists() else None
            normalized = normalize(raw, author_filter=author, existing=existing)
        except IngestError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        n_existing = len(existing) if existing is not None else 0
        days = normalized["days_ago"].dropna()
        date_range = (
            f"{int(days.min())}–{int(days.max())} jours" if len(days) else "N/A"
        )
        return IngestResponse(
            n_raw=len(raw),
            n_posts=len(normalized),
            n_new=len(normalized) - n_existing,
            duplicates_removed=len(raw) - (len(normalized) - n_existing),
            tail_rate=round(float((normalized["eng_score"] >= ENG_Q80).mean()), 3),
            date_range=date_range,
            snapshot=None,
            destination="(dry-run — aucune écriture)",
        )

    try:
        summary = ingest_to_path(
            content=content,
            filename=filename,
            dest=POSTS_CSV,
            snapshot_dir=SNAPSHOT_DIR,
            author_filter=author,
        )
    except IngestError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # Reset scorer singleton so next call reloads from fresh posts.csv
    global _scorer
    _scorer = None

    # Persist last ingest timestamp (atomic)
    import datetime as _dt
    _li_tmp = LAST_INGEST_JSON.with_suffix(".tmp")
    _now_iso = _dt.datetime.now().isoformat(timespec="seconds")
    with open(_li_tmp, "w") as _f:
        json.dump({"last_ingest_at": _now_iso}, _f)
    _li_tmp.rename(LAST_INGEST_JSON)

    # Compute and append precision@20% to precision_log.csv
    try:
        _pf = pd.read_csv(POSTS_CSV)
        _scorer_live = _get_scorer()
        _pf["text"] = _pf["text"].fillna("").astype(str)
        _pf["p_queue"] = _pf["text"].apply(_scorer_live.score)
        _pf["in_tail"] = (_pf["eng_score"] >= ENG_Q80).astype(int)
        n_p = len(_pf)
        k = max(1, round(n_p * 0.20))
        top_k_idx = _pf["p_queue"].nlargest(k).index
        precision_at_20 = float(_pf.loc[top_k_idx, "in_tail"].mean())

        _pr_rows: list[dict] = []
        if PRECISION_LOG_CSV.exists():
            try:
                _pr_rows = pd.read_csv(PRECISION_LOG_CSV).to_dict("records")
            except Exception:
                pass
        _pr_rows.append({"timestamp": _now_iso, "precision_at_20": round(precision_at_20, 4),
                          "n_posts": n_p})
        import csv as _csv2, tempfile as _tmp2, shutil as _shutil2
        with _tmp2.NamedTemporaryFile(mode="w", dir=DATA_DIR, suffix=".tmp",
                                      delete=False, newline="") as _pf2:
            _pw = _csv2.DictWriter(_pf2, fieldnames=["timestamp", "precision_at_20", "n_posts"])
            _pw.writeheader()
            _pw.writerows(_pr_rows)
            _pf2_path = _pf2.name
        _shutil2.move(_pf2_path, str(PRECISION_LOG_CSV))
    except Exception:
        pass  # never block ingest on log failure

    return IngestResponse(**summary)


# ---------------------------------------------------------------------------
# Endpoints — Jalon 3
# ---------------------------------------------------------------------------


class AlarmItem(BaseModel):
    date: str
    direction: str
    statistic: float


class FreshnessInfo(BaseModel):
    last_ingest_at: str | None
    last_retrain_at: str | None
    last_changepoint_at: str | None
    changepoint_unaddressed: bool


class LeadsByArchetypeItem(BaseModel):
    archetype: str
    tail_rate: float
    leads_total: int
    leads_per_post: float


class LeadsCoverage(BaseModel):
    n_with_leads: int
    n_total: int
    ratio: float


class MonitorResponse(BaseModel):
    dates: list[str]
    new_followers_smooth: list[float]
    impressions: list[float]
    alarms: list[AlarmItem]
    retrain_recommended: bool
    days_since_last_alarm: int | None
    totals: dict[str, float]
    leads_coverage: LeadsCoverage
    leads_by_archetype: list[LeadsByArchetypeItem]
    freshness: FreshnessInfo
    scoring_recent: list[dict[str, Any]]
    rework_rate_30d: float


@app.get("/api/monitor", response_model=MonitorResponse)
def monitor() -> MonitorResponse:
    """Run CUSUM on daily_audience.csv and return the time series + alarms."""
    if not AUDIENCE_CSV.exists():
        raise HTTPException(status_code=404, detail="daily_audience.csv introuvable.")

    df = pd.read_csv(AUDIENCE_CSV, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    smooth = df["new_followers"].rolling(14, min_periods=1).mean()
    calib = smooth.iloc[:60]
    mu0 = float(calib.mean())
    sigma = max(float(calib.std()), 1e-3)

    from growth_system.changepoint import CusumDetector
    detector = CusumDetector(mu0=mu0, k=0.5 * sigma, h=4.5 * sigma)
    alarms: list[AlarmItem] = []
    for _, row in df.iterrows():
        cp = detector.update(float(smooth[row.name]), row["date"].date())
        if cp:
            alarms.append(AlarmItem(
                date=str(cp.detected_date),
                direction=cp.direction,
                statistic=round(cp.statistic, 3),
            ))

    days_since: int | None = None
    retrain = False
    if alarms:
        from datetime import date as _date
        import datetime
        last = datetime.date.fromisoformat(alarms[-1].date)
        days_since = (datetime.date.today() - last).days
        retrain = days_since <= 90

    # ── Leads coverage & leads by archetype ─────────────────────────────────
    leads_coverage = LeadsCoverage(n_with_leads=0, n_total=0, ratio=0.0)
    leads_by_archetype: list[LeadsByArchetypeItem] = []

    if POSTS_CSV.exists():
        pf = pd.read_csv(POSTS_CSV)
        pf["text"] = pf["text"].fillna("").astype(str)
        pf["arm"] = pf["text"].apply(archetype_of)
        n_total = len(pf)

        leads_map: dict[str, int] = {}  # post_id (= permalink) -> qualified_contacts
        if LEADS_CSV.exists() and LEADS_CSV.stat().st_size > 0:
            try:
                lf = pd.read_csv(LEADS_CSV)
                for _, lr in lf.iterrows():
                    leads_map[str(lr["post_id"])] = int(lr.get("qualified_contacts", 0))
            except Exception:
                pass

        permalinks = pf["permalink"].astype(str)
        n_with_leads = int(permalinks.apply(lambda p: p in leads_map).sum())
        ratio = n_with_leads / n_total if n_total > 0 else 0.0
        leads_coverage = LeadsCoverage(
            n_with_leads=n_with_leads,
            n_total=n_total,
            ratio=round(ratio, 3),
        )

        for arm in ALL_ARMS:
            arm_df = pf[pf["arm"] == arm]
            n_arm = len(arm_df)
            if n_arm == 0:
                leads_by_archetype.append(LeadsByArchetypeItem(
                    archetype=arm, tail_rate=0.0, leads_total=0, leads_per_post=0.0,
                ))
                continue
            tail_rate = float((arm_df["eng_score"] >= ENG_Q80).mean())
            arm_leads = int(arm_df["permalink"].astype(str).apply(
                lambda p: leads_map.get(p, 0)
            ).sum())
            leads_by_archetype.append(LeadsByArchetypeItem(
                archetype=arm,
                tail_rate=round(tail_rate, 3),
                leads_total=arm_leads,
                leads_per_post=round(arm_leads / n_arm, 3),
            ))

    # ── Freshness ────────────────────────────────────────────────────────────
    last_ingest_at: str | None = None
    if LAST_INGEST_JSON.exists():
        try:
            with open(LAST_INGEST_JSON) as _f:
                last_ingest_at = json.load(_f).get("last_ingest_at")
        except Exception:
            pass

    last_retrain_at: str | None = None
    if MODEL_PATH.exists():
        try:
            with open(MODEL_PATH) as _f:
                last_retrain_at = json.load(_f).get("trained_at")
        except Exception:
            pass

    last_changepoint_at: str | None = alarms[-1].date if alarms else None

    changepoint_unaddressed = False
    if last_changepoint_at and last_retrain_at:
        changepoint_unaddressed = last_changepoint_at > last_retrain_at
    elif last_changepoint_at and last_retrain_at is None:
        changepoint_unaddressed = True  # alarm exists but model was never retrained

    freshness = FreshnessInfo(
        last_ingest_at=last_ingest_at,
        last_retrain_at=last_retrain_at,
        last_changepoint_at=last_changepoint_at,
        changepoint_unaddressed=changepoint_unaddressed,
    )

    # ── Score log ────────────────────────────────────────────────────────────
    scoring_recent: list[dict[str, Any]] = []
    rework_rate_30d = 0.0
    if SCORE_LOG_CSV.exists():
        try:
            sl = pd.read_csv(SCORE_LOG_CSV)
            scoring_recent = sl.tail(10).to_dict("records")
            if "timestamp" in sl.columns and "verdict" in sl.columns:
                import datetime as _dt2
                cutoff = (_dt2.datetime.now() - _dt2.timedelta(days=30)).isoformat()
                sl30 = sl[sl["timestamp"] >= cutoff]
                if len(sl30):
                    rework_rate_30d = round(float((sl30["verdict"] == "rework").mean()), 3)
        except Exception:
            pass

    return MonitorResponse(
        dates=[str(d.date()) for d in df["date"]],
        new_followers_smooth=[round(float(v), 2) for v in smooth],
        impressions=[float(v) for v in df["impressions"]],
        alarms=alarms,
        retrain_recommended=retrain,
        days_since_last_alarm=days_since,
        totals={
            "impressions": round(float(df["impressions"].sum()), 0),
            "new_followers": round(float(df["new_followers"].sum()), 0),
            "avg_daily_impressions": round(float(df["impressions"].mean()), 1),
        },
        leads_coverage=leads_coverage,
        leads_by_archetype=leads_by_archetype,
        freshness=freshness,
        scoring_recent=scoring_recent,
        rework_rate_30d=rework_rate_30d,
    )


class LeadItem(BaseModel):
    post_id: str
    qualified_contacts: int = Field(..., ge=0)


class LeadsResponse(BaseModel):
    leads: list[LeadItem]
    total_leads: int


class LeadUpsertResponse(BaseModel):
    post_id: str
    qualified_contacts: int
    action: str  # "created" or "updated"


@app.get("/api/leads", response_model=LeadsResponse)
def get_leads() -> LeadsResponse:
    """Return all manually-entered leads."""
    if not LEADS_CSV.exists():
        return LeadsResponse(leads=[], total_leads=0)
    df = pd.read_csv(LEADS_CSV)
    items = [
        LeadItem(post_id=str(r["post_id"]), qualified_contacts=int(r["qualified_contacts"]))
        for _, r in df.iterrows()
    ]
    return LeadsResponse(leads=items, total_leads=sum(i.qualified_contacts for i in items))


@app.post("/api/leads", response_model=LeadUpsertResponse)
def upsert_lead(req: LeadItem) -> LeadUpsertResponse:
    """Add or update a lead entry (upsert by post_id)."""
    rows: list[dict[str, Any]] = []
    if LEADS_CSV.exists():
        rows = pd.read_csv(LEADS_CSV).to_dict("records")
    action = "created"
    found = False
    for row in rows:
        if str(row.get("post_id")) == req.post_id:
            row["qualified_contacts"] = req.qualified_contacts
            found = True
            action = "updated"
            break
    if not found:
        rows.append({"post_id": req.post_id, "qualified_contacts": req.qualified_contacts})

    df = pd.DataFrame(rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    import tempfile, shutil
    with tempfile.NamedTemporaryFile(mode="w", dir=DATA_DIR, suffix=".tmp", delete=False) as tmp:
        df.to_csv(tmp, index=False)
        tmp_path = tmp.name
    shutil.move(tmp_path, str(LEADS_CSV))

    return LeadUpsertResponse(post_id=req.post_id,
                               qualified_contacts=req.qualified_contacts,
                               action=action)


class BacktestResponse(BaseModel):
    total_reward_P0: float
    total_reward_P1: float
    total_reward_P2: float
    regret_P2_vs_P0: float
    changepoints_detected: list[str]
    p2_beats_p1: bool
    p1_beats_p0: bool
    plot_path: str
    leads_csv_empty: bool
    # Series for frontend charts
    dates: list[str]
    cum_reward_P0: list[float]
    cum_reward_P1: list[float]
    cum_reward_P2: list[float]
    regret: list[float]
    posteriors_over_time: list[dict[str, float]]


@app.post("/api/backtest", response_model=BacktestResponse)
def run_backtest_endpoint() -> BacktestResponse:
    """Run the 3-policy backtest and return curves + summary."""
    if not POSTS_CSV.exists():
        raise HTTPException(status_code=404, detail="posts.csv introuvable.")
    if not AUDIENCE_CSV.exists():
        raise HTTPException(status_code=404, detail="daily_audience.csv introuvable.")
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=404, detail="scorer_model.json introuvable.")

    leads_empty = not LEADS_CSV.exists() or pd.read_csv(LEADS_CSV).empty if LEADS_CSV.exists() else True

    plot_path = str(DATA_DIR / "backtest_results.png")

    from growth_system.backtest import run_backtest, REFERENCE_DATE
    from datetime import timedelta
    import numpy as np

    # Run the standard backtest (produces PNG + summary dict)
    summary = run_backtest(
        posts_path=str(POSTS_CSV),
        audience_path=str(AUDIENCE_CSV),
        model_path=str(MODEL_PATH),
        output_path=plot_path,
    )

    # Rebuild series for the frontend (avoid re-running full backtest)
    posts_df = pd.read_csv(POSTS_CSV)
    posts_df = posts_df.sort_values("days_ago", ascending=False).reset_index(drop=True)
    posts_df["publish_date"] = posts_df["days_ago"].apply(
        lambda d: REFERENCE_DATE - timedelta(days=int(d))
    )
    posts_df["text"] = posts_df["text"].fillna("")
    eng_q80 = 98.2
    posts_df["in_tail"] = (posts_df["eng_score"] >= eng_q80).astype(float)

    from growth_system.reward import BusinessReward, PostOutcome, ProxyLeadSource, RewardWeights
    from growth_system.bandit import DiscountedThompsonBandit
    from growth_system.archetypes import ALL_ARMS, archetype_of

    reward_fn = BusinessReward(RewardWeights(), ProxyLeadSource())

    def _reward(row: pd.Series) -> float:
        return reward_fn.compute(PostOutcome(
            post_id=str(row.name), text=str(row["text"]),
            eng_score=float(row["eng_score"]), eng_q80=eng_q80,
        ))

    rewards = posts_df.apply(_reward, axis=1).values
    cum0 = list(np.cumsum(rewards).round(4))

    bandit1 = DiscountedThompsonBandit(ALL_ARMS, gamma=1.0, seed=43)
    bandit2 = DiscountedThompsonBandit(ALL_ARMS, gamma=0.985, seed=44)
    r1_arr, r2_arr, post_series = [], [], []
    for i, row in posts_df.iterrows():
        arm = archetype_of(str(row["text"]))
        r = rewards[int(i)]
        bandit1.update(arm, r)
        bandit2.update(arm, r)
        r1_arr.append(r)
        r2_arr.append(r)
        post_series.append({a: round(bandit2.posterior()[a][0] /
                                      (bandit2.posterior()[a][0] + bandit2.posterior()[a][1]), 3)
                             for a in ALL_ARMS})

    cum1 = list(np.cumsum(r1_arr).round(4))
    cum2 = list(np.cumsum(r2_arr).round(4))
    regret = [round(float(c0 - c2), 4) for c0, c2 in zip(cum0, cum2)]
    dates = [str(d) for d in posts_df["publish_date"]]

    response = BacktestResponse(
        **{k: summary[k] for k in summary},
        plot_path=plot_path,
        leads_csv_empty=leads_empty,
        dates=dates,
        cum_reward_P0=[round(float(v), 4) for v in cum0],
        cum_reward_P1=[round(float(v), 4) for v in cum1],
        cum_reward_P2=[round(float(v), 4) for v in cum2],
        regret=regret,
        posteriors_over_time=post_series,
    )

    # Persist posteriors for /api/monitor/figures (figure 3 — bandit evolution)
    _bt_path = DATA_DIR / "backtest_posteriors.json"
    _bt_tmp = _bt_path.with_suffix(".tmp")
    with open(_bt_tmp, "w") as _f:
        json.dump({"dates": dates, "posteriors": post_series}, _f)
    _bt_tmp.rename(_bt_path)

    return response


# ---------------------------------------------------------------------------
# Helpers — post table
# ---------------------------------------------------------------------------

_INGEST_REF_DATE = date(2026, 6, 22)  # date of the most recent ingest (approximation)

SORT_FIELDS = {
    "date": "days_ago",
    "eng_score": "eng_score",
    "likes": "likes",
    "comments": "comments",
    "shares": "shares",
    "text_len": "text_len",
}


def _build_post_row(idx: int, row: "pd.Series") -> dict:  # type: ignore[type-arg]
    import datetime
    text = str(row.get("text", ""))
    days = row.get("days_ago")
    if days is not None and not pd.isna(days):
        abs_date = (_INGEST_REF_DATE - datetime.timedelta(days=int(days))).isoformat()
    else:
        abs_date = None
    feats = extract_features(text)
    return {
        "rank": idx + 1,
        "excerpt": text[:160],
        "date": abs_date,
        "format": str(row.get("format", "")),
        "hook_type": hook_type_label(text),
        "likes": int(row.get("likes", 0)),
        "comments": int(row.get("comments", 0)),
        "shares": int(row.get("shares", 0)),
        "eng_score": float(row.get("eng_score", 0)),
        "text_len": len(text),
        "emoji": int(feats["emoji"]),
        "has_link": has_link(text),
        "has_cta": has_cta(text),
        "permalink": str(row.get("permalink", "")),
    }


class PostRow(BaseModel):
    rank: int
    excerpt: str
    date: str | None
    format: str
    hook_type: str
    likes: int
    comments: int
    shares: int
    eng_score: float
    text_len: int
    emoji: int
    has_link: bool
    has_cta: bool
    permalink: str


class PostsResponse(BaseModel):
    posts: list[PostRow]
    total: int
    limit: int
    offset: int


@app.get("/api/posts", response_model=PostsResponse)
def get_posts(
    sort_by: str = "date",
    order: str = "desc",
    format: str | None = None,
    hook_type: str | None = None,
    has_cta_filter: bool | None = None,
    limit: int = 50,
    offset: int = 0,
) -> PostsResponse:
    """Return the post corpus as a table with derived display columns."""
    if not POSTS_CSV.exists():
        return PostsResponse(posts=[], total=0, limit=limit, offset=offset)

    df = pd.read_csv(POSTS_CSV)
    df["text"] = df["text"].fillna("").astype(str)

    # Derived columns for filtering/sorting
    df["text_len"] = df["text"].str.len()
    df["hook_type_col"] = df["text"].apply(hook_type_label)
    df["has_cta_col"] = df["text"].apply(has_cta)

    # Filters
    if format:
        df = df[df["format"] == format]
    if hook_type:
        df = df[df["hook_type_col"] == hook_type]
    if has_cta_filter is not None:
        df = df[df["has_cta_col"] == has_cta_filter]

    # Sort
    sort_col = SORT_FIELDS.get(sort_by, "days_ago")
    if sort_col not in df.columns:
        sort_col = "days_ago"
    ascending = order == "asc"
    # days_ago desc = oldest dates last = chronological asc when sort_by=date
    if sort_by == "date":
        ascending = not ascending  # invert: smaller days_ago = more recent
    df = df.sort_values(sort_col, ascending=ascending, na_position="last")

    total = len(df)
    df = df.iloc[offset: offset + limit].reset_index(drop=True)

    posts = [PostRow(**_build_post_row(i, row)) for i, row in df.iterrows()]
    return PostsResponse(posts=posts, total=total, limit=limit, offset=offset)


# ---------------------------------------------------------------------------
# Endpoint — Plotly figures
# ---------------------------------------------------------------------------


@app.get("/api/monitor/figures")
def monitor_figures() -> dict:  # type: ignore[type-arg]
    """Return Plotly figures as JSON for the Monitoring tab."""
    import plotly.graph_objects as go
    import numpy as _np

    figures: dict[str, str] = {}

    # ── Figure 1: growth — λ(t) + impressions ───────────────────────────────
    if AUDIENCE_CSV.exists():
        df = pd.read_csv(AUDIENCE_CSV, parse_dates=["date"])
        df = df.sort_values("date").reset_index(drop=True)
        smooth = df["new_followers"].rolling(14, min_periods=1).mean()

        calib = smooth.iloc[:60]
        mu0 = float(calib.mean())
        sigma = max(float(calib.std()), 1e-3)
        from growth_system.changepoint import CusumDetector as _CD
        _det = _CD(mu0=mu0, k=0.5 * sigma, h=4.5 * sigma)
        alarms = []
        for _, row in df.iterrows():
            cp = _det.update(float(smooth[row.name]), row["date"].date())
            if cp:
                alarms.append(cp)

        # Build shapes as dicts (avoids slow add_vline / add_vrect calls)
        shapes = []
        annotations = []
        last_date = str(df["date"].dt.date.tolist()[-1]) if len(df) else None
        for cp in alarms:
            if cp.direction == "up":
                next_cp = next((c for c in alarms if c.detected_date > cp.detected_date), None)
                x1 = str(next_cp.detected_date) if next_cp else last_date
                if x1:
                    shapes.append(dict(type="rect", x0=str(cp.detected_date), x1=x1,
                                       y0=0, y1=1, xref="x", yref="paper",
                                       fillcolor="rgba(76,175,80,0.08)", line_width=0))
                    annotations.append(dict(x=str(cp.detected_date), y=0.98, xref="x", yref="paper",
                                            text=f"Accél. {cp.detected_date}", showarrow=False,
                                            font=dict(color="#4CAF50", size=10), xanchor="left"))
            shapes.append(dict(type="line", x0=str(cp.detected_date), x1=str(cp.detected_date),
                               y0=0, y1=1, xref="x", yref="paper",
                               line=dict(color="orange", dash="dot", width=1)))
            annotations.append(dict(x=str(cp.detected_date), y=0.88, xref="x", yref="paper",
                                    text=f"CUSUM {cp.direction}", showarrow=False,
                                    font=dict(color="orange", size=10), xanchor="left"))

        # Two y-axes via layout (avoids make_subplots overhead)
        imp_max = float(df["impressions"].max()) or 1.0
        smooth_max = float(smooth.max()) or 1.0
        imp_scaled = (df["impressions"] / imp_max * smooth_max).tolist()

        fig = go.Figure(data=[
            go.Scatter(x=df["date"].tolist(), y=smooth.round(2).tolist(),
                       name="Abonnés/j (lissé 14j)", line={"color": "#E8540A", "width": 2}),
            go.Scatter(x=df["date"].tolist(), y=imp_scaled,
                       name="Impressions/j (éch. droite)", line={"color": "#4A90D9", "width": 1},
                       opacity=0.6),
        ])
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#1A1A1A", plot_bgcolor="#1A1A1A",
            title="Dynamique d'audience", legend={"orientation": "h"},
            margin={"t": 50, "b": 40},
            shapes=shapes, annotations=annotations,
        )
        figures["growth"] = fig.to_json()

    # ── Figure 2: distribution eng_score ─────────────────────────────────────
    if POSTS_CSV.exists():
        df_p = pd.read_csv(POSTS_CSV)
        fig2 = go.Figure(data=[go.Histogram(
            x=df_p["eng_score"].tolist(), nbinsx=30,
            marker_color="#E8540A", opacity=0.8, name="Eng. Score",
        )])
        fig2.update_layout(
            template="plotly_dark", paper_bgcolor="#1A1A1A", plot_bgcolor="#1A1A1A",
            title="Distribution de l'engagement (eng_score)",
            xaxis_title="Eng. Score", yaxis_title="Nombre de posts",
            margin={"t": 50, "b": 40},
            shapes=[dict(type="line", x0=ENG_Q80, x1=ENG_Q80, y0=0, y1=1,
                         xref="x", yref="paper", line=dict(color="white", dash="dash"))],
            annotations=[dict(x=ENG_Q80, y=1, xref="x", yref="paper",
                              text=f"q80={ENG_Q80}", showarrow=False,
                              font=dict(color="white", size=10))],
        )
        figures["distribution"] = fig2.to_json()

    # ── Figure 3: bandit posteriors over time (from last backtest if available)
    backtest_series_path = DATA_DIR / "backtest_posteriors.json"
    if backtest_series_path.exists():
        with open(backtest_series_path) as f:
            bt = json.load(f)
        dates_bt = bt.get("dates", [])
        fig3 = go.Figure(data=[
            go.Scatter(x=dates_bt, y=[p.get(arm, 0) for p in bt.get("posteriors", [])],
                       name=arm, mode="lines")
            for arm in ALL_ARMS
        ])
        fig3.update_layout(
            template="plotly_dark", paper_bgcolor="#1A1A1A", plot_bgcolor="#1A1A1A",
            title="Évolution des posteriors par archétype (dernier backtest)",
            yaxis_title="θ̄", margin={"t": 50, "b": 40},
        )
        figures["bandit"] = fig3.to_json()
    else:
        fig3 = go.Figure()
        fig3.update_layout(
            template="plotly_dark", paper_bgcolor="#1A1A1A", plot_bgcolor="#1A1A1A",
            title="Posteriors bandit — lancez un backtest pour alimenter ce graphe",
        )
        figures["bandit"] = fig3.to_json()

    # ── Figure 4: CUSUM statistics over time ────────────────────────────────
    if AUDIENCE_CSV.exists():
        df = pd.read_csv(AUDIENCE_CSV, parse_dates=["date"])
        df = df.sort_values("date").reset_index(drop=True)
        smooth = df["new_followers"].rolling(14, min_periods=1).mean()
        calib = smooth.iloc[:60]
        mu0 = float(calib.mean())
        sigma = max(float(calib.std()), 1e-3)
        k = 0.5 * sigma
        h = 4.5 * sigma
        sp, sm = 0.0, 0.0
        sp_vals, sm_vals = [], []
        for v in smooth:
            sp = max(0.0, sp + (float(v) - mu0 - k))
            sm = max(0.0, sm - (float(v) - mu0 - k))
            sp_vals.append(round(sp, 3))
            sm_vals.append(round(sm, 3))
            if sp >= h or sm >= h:
                sp = sm = 0.0

        dates_list = df["date"].tolist()
        fig4 = go.Figure(data=[
            go.Scatter(x=dates_list, y=sp_vals, name="S+ (hausse)", line={"color": "#E8540A"}),
            go.Scatter(x=dates_list, y=sm_vals, name="S− (baisse)", line={"color": "#4A90D9"}),
        ])
        fig4.update_layout(
            template="plotly_dark", paper_bgcolor="#1A1A1A", plot_bgcolor="#1A1A1A",
            title="Statistiques CUSUM (S+, S−)", yaxis_title="Statistique",
            margin={"t": 50, "b": 40},
            shapes=[dict(type="line", x0=dates_list[0], x1=dates_list[-1], y0=h, y1=h,
                         xref="x", yref="y", line=dict(color="white", dash="dash"))],
            annotations=[dict(x=dates_list[-1], y=h, text=f"h={h:.1f}", showarrow=False,
                              font=dict(color="white", size=10))],
        )
        figures["cusum"] = fig4.to_json()

    # ── Figure 5: bandit posteriors EN DIRECT ────────────────────────────────
    import numpy as _np2

    bandit_live = _get_bandit()
    live_post = bandit_live.posterior()
    best_live = max(ALL_ARMS, key=lambda a: live_post[a][0] / (live_post[a][0] + live_post[a][1]))

    ts = "jamais"
    if STATE_JSON.exists():
        import datetime as _dt
        ts = _dt.datetime.fromtimestamp(STATE_JSON.stat().st_mtime).strftime("%Y-%m-%d %H:%M")

    _x = _np2.linspace(0, 1, 200)
    _COLORS = {"contrarian": "#E8540A", "data": "#4A90D9",
               "question": "#4CAF50", "statement": "#FF9800"}

    def _beta_pdf(x: "_np2.ndarray", a: float, b: float) -> "_np2.ndarray":  # type: ignore[name-defined]
        """Fast Beta PDF via numpy log-space (avoids scipy import)."""
        from math import lgamma
        log_norm = lgamma(a + b) - lgamma(a) - lgamma(b)
        log_p = (a - 1) * _np2.log(_np2.maximum(x, 1e-300)) + \
                (b - 1) * _np2.log(_np2.maximum(1 - x, 1e-300)) + log_norm
        return _np2.exp(log_p)

    fig_live = go.Figure(data=[
        go.Scatter(
            x=_x.tolist(),
            y=_beta_pdf(_x, live_post[arm][0], live_post[arm][1]).tolist(),
            name=arm + (" ← recommandé" if arm == best_live else ""),
            line={"color": _COLORS.get(arm, "#888"),
                  "width": 3 if arm == best_live else 1.5},
        )
        for arm in ALL_ARMS
    ])
    fig_live.update_layout(
        template="plotly_dark", paper_bgcolor="#1A1A1A", plot_bgcolor="#1A1A1A",
        title=f"Posteriors bandit EN DIRECT — Dernière MàJ : {ts}",
        xaxis_title="θ (taux de succès estimé)",
        yaxis_title="Densité Beta",
        margin={"t": 60, "b": 40},
    )
    figures["bandit_live"] = fig_live.to_json()

    # ── Figure 6: précision@20% dans le temps ────────────────────────────────
    fig_prec = go.Figure()
    if PRECISION_LOG_CSV.exists():
        try:
            pr_df = pd.read_csv(PRECISION_LOG_CSV)
            if len(pr_df) >= 1:
                fig_prec.add_trace(go.Scatter(
                    x=pr_df["timestamp"].tolist(),
                    y=pr_df["precision_at_20"].tolist(),
                    mode="lines+markers",
                    name="Précision@20%",
                    line={"color": "#E8540A"},
                    text=[f"n={n}" for n in pr_df["n_posts"].tolist()],
                    hovertemplate="%{x}<br>Préc@20: %{y:.3f}<br>%{text}<extra></extra>",
                ))
        except Exception:
            pass
    fig_prec.update_layout(
        template="plotly_dark", paper_bgcolor="#1A1A1A", plot_bgcolor="#1A1A1A",
        title="Tendance Précision@20% (par ingestion)",
        yaxis_title="Précision@20%", yaxis={"range": [0, 1]},
        margin={"t": 50, "b": 40},
    )
    figures["precision_trend"] = fig_prec.to_json()

    return figures


# ---------------------------------------------------------------------------
# Endpoints — Attribution leads LinkedIn → posts
# ---------------------------------------------------------------------------


class ProposedLeadRow(BaseModel):
    post_id: str
    qualified_contacts: float
    raw_count: int
    weighted_count: float


class UnattributedItem(BaseModel):
    nom: str
    raison: str


class AttributeDetailItem(BaseModel):
    nom: str
    titre: str
    date_brute: str
    date_parsee: str | None
    motivation: str
    type_lead: str
    poids: float
    post_id_attribue: str | None
    date_post_attribue: str | None
    eng_score_post: float | None
    fiabilite: str


class AttributeResponse(BaseModel):
    n_leads_total: int
    n_leads_parsed_date: int
    n_leads_attributed: int
    n_leads_unattributed: int
    unattributed: list[UnattributedItem]
    proposed_leads_csv: list[ProposedLeadRow]
    detail: list[AttributeDetailItem]
    estimation: bool


class ApplyLeadsRequest(BaseModel):
    proposed_leads_csv: list[ProposedLeadRow]
    merge_strategy: str = "add"  # "add" | "replace"


class ApplyLeadsResponse(BaseModel):
    applied: int
    snapshot: str


@app.post("/api/leads/attribute", response_model=AttributeResponse)
async def attribute_leads(
    file: UploadFile = File(...),
    window_days: int = Form(default=7),
    reference_date: str | None = Form(default=None),
    type_weights: str | None = Form(default=None),
) -> AttributeResponse:
    """Attribue des leads (export CSV connexions LinkedIn) aux posts, par inférence temporelle.

    NE MODIFIE JAMAIS leads.csv sur disque.
    """
    import io as _io
    from datetime import date as _date
    from growth_system.leads_attribution import (
        parse_lead_date,
        attribute_lead_to_post,
        aggregate_attribution,
    )
    from growth_system.config import DEFAULT_TYPE_WEIGHTS, DEFAULT_ATTRIBUTION_WEIGHT_FALLBACK

    # --- Reference date ---
    if reference_date:
        ref_date = _date.fromisoformat(reference_date)
    else:
        ref_date = _date.today()

    # --- Type weights ---
    tw: dict[str, float] = dict(DEFAULT_TYPE_WEIGHTS)
    if type_weights:
        try:
            tw.update(json.loads(type_weights))
        except Exception:
            pass

    # --- Read uploaded CSV ---
    content = await file.read()
    try:
        leads_df = pd.read_csv(_io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Impossible de lire le CSV : {exc}")

    # --- Detect columns (flexible) ---
    col_map: dict[str, str] = {}
    for col in leads_df.columns:
        cl = col.lower().strip()
        if cl in ("nom_prenom", "nom", "name", "prénom et nom", "prenom"):
            col_map.setdefault("nom_prenom", col)
        elif cl in ("titre", "title", "poste", "fonction"):
            col_map.setdefault("titre", col)
        elif cl in ("date_connexion", "date", "connected on", "date de connexion"):
            col_map.setdefault("date_connexion", col)
        elif cl in ("motivation", "note", "notes"):
            col_map.setdefault("motivation", col)
        elif cl in ("type_lead", "type", "catégorie", "categorie"):
            col_map.setdefault("type_lead", col)

    def _get(row: "pd.Series", key: str, default: str = "") -> str:
        col = col_map.get(key)
        if col and col in row.index:
            v = row[col]
            return str(v) if pd.notna(v) else default
        return default

    # --- Load posts ---
    posts_for_attr: pd.DataFrame = pd.DataFrame(
        columns=["post_id", "post_date", "eng_score"]
    )
    if POSTS_CSV.exists():
        try:
            pf = pd.read_csv(POSTS_CSV)
            import datetime as _dt
            ingest_ref = _date(2026, 6, 22)  # fallback; use _INGEST_REF_DATE
            try:
                from growth_system.web.api import _INGEST_REF_DATE as _iref
                ingest_ref = _iref
            except Exception:
                pass
            pf["post_date"] = pf["days_ago"].apply(
                lambda d: ingest_ref - _dt.timedelta(days=int(d)) if pd.notna(d) else None
            )
            pf = pf.dropna(subset=["post_date"])
            posts_for_attr = pf[["permalink", "post_date", "eng_score"]].copy()
            posts_for_attr = posts_for_attr.rename(columns={"permalink": "post_id"})
        except Exception:
            pass

    # --- Process each lead ---
    detail_rows: list[AttributeDetailItem] = []
    unattributed: list[UnattributedItem] = []
    attribution_records: list[dict] = []

    n_parsed = 0
    n_attributed = 0

    for _, row in leads_df.iterrows():
        nom = _get(row, "nom_prenom", "Inconnu")
        titre = _get(row, "titre")
        date_brute = _get(row, "date_connexion")
        motivation = _get(row, "motivation")
        type_lead = _get(row, "type_lead", "À vérifier")

        poids = tw.get(type_lead, DEFAULT_ATTRIBUTION_WEIGHT_FALLBACK)

        # Parse date
        parsed_date = parse_lead_date(date_brute, ref_date)
        if parsed_date is None:
            unattributed.append(UnattributedItem(nom=nom, raison="date non résolue"))
            detail_rows.append(AttributeDetailItem(
                nom=nom, titre=titre, date_brute=date_brute, date_parsee=None,
                motivation=motivation, type_lead=type_lead, poids=poids,
                post_id_attribue=None, date_post_attribue=None, eng_score_post=None,
                fiabilite="date non résolue",
            ))
            continue

        n_parsed += 1

        # Attribute to post
        attr = attribute_lead_to_post(parsed_date, posts_for_attr, window_days=window_days)
        if attr is None:
            unattributed.append(UnattributedItem(nom=nom, raison="hors fenêtre"))
            detail_rows.append(AttributeDetailItem(
                nom=nom, titre=titre, date_brute=date_brute,
                date_parsee=str(parsed_date),
                motivation=motivation, type_lead=type_lead, poids=poids,
                post_id_attribue=None, date_post_attribue=None, eng_score_post=None,
                fiabilite="hors fenêtre",
            ))
            continue

        n_attributed += 1
        fiab = f"estimée (fenêtre {window_days}j)"
        detail_rows.append(AttributeDetailItem(
            nom=nom, titre=titre, date_brute=date_brute,
            date_parsee=str(parsed_date),
            motivation=motivation, type_lead=type_lead, poids=poids,
            post_id_attribue=attr.post_id,
            date_post_attribue=str(attr.post_date),
            eng_score_post=attr.eng_score,
            fiabilite=fiab,
        ))
        attribution_records.append({
            "post_id_attribue": attr.post_id,
            "poids": poids,
            "nom_prenom": nom,
            "type_lead": type_lead,
        })

    # --- Aggregate ---
    proposed: list[ProposedLeadRow] = []
    if attribution_records:
        attr_df = pd.DataFrame(attribution_records)
        agg = aggregate_attribution(attr_df)
        for _, agg_row in agg.iterrows():
            proposed.append(ProposedLeadRow(
                post_id=str(agg_row["post_id_attribue"]),
                qualified_contacts=float(agg_row["weighted_count"]),
                raw_count=int(agg_row["raw_count"]),
                weighted_count=float(agg_row["weighted_count"]),
            ))

    n_total = len(leads_df)
    return AttributeResponse(
        n_leads_total=n_total,
        n_leads_parsed_date=n_parsed,
        n_leads_attributed=n_attributed,
        n_leads_unattributed=n_total - n_attributed,
        unattributed=unattributed,
        proposed_leads_csv=proposed,
        detail=detail_rows,
        estimation=True,
    )


@app.post("/api/leads/attribute/apply", response_model=ApplyLeadsResponse)
def apply_leads(req: ApplyLeadsRequest) -> ApplyLeadsResponse:
    """Applique les leads proposés dans leads.csv de façon atomique.

    Crée un snapshot horodaté dans data/snapshots/ AVANT l'écriture.
    merge_strategy :
      - "add"     : ajoute les qualified_contacts aux valeurs existantes
      - "replace" : écrase les valeurs existantes
    """
    import tempfile
    import shutil
    import datetime as _dt

    # --- Snapshot AVANT écriture ---
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = SNAPSHOT_DIR / f"leads_{ts}.csv"

    existing_rows: list[dict[str, Any]] = []
    if LEADS_CSV.exists():
        try:
            existing_rows = pd.read_csv(LEADS_CSV).to_dict("records")
        except Exception:
            pass

    # Écrire snapshot (même si vide)
    snap_df = pd.DataFrame(existing_rows) if existing_rows else pd.DataFrame(
        columns=["post_id", "qualified_contacts"]
    )
    snap_df.to_csv(snapshot_path, index=False)

    # --- Merge ---
    # Construire un dict existant : post_id → qualified_contacts
    existing_map: dict[str, float] = {
        str(r.get("post_id", "")): float(r.get("qualified_contacts", 0))
        for r in existing_rows
    }

    for item in req.proposed_leads_csv:
        if req.merge_strategy == "add":
            existing_map[item.post_id] = existing_map.get(item.post_id, 0.0) + item.qualified_contacts
        else:  # "replace"
            existing_map[item.post_id] = item.qualified_contacts

    # --- Écriture atomique ---
    merged_rows = [
        {"post_id": k, "qualified_contacts": v}
        for k, v in existing_map.items()
    ]
    out_df = pd.DataFrame(merged_rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=DATA_DIR, suffix=".tmp", delete=False, newline=""
    ) as tmp:
        out_df.to_csv(tmp, index=False)
        tmp_path = tmp.name
    shutil.move(tmp_path, str(LEADS_CSV))

    return ApplyLeadsResponse(
        applied=len(req.proposed_leads_csv),
        snapshot=str(snapshot_path.name),
    )


def _append_lead(post_id: str, leads: int) -> None:
    """Atomically append a lead entry to leads.csv."""
    import tempfile, shutil
    rows: list[dict[str, Any]] = []
    if LEADS_CSV.exists():
        rows = pd.read_csv(LEADS_CSV).to_dict("records")
    # upsert
    found = False
    for row in rows:
        if str(row.get("post_id")) == post_id:
            row["qualified_contacts"] = leads
            found = True
            break
    if not found:
        rows.append({"post_id": post_id, "qualified_contacts": leads})
    df = pd.DataFrame(rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=DATA_DIR, suffix=".tmp", delete=False
    ) as tmp:
        df.to_csv(tmp, index=False)
        tmp_path = tmp.name
    shutil.move(tmp_path, str(LEADS_CSV))
