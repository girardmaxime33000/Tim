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

    return IngestResponse(**summary)


# ---------------------------------------------------------------------------
# Endpoints — Jalon 3
# ---------------------------------------------------------------------------


class AlarmItem(BaseModel):
    date: str
    direction: str
    statistic: float


class MonitorResponse(BaseModel):
    dates: list[str]
    new_followers_smooth: list[float]
    impressions: list[float]
    alarms: list[AlarmItem]
    retrain_recommended: bool
    days_since_last_alarm: int | None
    totals: dict[str, float]


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

    return BacktestResponse(
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
