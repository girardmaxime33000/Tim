from __future__ import annotations

import json
import os
from pathlib import Path

import typer

app = typer.Typer(help="Growth System CLI — score, recommend, update, monitor, backtest")


@app.command()
def score(
    draft: Path = typer.Option(..., help="Path to draft text file"),
    format: str = typer.Option("image", help="Post format (image/video/text)"),
    model: str = typer.Option("data/scorer_model.json", help="Path to scorer model"),
) -> None:
    """Score a draft and get publish verdict."""
    from .scorer import TailScorer

    text = draft.read_text()
    scorer = TailScorer(model)
    p = scorer.score(text)
    verdict = scorer.verdict(text)
    contribs = scorer.explain(text)
    typer.echo(f"P(queue) = {p:.3f} → {verdict.upper()}")
    typer.echo("\nContributions par feature:")
    for c in sorted(contribs, key=lambda x: abs(x.contribution), reverse=True):
        sign = "+" if c.contribution >= 0 else ""
        typer.echo(f"  {c.feature:20s} val={c.value:.2f}  contrib={sign}{c.contribution:.3f}")


@app.command()
def recommend(
    state: str = typer.Option("growth_state.json", help="Path to state file"),
) -> None:
    """Recommend the archetype to use for the next post."""
    from .bandit import DiscountedThompsonBandit
    from .archetypes import ALL_ARMS

    bandit = DiscountedThompsonBandit(ALL_ARMS, seed=42)
    if os.path.exists(state):
        with open(state) as f:
            s = json.load(f)
        if "bandit" in s:
            bandit.load_state_dict(s["bandit"])
    arm = bandit.recommend()
    typer.echo(f"Archétype recommandé : {arm.upper()}")
    post = bandit.posterior()
    typer.echo("\nPosterior (alpha, beta) par bras:")
    for a, (alpha, beta) in post.items():
        mean = alpha / (alpha + beta)
        typer.echo(f"  {a:15s} α={alpha:.2f} β={beta:.2f} θ̄={mean:.3f}")


@app.command()
def update(
    arm: str = typer.Option(..., help="Archetype arm used"),
    tail: int = typer.Option(..., help="1 if post reached tail (eng >= q80), 0 otherwise"),
    leads: int = typer.Option(0, help="Number of qualified leads"),
    state: str = typer.Option("growth_state.json", help="Path to state file"),
) -> None:
    """Record a post outcome and update bandit beliefs."""
    from .bandit import DiscountedThompsonBandit
    from .archetypes import ALL_ARMS

    bandit = DiscountedThompsonBandit(ALL_ARMS, seed=42)
    if os.path.exists(state):
        with open(state) as f:
            s = json.load(f)
        if "bandit" in s:
            bandit.load_state_dict(s["bandit"])

    reward = 0.3 * tail + 0.6 * min(leads / 3.0, 1.0)
    bandit.update(arm, reward)  # type: ignore[arg-type]

    state_data: dict[str, object] = {"bandit": bandit.state_dict()}
    with open(state, "w") as f:
        json.dump(state_data, f)
    typer.echo(f"Mis à jour : bras={arm}, reward={reward:.3f}")
    typer.echo(f"État sauvegardé dans {state}")


@app.command()
def monitor(
    audience: Path = typer.Option(..., help="Path to daily_audience.csv"),
) -> None:
    """Run CUSUM detector on audience series and report changepoints."""
    import pandas as pd

    from .changepoint import CusumDetector

    df = pd.read_csv(audience, parse_dates=["date"])
    df = df.sort_values("date")
    smooth = df["new_followers"].rolling(14, min_periods=1).mean()
    calib = smooth.iloc[:60]
    detector = CusumDetector(mu0=0.0, k=0.0, h=0.0)
    detector.calibrate(calib)
    typer.echo(f"Calibration: μ₀={detector.mu0:.2f}, k={detector.k:.2f}, h={detector.h:.2f}")
    changepoints = []
    for i, (_, row) in enumerate(df.iterrows()):
        cp = detector.update(float(smooth.iloc[i]), row["date"].date())
        if cp:
            typer.echo(
                f"CHANGEPOINT détecté : {cp.detected_date} ({cp.direction}, stat={cp.statistic:.2f})"
            )
            changepoints.append(cp)
    if not changepoints:
        typer.echo("Aucun changepoint détecté.")
    typer.echo(f"\nTotal : {len(changepoints)} changepoint(s)")


@app.command()
def backtest(
    posts: str = typer.Option("data/posts.csv"),
    audience: str = typer.Option("data/daily_audience.csv"),
    model: str = typer.Option("data/scorer_model.json"),
    output: str = typer.Option("backtest_results.png"),
) -> None:
    """Run backtest comparing 3 policies. Outputs PNG + summary."""
    from .backtest import run_backtest

    typer.echo("Lancement du backtest...")
    summary = run_backtest(posts, audience, model, output)
    typer.echo(json.dumps(summary, indent=2))
    typer.echo(f"\nGraphique sauvegardé : {output}")
