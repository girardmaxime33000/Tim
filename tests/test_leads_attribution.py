"""Tests du pipeline d'attribution leads LinkedIn → posts."""
from __future__ import annotations

import io
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from growth_system.leads_attribution import (
    AttributionResult,
    aggregate_attribution,
    attribute_lead_to_post,
    normalize_post_id,
    parse_lead_date,
)


# ---------------------------------------------------------------------------
# parse_lead_date
# ---------------------------------------------------------------------------


def test_parse_aujourdhui():
    ref = date(2026, 5, 15)
    assert parse_lead_date("Aujourd'hui", ref) == ref


def test_parse_aujourdhui_lowercase():
    ref = date(2026, 5, 15)
    assert parse_lead_date("aujourd'hui", ref) == ref


def test_parse_jour_semaine():
    # ref = mercredi 2026-06-17 → vendredi précédent = 2026-06-12
    ref = date(2026, 6, 17)  # mercredi
    result = parse_lead_date("Vendredi", ref)
    assert result == date(2026, 6, 12)
    assert result.weekday() == 4  # vendredi


def test_parse_jour_semaine_same_day():
    # ref = vendredi 2026-06-13 → vendredi précédent = 2026-06-06
    ref = date(2026, 6, 13)  # c'est un samedi en 2026, cherchons un vrai vendredi
    # On cherche un ref qui est un vendredi
    # 2026-06-12 est un vendredi
    ref_friday = date(2026, 6, 12)
    assert ref_friday.weekday() == 4  # vendredi
    result = parse_lead_date("Vendredi", ref_friday)
    # Doit renvoyer le vendredi PRÉCÉDENT, pas ref_friday lui-même
    assert result == date(2026, 6, 5)
    assert result != ref_friday


def test_parse_date_numerique():
    ref = date(2026, 5, 20)
    result = parse_lead_date("19 avr.", ref)
    assert result == date(2026, 4, 19)


def test_parse_date_juin():
    ref = date(2026, 6, 20)
    result = parse_lead_date("6 juin", ref)
    assert result == date(2026, 6, 6)


def test_parse_compose():
    # "31 mai / mardi" → 31 mai de l'année appropriée
    ref = date(2026, 6, 10)
    result = parse_lead_date("31 mai / mardi", ref)
    assert result == date(2026, 5, 31)


def test_parse_a_verifier():
    ref = date(2026, 5, 15)
    assert parse_lead_date("À vérifier", ref) is None


def test_parse_inconnu():
    ref = date(2026, 5, 15)
    assert parse_lead_date("blah blah", ref) is None


def test_parse_annee_deduite():
    # "15 janv." avec ref=2026-03-01 → 2026-01-15 (pas 2025)
    ref = date(2026, 3, 1)
    result = parse_lead_date("15 janv.", ref)
    assert result == date(2026, 1, 15)


def test_parse_annee_futur_evite():
    # "15 mars" avec ref=2026-02-01 → 2025-03-15 (pas 2026 qui serait dans le futur)
    ref = date(2026, 2, 1)
    result = parse_lead_date("15 mars", ref)
    assert result == date(2025, 3, 15)


# ---------------------------------------------------------------------------
# attribute_lead_to_post
# ---------------------------------------------------------------------------


def _make_posts(*rows: tuple[str, date, float]) -> pd.DataFrame:
    """Helper : (post_id, post_date, eng_score)"""
    return pd.DataFrame(rows, columns=["post_id", "post_date", "eng_score"])


def test_fenetre_vide():
    lead_date = date(2026, 5, 15)
    posts = _make_posts(
        ("p1", date(2026, 5, 5), 100.0),   # trop ancien
        ("p2", date(2026, 5, 25), 200.0),  # futur
    )
    result = attribute_lead_to_post(lead_date, posts, window_days=7)
    assert result is None


def test_plusieurs_posts_retourne_max_eng():
    lead_date = date(2026, 5, 15)
    posts = _make_posts(
        ("p1", date(2026, 5, 10), 50.0),
        ("p2", date(2026, 5, 12), 200.0),  # meilleur
        ("p3", date(2026, 5, 14), 100.0),
    )
    result = attribute_lead_to_post(lead_date, posts, window_days=7)
    assert result is not None
    assert result.post_id == "p2"
    assert result.eng_score == 200.0


def test_borne_j_moins_7_incluse():
    lead_date = date(2026, 5, 15)
    # post exactement à lead_date - 7 doit être inclus
    posts = _make_posts(
        ("p1", date(2026, 5, 8), 99.0),  # = lead_date - 7
    )
    result = attribute_lead_to_post(lead_date, posts, window_days=7)
    assert result is not None
    assert result.post_id == "p1"


def test_borne_j_moins_8_exclue():
    lead_date = date(2026, 5, 15)
    # post à lead_date - 8 doit être exclu
    posts = _make_posts(
        ("p1", date(2026, 5, 7), 99.0),  # = lead_date - 8
    )
    result = attribute_lead_to_post(lead_date, posts, window_days=7)
    assert result is None


def test_borne_lead_date_incluse():
    lead_date = date(2026, 5, 15)
    # post à lead_date exact doit être inclus
    posts = _make_posts(
        ("p1", date(2026, 5, 15), 150.0),
    )
    result = attribute_lead_to_post(lead_date, posts, window_days=7)
    assert result is not None
    assert result.post_id == "p1"


# ---------------------------------------------------------------------------
# aggregate_attribution
# ---------------------------------------------------------------------------


def test_weighted_count():
    df = pd.DataFrame([
        {"post_id_attribue": "p1", "poids": 1.0, "nom_prenom": "Alice", "type_lead": "Artiste"},
        {"post_id_attribue": "p1", "poids": 0.9, "nom_prenom": "Bob", "type_lead": "Partenaire"},
        {"post_id_attribue": "p2", "poids": 0.4, "nom_prenom": "Carol", "type_lead": "Prestataire"},
    ])
    result = aggregate_attribution(df)
    p1 = result[result["post_id_attribue"] == "p1"].iloc[0]
    assert p1["raw_count"] == 2
    assert p1["weighted_count"] == 1.9
    p2 = result[result["post_id_attribue"] == "p2"].iloc[0]
    assert p2["raw_count"] == 1
    assert p2["weighted_count"] == 0.4


def test_type_inconnu_poids_defaut():
    """Type inconnu dans aggregate_attribution — les poids sont déjà calculés en amont."""
    from growth_system.config import DEFAULT_ATTRIBUTION_WEIGHT_FALLBACK, DEFAULT_TYPE_WEIGHTS
    type_inconnu = "TypeInexistant"
    poids = DEFAULT_TYPE_WEIGHTS.get(type_inconnu, DEFAULT_ATTRIBUTION_WEIGHT_FALLBACK)
    assert poids == DEFAULT_ATTRIBUTION_WEIGHT_FALLBACK

    df = pd.DataFrame([
        {"post_id_attribue": "p1", "poids": poids, "nom_prenom": "Dave", "type_lead": type_inconnu},
    ])
    result = aggregate_attribution(df)
    assert len(result) == 1
    assert result.iloc[0]["weighted_count"] == round(DEFAULT_ATTRIBUTION_WEIGHT_FALLBACK, 1)


# ---------------------------------------------------------------------------
# Endpoint /api/leads/attribute — pas d'effet de bord sur leads.csv
# ---------------------------------------------------------------------------


def _make_csv_bytes(rows: list[dict]) -> bytes:
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


@pytest.fixture()
def api_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import growth_system.web.api as api_mod
    monkeypatch.setattr(api_mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(api_mod, "LEADS_CSV", tmp_path / "leads.csv")
    monkeypatch.setattr(api_mod, "POSTS_CSV", tmp_path / "posts.csv")
    monkeypatch.setattr(api_mod, "SNAPSHOT_DIR", tmp_path / "snapshots")
    # Créer un posts.csv minimal
    posts_df = pd.DataFrame([
        {"permalink": "post-1", "days_ago": 5, "eng_score": 150.0, "text": "Post test 1",
         "format": "Text only", "likes": 10, "comments": 2, "shares": 1},
        {"permalink": "post-2", "days_ago": 10, "eng_score": 80.0, "text": "Post test 2",
         "format": "Text only", "likes": 5, "comments": 1, "shares": 0},
    ])
    posts_df.to_csv(tmp_path / "posts.csv", index=False)

    from fastapi.testclient import TestClient
    from growth_system.web.api import app
    return TestClient(app)


def test_attribute_no_side_effect(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """POST /api/leads/attribute ne doit pas modifier leads.csv."""
    import growth_system.web.api as api_mod
    leads_csv = tmp_path / "leads.csv"
    monkeypatch.setattr(api_mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(api_mod, "LEADS_CSV", leads_csv)
    monkeypatch.setattr(api_mod, "POSTS_CSV", tmp_path / "posts.csv")
    monkeypatch.setattr(api_mod, "SNAPSHOT_DIR", tmp_path / "snapshots")

    # Créer posts.csv
    posts_df = pd.DataFrame([
        {"permalink": "post-1", "days_ago": 5, "eng_score": 150.0, "text": "Post test",
         "format": "Text only", "likes": 10, "comments": 2, "shares": 1},
    ])
    posts_df.to_csv(tmp_path / "posts.csv", index=False)

    # leads.csv n'existe pas au départ
    assert not leads_csv.exists()

    csv_bytes = _make_csv_bytes([
        {"nom_prenom": "Alice Martin", "titre": "Artiste", "date_connexion": "Aujourd'hui",
         "motivation": "test", "type_lead": "Artiste"},
    ])

    from fastapi.testclient import TestClient
    from growth_system.web.api import app
    client = TestClient(app)

    # Utiliser reference_date explicite pour éviter dépendance à date.today()
    r = client.post(
        "/api/leads/attribute",
        files={"file": ("leads.csv", csv_bytes, "text/csv")},
        data={"reference_date": "2026-06-22", "window_days": "7"},
    )
    assert r.status_code == 200

    # leads.csv NE DOIT PAS avoir été créé
    assert not leads_csv.exists(), "leads.csv ne doit pas être créé par /api/leads/attribute"


def test_apply_add_merges_correctly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """apply avec merge_strategy=add doit additionner les qualified_contacts."""
    import growth_system.web.api as api_mod
    leads_csv = tmp_path / "leads.csv"
    monkeypatch.setattr(api_mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(api_mod, "LEADS_CSV", leads_csv)
    monkeypatch.setattr(api_mod, "SNAPSHOT_DIR", tmp_path / "snapshots")

    # leads.csv existant avec post-1 = 1
    pd.DataFrame([{"post_id": "post-1", "qualified_contacts": 1}]).to_csv(leads_csv, index=False)

    from fastapi.testclient import TestClient
    from growth_system.web.api import app
    client = TestClient(app)

    body = {
        "proposed_leads_csv": [
            {"post_id": "post-1", "qualified_contacts": 2.0, "raw_count": 2, "weighted_count": 2.0}
        ],
        "merge_strategy": "add",
    }
    r = client.post("/api/leads/attribute/apply", json=body)
    assert r.status_code == 200

    df = pd.read_csv(leads_csv)
    row = df[df["post_id"] == "post-1"].iloc[0]
    assert float(row["qualified_contacts"]) == 3.0


def test_apply_replace_overwrites(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """apply avec merge_strategy=replace doit écraser les qualified_contacts."""
    import growth_system.web.api as api_mod
    leads_csv = tmp_path / "leads.csv"
    monkeypatch.setattr(api_mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(api_mod, "LEADS_CSV", leads_csv)
    monkeypatch.setattr(api_mod, "SNAPSHOT_DIR", tmp_path / "snapshots")

    # leads.csv existant avec post-1 = 5
    pd.DataFrame([{"post_id": "post-1", "qualified_contacts": 5}]).to_csv(leads_csv, index=False)

    from fastapi.testclient import TestClient
    from growth_system.web.api import app
    client = TestClient(app)

    body = {
        "proposed_leads_csv": [
            {"post_id": "post-1", "qualified_contacts": 2.0, "raw_count": 1, "weighted_count": 2.0}
        ],
        "merge_strategy": "replace",
    }
    r = client.post("/api/leads/attribute/apply", json=body)
    assert r.status_code == 200

    df = pd.read_csv(leads_csv)
    row = df[df["post_id"] == "post-1"].iloc[0]
    assert float(row["qualified_contacts"]) == 2.0


def test_apply_creates_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """apply doit créer un snapshot dans snapshots/ avant l'écriture."""
    import growth_system.web.api as api_mod
    leads_csv = tmp_path / "leads.csv"
    snapshot_dir = tmp_path / "snapshots"
    monkeypatch.setattr(api_mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(api_mod, "LEADS_CSV", leads_csv)
    monkeypatch.setattr(api_mod, "SNAPSHOT_DIR", snapshot_dir)

    # leads.csv existant
    pd.DataFrame([{"post_id": "post-A", "qualified_contacts": 1}]).to_csv(leads_csv, index=False)

    from fastapi.testclient import TestClient
    from growth_system.web.api import app
    client = TestClient(app)

    body = {
        "proposed_leads_csv": [
            {"post_id": "post-B", "qualified_contacts": 1.0, "raw_count": 1, "weighted_count": 1.0}
        ],
        "merge_strategy": "add",
    }
    r = client.post("/api/leads/attribute/apply", json=body)
    assert r.status_code == 200

    snapshots = list(snapshot_dir.glob("leads_*.csv"))
    assert len(snapshots) >= 1, "Un snapshot doit être créé avant l'écriture"


# ---------------------------------------------------------------------------
# load_analytics_post_dates + enrichissement des dates de posts
# ---------------------------------------------------------------------------


def _make_analytics_xlsx(path: Path, rows: list[dict]) -> None:
    """Crée un AggregateAnalytics_*.xlsx minimal avec feuille MEILLEURS POSTS.

    rows = [{"url": "...", "date": "JJ/MM/AAAA"}, ...]
    Simule deux blocs côte à côte (même colonnes dupliquées) pour couvrir la
    déduplification.
    """
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "MEILLEURS POSTS"
    # Deux blocs : colonnes A-B (tri Interactions) et D-E (tri Impressions)
    ws["A1"] = "URL du post"
    ws["B1"] = "Date de publication du post"
    ws["D1"] = "URL du post"
    ws["E1"] = "Date de publication du post"
    for i, row in enumerate(rows, start=2):
        ws[f"A{i}"] = row["url"]
        ws[f"B{i}"] = row["date"]
        ws[f"D{i}"] = row["url"]   # doublon intentionnel
        ws[f"E{i}"] = row["date"]
    wb.save(path)


def test_load_analytics_no_dir():
    """Sans répertoire analytics → dict vide, pas d'erreur."""
    from growth_system.leads_attribution import load_analytics_post_dates
    result = load_analytics_post_dates(Path("/tmp/inexistant_analytics_xyz"))
    assert result == {}


def test_load_analytics_no_file(tmp_path: Path):
    """Répertoire présent mais aucun fichier → dict vide."""
    from growth_system.leads_attribution import load_analytics_post_dates
    analytics_dir = tmp_path / "analytics"
    analytics_dir.mkdir()
    result = load_analytics_post_dates(analytics_dir)
    assert result == {}


def test_load_analytics_exact_dates(tmp_path: Path):
    """Les URLs présentes dans MEILLEURS POSTS obtiennent leur date exacte."""
    from growth_system.leads_attribution import load_analytics_post_dates
    analytics_dir = tmp_path / "analytics"
    analytics_dir.mkdir()
    xlsx_path = analytics_dir / "AggregateAnalytics_2026-06-01.xlsx"
    _make_analytics_xlsx(xlsx_path, [
        {"url": "https://www.linkedin.com/posts/post-1", "date": "28/04/2026"},
        {"url": "https://www.linkedin.com/posts/post-2", "date": "03/05/2026"},
    ])
    result = load_analytics_post_dates(analytics_dir)
    assert result["https://www.linkedin.com/posts/post-1"] == date(2026, 4, 28)
    assert result["https://www.linkedin.com/posts/post-2"] == date(2026, 5, 3)


def test_load_analytics_deduplication(tmp_path: Path):
    """Les URLs dupliquées (deux blocs) ne créent qu'une entrée."""
    from growth_system.leads_attribution import load_analytics_post_dates
    analytics_dir = tmp_path / "analytics"
    analytics_dir.mkdir()
    xlsx_path = analytics_dir / "AggregateAnalytics_2026-06-01.xlsx"
    _make_analytics_xlsx(xlsx_path, [
        {"url": "https://www.linkedin.com/posts/post-1", "date": "28/04/2026"},
    ])
    result = load_analytics_post_dates(analytics_dir)
    assert len(result) == 1


def test_load_analytics_picks_most_recent(tmp_path: Path):
    """Quand plusieurs fichiers, le plus récent (tri nom) est utilisé."""
    from growth_system.leads_attribution import load_analytics_post_dates
    analytics_dir = tmp_path / "analytics"
    analytics_dir.mkdir()
    _make_analytics_xlsx(
        analytics_dir / "AggregateAnalytics_2026-05-01.xlsx",
        [{"url": "https://post-old", "date": "01/01/2026"}],
    )
    _make_analytics_xlsx(
        analytics_dir / "AggregateAnalytics_2026-06-01.xlsx",
        [{"url": "https://post-new", "date": "15/05/2026"}],
    )
    result = load_analytics_post_dates(analytics_dir)
    assert "https://post-new" in result
    assert "https://post-old" not in result


def test_endpoint_fiabilite_date_post_exact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Les posts dont la date est dans MEILLEURS POSTS reçoivent fiabilite_date_post='exacte...'."""
    import growth_system.web.api as api_mod
    analytics_dir = tmp_path / "analytics"
    analytics_dir.mkdir()

    # posts.csv avec permalink = URL exacte
    permalink = "https://www.linkedin.com/posts/post-exact"
    posts_df = pd.DataFrame([
        {"permalink": permalink, "days_ago": 10, "eng_score": 200.0,
         "text": "Post exact", "format": "Text only", "likes": 20, "comments": 3, "shares": 1},
    ])
    posts_df.to_csv(tmp_path / "posts.csv", index=False)

    # Fichier analytics avec date exacte pour ce post
    _make_analytics_xlsx(
        analytics_dir / "AggregateAnalytics_2026-06-20.xlsx",
        [{"url": permalink, "date": "12/06/2026"}],
    )

    monkeypatch.setattr(api_mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(api_mod, "LEADS_CSV", tmp_path / "leads.csv")
    monkeypatch.setattr(api_mod, "POSTS_CSV", tmp_path / "posts.csv")
    monkeypatch.setattr(api_mod, "SNAPSHOT_DIR", tmp_path / "snapshots")
    monkeypatch.setattr(api_mod, "ANALYTICS_DIR", analytics_dir)

    csv_bytes = _make_csv_bytes([
        {"nom_prenom": "Alice", "titre": "Artiste", "date_connexion": "14 juin",
         "motivation": "post inspirant", "type_lead": "Artiste"},
    ])

    from fastapi.testclient import TestClient
    from growth_system.web.api import app
    client = TestClient(app)

    r = client.post(
        "/api/leads/attribute",
        files={"file": ("leads.csv", csv_bytes, "text/csv")},
        data={"reference_date": "2026-06-22", "window_days": "14"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["n_leads_attributed"] == 1
    detail = body["detail"][0]
    assert detail["fiabilite_date_post"] == "exacte (MEILLEURS POSTS)"


def test_endpoint_fiabilite_date_post_approximate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Un post absent de MEILLEURS POSTS reçoit fiabilite_date_post='approximative...'."""
    import growth_system.web.api as api_mod

    # Pas de répertoire analytics → aucun enrichissement
    analytics_dir = tmp_path / "analytics"  # n'existe pas

    permalink = "https://www.linkedin.com/posts/post-approx"
    posts_df = pd.DataFrame([
        {"permalink": permalink, "days_ago": 5, "eng_score": 100.0,
         "text": "Post approx", "format": "Text only", "likes": 5, "comments": 1, "shares": 0},
    ])
    posts_df.to_csv(tmp_path / "posts.csv", index=False)

    monkeypatch.setattr(api_mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(api_mod, "LEADS_CSV", tmp_path / "leads.csv")
    monkeypatch.setattr(api_mod, "POSTS_CSV", tmp_path / "posts.csv")
    monkeypatch.setattr(api_mod, "SNAPSHOT_DIR", tmp_path / "snapshots")
    monkeypatch.setattr(api_mod, "ANALYTICS_DIR", analytics_dir)

    csv_bytes = _make_csv_bytes([
        {"nom_prenom": "Bob", "titre": "Galerie", "date_connexion": "Aujourd'hui",
         "motivation": "", "type_lead": "Galerie"},
    ])

    from fastapi.testclient import TestClient
    from growth_system.web.api import app
    client = TestClient(app)

    r = client.post(
        "/api/leads/attribute",
        files={"file": ("leads.csv", csv_bytes, "text/csv")},
        data={"reference_date": "2026-06-22", "window_days": "7"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["n_leads_attributed"] == 1
    detail = body["detail"][0]
    assert detail["fiabilite_date_post"] == "approximative (reconstruite)"


# ---------------------------------------------------------------------------
# normalize_post_id — dédoublonnage de clé
# ---------------------------------------------------------------------------


def test_normalize_post_id_bare_id():
    assert normalize_post_id("7464594459641548802") == "7464594459641548802"


def test_normalize_post_id_full_url():
    url = "https://www.linkedin.com/posts/foo_bar-7464594459641548802"
    assert normalize_post_id(url) == "7464594459641548802"


def test_normalize_post_id_urn():
    urn = "urn:li:activity:7464594459641548802"
    assert normalize_post_id(urn) == "7464594459641548802"


def test_normalize_post_id_url_with_utm():
    url = "https://www.linkedin.com/posts/foo-7464594459641548802?utm_source=share&rcm=ACoAAA"
    assert normalize_post_id(url) == "7464594459641548802"


def test_normalize_post_id_fallback_no_digits():
    raw = "pas-de-chiffres"
    assert normalize_post_id(raw) == "pas-de-chiffres"


def test_upsert_lead_deduplication_by_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """POST /api/leads avec deux formes d'URL pour le même post → une seule ligne dans leads.csv."""
    import growth_system.web.api as api_mod
    monkeypatch.setattr(api_mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(api_mod, "LEADS_CSV", tmp_path / "leads.csv")

    from fastapi.testclient import TestClient
    from growth_system.web.api import app
    client = TestClient(app)

    activity_id = "7464594459641548802"
    url_clean = f"https://www.linkedin.com/posts/timothee-roy-{activity_id}"
    url_utm = f"https://www.linkedin.com/posts/timothee-roy-{activity_id}?utm_source=share"

    r1 = client.post("/api/leads", json={"post_id": url_clean, "qualified_contacts": 3})
    assert r1.status_code == 200
    assert r1.json()["action"] == "created"
    assert r1.json()["post_id"] == activity_id

    r2 = client.post("/api/leads", json={"post_id": url_utm, "qualified_contacts": 5})
    assert r2.status_code == 200
    assert r2.json()["action"] == "updated"

    df = pd.read_csv(tmp_path / "leads.csv")
    assert len(df) == 1, f"Attendu 1 ligne, obtenu {len(df)}"
    assert str(df.iloc[0]["post_id"]) == activity_id
    assert int(df.iloc[0]["qualified_contacts"]) == 5
