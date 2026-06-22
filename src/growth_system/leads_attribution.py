"""Attribution de leads LinkedIn → posts.

Pipeline d'inférence temporelle : étant donné une date de connexion LinkedIn
(extraite de l'export), on cherche le post le plus engageant dans une fenêtre
glissante précédant cette date.

IMPORTANT : ce module ne modifie jamais leads.csv sur disque.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

_WEEKDAYS_FR = {
    "lundi": 0,
    "mardi": 1,
    "mercredi": 2,
    "jeudi": 3,
    "vendredi": 4,
    "samedi": 5,
    "dimanche": 6,
}

_MONTHS_FR = {
    "janv": 1, "janv.": 1, "janvier": 1,
    "févr": 2, "févr.": 2, "février": 2,
    "mars": 3,
    "avr": 4, "avr.": 4, "avril": 4,
    "mai": 5,
    "juin": 6,
    "juil": 7, "juil.": 7, "juillet": 7,
    "août": 8,
    "sept": 9, "sept.": 9, "septembre": 9,
    "oct": 10, "oct.": 10, "octobre": 10,
    "nov": 11, "nov.": 11, "novembre": 11,
    "déc": 12, "déc.": 12, "décembre": 12,
}


# ---------------------------------------------------------------------------
# Dataclass résultat
# ---------------------------------------------------------------------------

@dataclass
class AttributionResult:
    post_id: str
    post_date: date
    eng_score: float


# ---------------------------------------------------------------------------
# Fonctions publiques
# ---------------------------------------------------------------------------

def parse_lead_date(raw: str, reference_date: date) -> date | None:
    """Parse une date LinkedIn brute en date Python.

    Formats supportés :
    - "Aujourd'hui" → reference_date
    - jour de semaine fr ("Lundi"…"Dimanche") → occurrence la plus récente
      STRICTEMENT AVANT reference_date
    - "JJ mmm." ex. "19 avr.", "6 juin" → date absolue, année la plus récente
      telle que date <= reference_date
    - forme composée "31 mai / mardi" → utilise la partie numérique "31 mai"
    - "À vérifier" ou non reconnu → None (jamais une valeur par défaut)
    """
    if not raw or not isinstance(raw, str):
        return None

    s = raw.strip()

    # 1. "Aujourd'hui"
    if s.lower() in ("aujourd'hui", "aujourd’hui", "aujourdhui"):
        return reference_date

    # 2. Forme composée "31 mai / mardi" → extraire la partie numérique avant "/"
    if "/" in s:
        parts = s.split("/")
        # prendre la première partie qui ressemble à "JJ mmm"
        candidate = parts[0].strip()
        result = _parse_numeric_date(candidate, reference_date)
        if result is not None:
            return result
        # sinon essayer les autres parties numériques
        for part in parts[1:]:
            result = _parse_numeric_date(part.strip(), reference_date)
            if result is not None:
                return result
        # essayer les jours de semaine en dernier recours
        for part in parts:
            result = _parse_weekday(part.strip(), reference_date)
            if result is not None:
                return result
        return None

    # 3. Jour de semaine
    wd = _parse_weekday(s, reference_date)
    if wd is not None:
        return wd

    # 4. Date numérique "JJ mmm."
    nd = _parse_numeric_date(s, reference_date)
    if nd is not None:
        return nd

    # 5. Non reconnu → None (inclut "À vérifier")
    return None


def _parse_weekday(s: str, reference_date: date) -> date | None:
    """Renvoie l'occurrence la plus récente du jour de semaine STRICTEMENT AVANT reference_date."""
    key = s.lower().strip(".")
    if key not in _WEEKDAYS_FR:
        return None
    target_wd = _WEEKDAYS_FR[key]
    # Reculer depuis reference_date - 1 jusqu'à trouver le bon jour
    delta = (reference_date.weekday() - target_wd) % 7
    if delta == 0:
        delta = 7  # même jour que reference_date → prendre le précédent
    return reference_date - timedelta(days=delta)


def _parse_numeric_date(s: str, reference_date: date) -> date | None:
    """Parse "JJ mmm" ou "JJ mmm." → date absolue avec année déduite."""
    # Normaliser
    s = s.strip()

    # Pattern : un ou deux chiffres + espace + nom de mois (avec ou sans point)
    m = re.match(r'^(\d{1,2})\s+([a-zéûôàèêîù]+\.?)$', s, re.IGNORECASE)
    if not m:
        return None
    day = int(m.group(1))
    month_str = m.group(2).lower()
    # essayer avec et sans point final
    month = _MONTHS_FR.get(month_str) or _MONTHS_FR.get(month_str.rstrip("."))
    if month is None:
        return None

    # Déduire l'année : la plus récente telle que date <= reference_date
    year = reference_date.year
    try:
        candidate = date(year, month, day)
    except ValueError:
        return None

    if candidate > reference_date:
        # Essayer l'année précédente
        try:
            candidate = date(year - 1, month, day)
        except ValueError:
            return None

    return candidate


def attribute_lead_to_post(
    lead_date: date,
    posts: pd.DataFrame,  # colonnes: post_id, post_date (date), eng_score (float)
    window_days: int = 7,
) -> AttributionResult | None:
    """Attribue un lead au post le plus engageant dans la fenêtre [lead_date - window_days, lead_date].

    Les bornes sont incluses. Retourne None si aucun post dans la fenêtre.
    """
    if posts.empty:
        return None

    start = lead_date - timedelta(days=window_days)
    mask = (posts["post_date"] >= start) & (posts["post_date"] <= lead_date)
    window = posts[mask]

    if window.empty:
        return None

    best_idx = window["eng_score"].idxmax()
    best = window.loc[best_idx]
    return AttributionResult(
        post_id=str(best["post_id"]),
        post_date=best["post_date"],
        eng_score=float(best["eng_score"]),
    )


def load_analytics_post_dates(analytics_dir: Path) -> dict[str, date]:
    """Charge les dates exactes de posts depuis le fichier AggregateAnalytics le plus récent.

    Lit la feuille "MEILLEURS POSTS" du fichier AggregateAnalytics_*.xlsx le plus récent
    dans analytics_dir, extrait les colonnes "URL du post" et "Date de publication du post"
    (deux blocs côte à côte → déduplique par URL), et retourne {url: date}.

    Retourne {} sans erreur si :
    - analytics_dir n'existe pas
    - aucun fichier AggregateAnalytics_*.xlsx présent
    - la feuille est absente ou illisible
    Le fallback approximatif existant prend le relais dans tous ces cas.
    """
    if not analytics_dir.exists():
        return {}

    xlsx_files = sorted(analytics_dir.glob("AggregateAnalytics_*.xlsx"))
    if not xlsx_files:
        return {}

    latest = xlsx_files[-1]  # tri lexicographique AAAA-MM-JJ → le plus récent en dernier

    try:
        df = pd.read_excel(latest, sheet_name="MEILLEURS POSTS", header=0)
    except Exception:
        return {}

    # Deux blocs côte à côte → les noms de colonnes peuvent être dupliqués.
    # pandas les suffixe automatiquement (.1, .2 …). On repère toutes les colonnes
    # dont le nom (sans suffixe numérique) contient les mots-clés attendus.
    url_cols = [c for c in df.columns if "url" in str(c).lower()]
    date_cols = [c for c in df.columns if "date de publication" in str(c).lower()]

    result: dict[str, date] = {}
    from datetime import datetime as _dt

    for url_col, date_col in zip(url_cols, date_cols):
        for url_val, date_val in zip(df[url_col], df[date_col]):
            if pd.isna(url_val) or pd.isna(date_val):
                continue
            url_str = str(url_val).strip()
            if not url_str:
                continue
            try:
                d = _dt.strptime(str(date_val).strip(), "%d/%m/%Y").date()
            except ValueError:
                continue
            if url_str not in result:  # première occurrence = déduplique
                result[url_str] = d

    return result


def aggregate_attribution(attributed_leads: pd.DataFrame) -> pd.DataFrame:
    """Agrège les leads attribués par post.

    Input DataFrame colonnes attendues :
      post_id_attribue, poids, nom_prenom, type_lead

    Output par post_id_attribue :
      raw_count, weighted_count (arrondi 1 décimale), lead_names (list), lead_types (dict)
    """
    if attributed_leads.empty:
        return pd.DataFrame(columns=["post_id_attribue", "raw_count", "weighted_count",
                                     "lead_names", "lead_types"])

    rows = []
    for post_id, group in attributed_leads.groupby("post_id_attribue"):
        raw_count = len(group)
        weighted_count = round(float(group["poids"].sum()), 1)
        lead_names = group["nom_prenom"].tolist()
        # Compter les types
        lead_types: dict[str, int] = {}
        for t in group["type_lead"]:
            lead_types[str(t)] = lead_types.get(str(t), 0) + 1
        rows.append({
            "post_id_attribue": post_id,
            "raw_count": raw_count,
            "weighted_count": weighted_count,
            "lead_names": lead_names,
            "lead_types": lead_types,
        })

    return pd.DataFrame(rows)
