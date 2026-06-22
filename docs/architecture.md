# Architecture — Les 4 briques et leurs flux

## Schéma général

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          BOUCLE DE DÉCISION (1 tour = 1 post)           │
│                                                                          │
│  ┌─────────────┐   archétype    ┌──────────────────────────────────┐    │
│  │  BRIQUE 2   │─────────────▶  │  Rédacteur humain                │    │
│  │   Bandit    │                │  (écrit dans l'archétype cible)  │    │
│  │  à escompte │◀───────────────│                                  │    │
│  └─────────────┘  update(arm,r) └──────────────┬─────────────────┘    │
│         ▲                                        │ brouillon             │
│         │ r ∈ [0,1]                              ▼                      │
│  ┌──────┴──────┐              ┌──────────────────────────────────────┐  │
│  │  BRIQUE 4   │◀─ outcome ── │  BRIQUE 1 — Scoreur de queue        │  │
│  │  Récompense │              │  P(queue) → verdict                  │  │
│  │   métier    │              │  • publish  (P ≥ 0.55) → publier    │  │
│  └─────────────┘              │  • optimize (P ≥ 0.30) → améliorer  │  │
│                               │  • rework   (P < 0.30) → réécrire  │  │
│                               └──────────────────────────────────────┘  │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  BRIQUE 3 — CUSUM  (surveillance en parallèle)                 │    │
│  │  Input : λ(t) = nouveaux abonnés/jour (lissé 14 j)             │    │
│  │  Alarme → (a) flag ré-entraînement scoreur                     │    │
│  │          → (b) reset partiel des posteriors du bandit           │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────┘
```

## Brique 1 — Scoreur de queue (`scorer.py`, `features.py`)

**Rôle :** filtre de production. Décide si un brouillon vaut la peine d'être publié.

**Entrée :** texte brut du brouillon.

**Sortie :**
- `P(queue)` ∈ [0,1] — probabilité d'appartenir au top 20 % d'engagement
- `verdict` ∈ {publish, optimize, rework}
- `explain()` — contribution signée par feature

**Pipeline :**
```
texte → extract_features() → standardisation (mean/scale) → log-odds → sigmoïde
```

**Features (8) :** `log_len`, `tags`, `emoji`, `hashtag`, `question_body`, `hook_contrarian`, `hook_question`, `hook_data`.

Toutes les règles sont **déterministes et documentées** dans `features.py`. Aucune ambiguïté : reproduire exactement les features d'entraînement est une condition de validité du modèle.

**Modèle :** régression logistique pré-entraînée (`scorer_model.json`), AUC = 0.90, précision@20% = 0.643 sur l'historique réel.

**Seuils :**
- `publish` : P ≥ 0.55
- `optimize` : 0.30 ≤ P < 0.55
- `rework` : P < 0.30

---

## Brique 2 — Bandit à escompte (`bandit.py`, `archetypes.py`)

**Rôle :** allocation. Recommande l'archétype à tester au prochain tour.

**4 bras (archetypes.py) :**
| Bras | Règle de classification |
|---|---|
| `contrarian` | 1re ligne matche le pattern regex contrarian |
| `question` | 1re ligne se termine par `?` |
| `data` | 1re ligne contient un chiffre |
| `statement` | aucun des 3 ci-dessus |

**Algorithme : Thompson Sampling Beta-Bernoulli à escompte géométrique**

À chaque update :
```
α_a ← γ · α_a + α₀   (pour tous les bras)
β_a ← γ · β_a + β₀
α_arm ← α_arm + reward
β_arm ← β_arm + (1 - reward)
```

Sélection : `argmax θ_a ~ Beta(α_a, β_a)`

**γ = 0.985** → demi-vie d'oubli ≈ 46 posts ≈ 6.5 semaines (voir `docs/parameters.md` pour justification).

**Variante fenêtre glissante (`SlidingWindowThompsonBandit`) :** W = 35 tours. Plus interprétable, comportement équivalent.

**Garde-fou anti-essoufflement :** si un archétype se fatigue, son θ posterior baisse → réallocation naturelle vers d'autres bras. Aucun mécanisme explicite nécessaire.

---

## Brique 3 — Détecteur de rupture (`changepoint.py`)

**Rôle :** surveillance. Lève une alarme quand le régime de croissance change.

**Input :** λ(t) = `new_followers` lissé sur 14 jours.

**Algorithme : CUSUM bilatéral**
```
S⁺ = max(0, S⁺ + (x_t - μ₀ - k))   # détecte hausse
S⁻ = max(0, S⁻ - (x_t - μ₀ + k))   # détecte baisse
Alarme si S⁺ ≥ h ou S⁻ ≥ h → ChangePoint(date, direction, statistic)
```

**Calibration automatique** sur une fenêtre initiale (60 jours par défaut) :
- μ₀ = moyenne de la série de calibration
- k = 0.5 · σ (demi-amplitude tolérée)
- h = 4.5 · σ (seuil d'alarme)

**Validation sur données réelles :** alarme levée le 25/05/2026, date exacte de la rupture connue (×3.3 sur le taux d'acquisition).

**Action à l'alarme :**
1. Flag « ré-entraîner le scoreur » (ré-entraînement hors-ligne du modèle logistique)
2. `bandit.reset_arm(a)` pour tous les bras (oubli accéléré)
3. Recalibrer le CUSUM sur la nouvelle baseline

---

## Brique 4 — Récompense métier (`reward.py`)

**Rôle :** ancrage. Garantit que l'optimisation sert les leads, pas la vanité métrique.

**Formule :**
```
reward = w_tail · tail_indicator
       + w_lead · lead_signal
       − w_brand · brand_penalty
```

| Composant | Source | Poids défaut |
|---|---|---|
| `tail_indicator` | eng_score ≥ q80 | 0.3 |
| `lead_signal` | LeadSource (CRM ou proxy) | 0.6 |
| `brand_penalty` | heuristiques texte | 0.4 |

**`lead_signal` (le vrai objectif)** — à brancher sur :
- CRM : nb de leads qualifiés attribués au post, normalisé ∈ [0,1]
- Proxy par défaut : 0.0 (avec avertissement dans les logs)

**`brand_penalty`** — heuristiques :
- Appât à commentaires (`commentez`, `taguez`, etc.) : +0.5
- Sur-tagging (> 15 `@mentions`) : +0.5

**Interface `LeadSource` (Protocol) :** remplacer `ProxyLeadSource` par toute classe qui implémente `leads_for(post_id: str) -> float`.

---

## Orchestrateur (`orchestrator.py`)

La classe `GrowthSystem` câble les 4 briques :

```python
# Tour complet
arm = system.recommend()                          # Brique 2
p, verdict = system.score_draft(draft_text)       # Brique 1
r = system.record_outcome(text, eng, id, t, λ)   # Brique 4 + 2 + 3
```

**État persisté** (`save_state` / `load_state`) en JSON — les posteriors du bandit et l'historique des changepoints survivent entre les sessions.

---

## Flux de données

```
posts.csv ──────────────────▶ features.py ──▶ scorer.py
                                                  │
scorer_model.json ────────────────────────────────┘

daily_audience.csv ─────────▶ changepoint.py

CRM / proxy ────────────────▶ reward.py

Tout ───────────────────────▶ orchestrator.py ──▶ bandit.py
```
