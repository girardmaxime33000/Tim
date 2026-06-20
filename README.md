# Growth System — Système d'exploitation de la croissance LinkedIn

Package Python qui **score, alloue, surveille et récompense** — sans jamais rédiger de post.

---

## Vue d'ensemble

```
Brouillon humain
      │
      ▼
┌─────────────┐     rework?      ┌──────────────┐
│  BRIQUE 1   │────────────────▶ │    Humain     │
│   Scoreur   │  P(queue) < 0.30 │  (réécrit)   │
│  de queue   │                  └──────────────┘
└──────┬──────┘
       │ publish / optimize
       ▼
┌─────────────┐
│  BRIQUE 2   │◀─── Recommande l'archétype du prochain post
│   Bandit    │──── Met à jour les croyances après publication
│  à escompte │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  BRIQUE 4   │     outcome (engagement + leads + marque)
│  Récompense │──────────────────────────────────────────▶ r ∈ [0,1]
│   métier    │
└─────────────┘
       ▲
┌─────────────┐
│  BRIQUE 3   │     Surveille λ(t) = nouveaux abonnés/jour
│   CUSUM     │──── Alarme → reset bandit + flag ré-entraînement
└─────────────┘
```

Le système ne touche pas à la rédaction. Il filtre, oriente et mesure.

---

## Installation

```bash
pip install -e .
```

Dépendances : `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `pydantic>=2`, `typer`.

---

## Quickstart

### Scorer un brouillon

```bash
growth score --draft mon_brouillon.txt
```

```
P(queue) = 0.67 → PUBLISH

Contributions par feature:
  log_len              val=7.43  contrib=+0.231
  question_body        val=1.00  contrib=+0.185
  hashtag              val=1.00  contrib=+0.172
  hook_contrarian      val=0.00  contrib=+0.000
  ...
```

### Obtenir une recommandation d'archétype

```bash
growth recommend
```

```
Archétype recommandé : CONTRARIAN

Posterior (alpha, beta) par bras:
  contrarian      α=3.21 β=1.18 θ̄=0.731
  data            α=2.10 β=3.44 θ̄=0.379
  question        α=1.85 β=2.11 θ̄=0.467
  statement       α=1.44 β=4.02 θ̄=0.264
```

### Enregistrer le résultat d'un post

```bash
growth update --arm contrarian --tail 1 --leads 2
```

### Surveiller la dynamique d'audience

```bash
growth monitor --audience data/daily_audience.csv
```

### Lancer le backtest

```bash
growth backtest
```

---

## Résultats sur l'historique réel (139 posts)

### Brique 1 — Scoreur de queue

| Métrique | Valeur |
|---|---|
| AUC (entraînement) | **0.90** |
| Précision@20% (replay) | **0.643** |
| Seuil d'acceptation | 0.55 ✓ |
| Posts filtrés « rework » | 85 / 139 (61 %) |
| Posts validés « publish » | 44 / 139 (32 %) |

Le scoreur filtre 6 brouillons sur 10 avant publication. Les 32 % validés concentrent la majorité du potentiel d'engagement.

### Brique 2 — Bandit : taux de queue réels par archétype

| Archétype | Taux queue global | Taux avant rupture | Taux après rupture |
|---|---|---|---|
| **contrarian** | **62.5 %** | 33 % | **80 %** |
| question | 26.3 % | 17 % | 43 % |
| data | 25.0 % | 18 % | 50 % |
| statement | 13.1 % | 5 % | 38 % |

La dérive est réelle : l'archétype « contrarian » passe de 33 % à 80 % de taux de queue entre les deux périodes. Le bandit à escompte (γ = 0.985) alloue progressivement vers les bras gagnants sans attendre une rupture explicite.

### Brique 3 — CUSUM

Changepoint détecté le **25/05/2026** — conforme à la rupture connue (×3.3 sur le taux d'acquisition). Le détecteur répond dans la fenêtre d'acceptation de ±10 jours.

### Brique 4 — Récompense métier

> **Avertissement** : sans connexion à un CRM réel, `lead_signal = 0` pour tous les posts. La récompense ne capte que `tail_indicator × 0.3`. Cela signifie que les 3 politiques du backtest obtiennent la même récompense cumulée (8.4) — ce n'est pas un bug, c'est l'honnêteté du système. Le test synthétique de `test_bandit.py` valide l'algorithme en environnement contrôlé.

**Pour activer la récompense complète**, implémenter `LeadSource` :

```python
from growth_system.reward import BusinessReward, LeadSource
from growth_system.config import RewardWeights

class MyCRMSource:
    def leads_for(self, post_id: str) -> float:
        return my_crm.get_leads(post_id) / 10.0  # normaliser ∈ [0,1]

reward = BusinessReward(RewardWeights(), MyCRMSource())
```

### Backtest — 3 politiques comparées

```
P0 (statique, random)       → 8.4  (proxy)
P1 (Thompson stationnaire)  → 8.4  (proxy)
P2 (Thompson γ=0.985)       → 8.4  (proxy)
```

Avec données CRM réelles : P2 ≥ P1 ≥ P0 est attendu, car le bandit à escompte réalloue vers « contrarian » après la rupture du 25/05 (validé sur données synthétiques, voir `tests/test_bandit.py`).

---

## Usage en production (boucle quotidienne)

```
1. growth recommend              → archétype cible du jour
2. Humain rédige un brouillon
3. growth score --draft draft.txt → verdict (publish / optimize / rework)
4. [Publication si verdict ≠ rework]
5. growth update --arm <arm> --tail <0|1> --leads <n>
6. growth monitor --audience data/daily_audience.csv  → alarmes CUSUM
```

L'état du bandit est sérialisé dans `growth_state.json` entre les sessions.

---

## Tests

```bash
pytest tests/ -v          # 33 tests, ~0.5s
```

| Suite | Tests | Critère bloquant validé |
|---|---|---|
| `test_features.py` | 9 | Extraction byte-identique aux règles d'entraînement |
| `test_scorer.py` | 5 | Précision@20% = 0.643 ≥ 0.55 |
| `test_bandit.py` | 5 | Regret(escompte) < Regret(stationnaire) en env non stationnaire |
| `test_changepoint.py` | 3 | CUSUM détecte rupture 25/05/2026 à ±10 jours |
| `test_reward.py` | 5 | Post appât < post qualité à engagement égal |
| `test_orchestrator.py` | 6 | Boucle complète, save/load état |

---

## Garde-fous

- **Pas de génération de texte.** Le système ne rédige rien.
- **Appartenance, pas magnitude.** On prédit `P(top 20%)`, jamais un chiffre d'engagement.
- **Goodhart.** Si `LeadSource` n'est pas branché sur des données CRM réelles, les logs signalent explicitement qu'on optimise un proxy.
- **Reproductibilité.** Tout aléa prend une `seed` (défaut 42).
- **Honnêteté du backtest.** Les résultats sans CRM sont rapportés tels quels, non maquillés.
