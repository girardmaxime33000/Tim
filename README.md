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

> **Règle d'or :** toutes les commandes du projet (`growth`, `pytest`,
> `python scripts/…`) doivent s'exécuter depuis le **même environnement
> virtuel**. Ne pas installer le package dans le Python global ni via
> Homebrew — cela crée des environnements parallèles incompatibles.

### Première installation (clone propre)

```bash
# 1. Créer et activer le venv à la racine du repo
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows (PowerShell)

# 2. Installer le package en mode éditable avec toutes ses dépendances
pip install -e .

# 3. Vérifier
growth --help                    # CLI disponible
python -m pytest tests/ -q       # tous les tests doivent passer
python scripts/deduplicate_leads.py  # script opérationnel
```

Le venv est ignoré par git (`.gitignore` contient `.venv/`).

### Sessions suivantes

```bash
# Toujours activer le venv avant de travailler
source .venv/bin/activate

growth serve                     # interface web
```

### Dépendances

Toutes les dépendances sont déclarées dans `pyproject.toml` (section
`[project] dependencies`) — `numpy`, `pandas`, `scipy`, `scikit-learn`,
`matplotlib`, `plotly`, `fastapi`, `pydantic>=2`, `typer`, `openpyxl`,
`uvicorn`, `python-multipart`. Pas de `requirements.txt` séparé à
maintenir.

### Résolution du conflit Homebrew / Python système (macOS)

Si `which growth` → `/opt/homebrew/bin/growth` mais
`python3 scripts/…` échoue avec `ModuleNotFoundError` :

```bash
# 1. Désinstaller l'éventuelle install globale Homebrew/pip
pip3 uninstall growth-system -y 2>/dev/null || true
brew uninstall growth-system 2>/dev/null || true

# 2. Repartir du venv
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# 3. Confirmer que tout pointe vers le même Python
which growth          # doit afficher …/.venv/bin/growth
which python          # doit afficher …/.venv/bin/python
python -c "import pandas; print('OK')"
```

---

## Interface web — démarrage rapide

```bash
growth serve
```

Ouvre automatiquement `http://127.0.0.1:8000` dans le navigateur.
**Tout le workflow quotidien se fait depuis l'interface** — aucune autre
commande requise.

```
growth serve [--host 127.0.0.1] [--port 8000] [--no-open]
```

### Flux web quotidien

```
Onglet Profil/MAJ  → déposer l'export CSV de l'extension li-posts-export
                     → ingestion + snapshot automatique
Onglet Scoreur     → coller le brouillon → P(queue), verdict, suggestions
Onglet Bandit      → recommander l'archétype du jour
                   → enregistrer le résultat après publication
Onglet Monitoring  → graphes audience, alarmes CUSUM, dashboards décisionnels
Onglet Leads       → saisir les leads manuellement + lancer le backtest
```

### Mettre à jour le corpus LinkedIn

1. Ouvrir l'extension **li-posts-export** sur le profil de Timothée Roy.
2. Lancer l'export **CSV** (format recommandé — texte intégral garanti).
3. Déposer le fichier dans l'onglet **Profil/MAJ** de l'interface web.
4. L'API ingère, déduplique (max engagement gagne sur doublon), crée un
   snapshot horodaté dans `data/snapshots/`, et met à jour `data/posts.csv`.

> Un export partiel (< 50 posts après filtrage) est refusé avec un message
> explicite. Le corpus ne peut que croître — un re-export partiel n'écrase
> jamais les posts existants.

---

## Quickstart CLI

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

## Usage en production

### Via l'interface web (recommandé)

```bash
growth serve   # tout depuis le navigateur
```

### Via la CLI

```
1. growth recommend              → archétype cible du jour
2. Humain rédige un brouillon
3. growth score --draft draft.txt → verdict (publish / optimize / rework)
4. [Publication si verdict ≠ rework]
5. growth update --arm <arm> --tail <0|1> --leads <n>
6. growth monitor --audience data/daily_audience.csv  → alarmes CUSUM
```

L'état du bandit est sérialisé dans `growth_state.json` entre les sessions.

### Pont extension → API (Option B, avancé)

Pour envoyer les exports directement depuis le navigateur sans manipulation
de fichier, voir [`docs/extension_bridge.md`](docs/extension_bridge.md)
(userscript Tampermonkey ou mini-extension Chrome maison).

---

## Tests

```bash
pytest tests/ -v          # 130 tests
```

| Suite | Tests | Critère bloquant validé |
|---|---|---|
| `test_features.py` | 9 | Extraction byte-identique aux règles d'entraînement |
| `test_scorer.py` | 5 | Précision@20% = 0.643 ≥ 0.55 |
| `test_bandit.py` | 5 | Regret(escompte) < Regret(stationnaire) en env non stationnaire |
| `test_changepoint.py` | 3 | CUSUM détecte rupture 25/05/2026 à ±10 jours |
| `test_reward.py` | 5 | Post appât < post qualité à engagement égal |
| `test_orchestrator.py` | 6 | Boucle complète, save/load état |
| `test_web_api.py` | 31 | Tous les endpoints jalon 1 (health/profile/score/recommend/update) |
| `test_web_ingest.py` | 39 | Ingest sur fixture réelle, merge, garde-fous, atomicité |
| `test_web_jalon3.py` | 27 | Monitor / Leads / Backtest |
| `test_decisional_p1.py` | 19 | Bandit live (posteriors Beta) + couverture leads + table par archétype |
| `test_decisional_p2.py` | 12 | Fraîcheur corpus : last\_ingest\_at, last\_retrain\_at, changepoint\_unaddressed |
| `test_decisional_p3.py` | 17 | score\_log.csv, rework\_rate\_30d, bandes CUSUM, precision\_trend |

| `test_leads_attribution.py` | 22 | Attribution leads : parse dates FR, fenêtre, agrégation, endpoints |

```bash
pytest tests/ -v          # 241 tests
```

---

## Dashboards décisionnels (onglet Monitoring)

### Bandit live — posteriors Beta
Courbes Beta en temps réel pour chaque archétype (Thompson Sampling γ = 0.985).  
Recommandation active mise en évidence. Source : `data/growth_state.json`.

### Couverture leads
Bannière orange si < 50 % des posts LinkedIn sont rattachés à un lead, verte sinon.  
Table de répartition leads par archétype.

### Carte de fraîcheur
Trois KPI : dernière ingestion (`last_ingest.json`), dernier ré-entraînement  
(`scorer_model.json → trained_at`), dernier changepoint CUSUM.  
Flag `changepoint_unaddressed` : alerte si un changepoint n'a pas été suivi d'un ré-entraînement.

### Journal de scoring (`score_log.csv`)
Chaque appel à `/api/score` est journalisé : timestamp, extrait, P(queue), verdict, format.  
Taux de rework sur 30 jours visible dans la carte Monitoring.

### Bandes d'accélération CUSUM
Zones colorées sur le graphe de croissance pour chaque phase d'accélération détectée.

### Tendance précision@20% (`precision_log.csv`)
Courbe historique de la précision@20% du scoreur, mise à jour à chaque ingestion.

---

## Attribution leads LinkedIn → posts (outil de rattrapage ponctuel)

L'onglet **Leads & Backtest** expose une section *Importer des leads depuis un export de connexions*.

### Principe

L'export LinkedIn des connexions contient une date par connexion mais **pas de lien direct vers un post**.  
Le pipeline infère l'attribution par fenêtre temporelle : pour chaque connexion, on cherche  
le post le plus engageant dans une fenêtre `[date_connexion − window_days, date_connexion]` (défaut 7 jours).

### Utilisation

1. Exporter les connexions LinkedIn (Données > Obtenir une copie de vos données > Connexions).
2. Déposer le CSV dans la section dédiée, ajuster la fenêtre et la date de référence.
3. Cliquer **Analyser** — *aucun fichier n'est modifié à cette étape*.
4. Vérifier le détail (colonne Fiabilité = "estimée (fenêtre Nj)").
5. Choisir la stratégie (`add` ou `replace`) puis **Appliquer à leads.csv**.

Un snapshot horodaté est automatiquement créé dans `data/snapshots/` avant toute écriture.

### Limites (§2)

- **Estimation uniquement.** La correspondance connexion → post est une approximation temporelle,  
  pas une donnée LinkedIn officielle. Toujours vérifier la cohérence avant d'appliquer.
- **Biais de sélection.** Un post très engageant dans la fenêtre peut capter des leads  
  qui ne l'ont pas réellement vu.
- **Dates LinkedIn partielles.** LinkedIn n'expose que le jour de la semaine ou "Aujourd'hui"  
  pour les connexions récentes ; les dates absolues sont déduites par inférence d'année.
- **Pas de suppression.** L'outil ajoute ou remplace des entrées dans `leads.csv` mais  
  ne supprime jamais de lignes existantes.

---

## Garde-fous

- **Pas de génération de texte.** Le système ne rédige rien.
- **Appartenance, pas magnitude.** On prédit `P(top 20%)`, jamais un chiffre d'engagement.
- **Goodhart.** Si `LeadSource` n'est pas branché sur des données CRM réelles, les logs signalent explicitement qu'on optimise un proxy.
- **Reproductibilité.** Tout aléa prend une `seed` (défaut 42).
- **Honnêteté du backtest.** Les résultats sans CRM sont rapportés tels quels, non maquillés.
