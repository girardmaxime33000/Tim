# Brief Claude Code — Système d'exploitation robuste de la croissance LinkedIn

> À coller dans Claude Code à la racine d'un nouveau repo. Tu (Claude Code) es chargé de construire le système décrit ci-dessous, de bout en bout, testé et documenté. Lis l'intégralité avant d'écrire la moindre ligne. Les fichiers de données sont déjà dans le repo (voir §11) ; si l'un manque, demande son chemin plutôt que d'inventer des données.

---

## 0. Objectif

Construire un **système d'exploitation robuste de la croissance** d'un profil LinkedIn, composé de **4 briques articulées** :

1. **Scoreur de queue** — filtre la production éditoriale (publier seulement les brouillons à fort potentiel).
2. **Bandit à escompte** — décide quel *archétype* de post tester en priorité, tout en évitant l'essoufflement d'audience.
3. **Détecteur de rupture (CUSUM)** — surveille la dynamique de croissance et signale le moment de ré-apprendre.
4. **Fonction de récompense métier** — garantit que l'optimisation sert l'objectif commercial (leads qualifiés) et non la vanité métrique (engagement brut).

Le livrable est un **package Python installable, testé, documenté**, accompagné d'un **backtest** qui prouve que la politique adaptative bat une politique statique sur l'historique réel.

Ce n'est pas un générateur de texte. Aucune brique ne rédige de post. Le système **score, alloue, surveille et récompense** ; la rédaction reste humaine.

---

## 1. Contexte minimal (suffisant pour exécuter)

- Le profil suivi (« Timothée Roy », fondateur d'InRealArt) publie ~1 post/jour sur LinkedIn. On dispose d'un historique de **139 publications** avec, pour chacune : texte intégral, date relative, likes, commentaires, partages, et un score d'engagement composite.
- On dispose aussi de séries d'audience quotidiennes sur **365 jours** (impressions/jour, interactions/jour, nouveaux abonnés/jour).
- Un **scoreur de queue déjà entraîné** est fourni sous forme de `scorer_model.json` (régression logistique, AUC 0,90). Tu ne le ré-entraînes pas : tu l'**implémentes et l'intègres**. Son format est décrit en §4.1.

### Propriétés statistiques établies (elles fixent les paramètres, ne les redécouvre pas)

- **Queue lourde** : l'engagement par post suit une loi à queue lourde (exposant ≈ 1,9). Le top 20 % des posts concentre 57 % de l'engagement. → On ne prédit jamais la *magnitude*, seulement l'*appartenance à la queue* (binaire).
- **Auto-excitation** : autocorrélation des impressions = 0,82 à 1 jour, demi-vie de mémoire ≈ **8 jours**.
- **Non-stationnarité** : rupture de régime détectée vers le **25/05/2026** (taux d'acquisition ×3,3). La valeur des archétypes dérive : l'accroche « contrarian » passe de 0 % à 71 % de taux de queue entre la période ancienne et récente. → Tout modèle figé devient obsolète ; d'où l'escompte (brique 2) et le détecteur (brique 3).
- **Échelle de dérive structurelle** : ~6 à 10 semaines. C'est l'horizon que l'escompte doit cibler (entre les 8 jours de mémoire courte et les ~3 mois de la rupture).

---

## 2. Stack & conventions

- **Python 3.11+**, gestion par `pyproject.toml` (PEP 621), pas de `setup.py`.
- Dépendances : `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `openpyxl` (lecture xlsx), `pydantic` (config typée), `pytest`, `typer` (CLI). Pas de framework lourd.
- Typage statique partout (`mypy --strict` doit passer). Docstrings format NumPy.
- Aucune dépendance réseau à l'exécution. Reproductibilité : toute source d'aléa prend une `seed`.
- Style : `ruff` pour lint+format. Code en anglais (identifiants, commentaires) ; documentation utilisateur (`README`, `docs/`) en français.

---

## 3. Architecture du repo

```
growth_system/
├── pyproject.toml
├── README.md                       # FR, quickstart + schéma d'articulation
├── data/
│   ├── posts.csv                   # 139 posts, texte intégral (FOURNI, voir §11)
│   ├── daily_audience.csv          # 365 jours: impressions/interactions/abonnés (FOURNI)
│   ├── scorer_model.json           # scoreur pré-entraîné (FOURNI, déjà en possession)
│   └── raw/                         # sources brutes optionnelles (xlsx, voir §11.3)
│       ├── AggregateAnalytics_Timothee_Roy_2025-06-20_2026-06-19.xlsx
│       └── InRealArt_LinkedIn_Positioning_Audit.xlsx
├── src/growth_system/
│   ├── __init__.py
│   ├── config.py                   # config pydantic centralisée + valeurs par défaut
│   ├── data_io.py                  # chargement posts.csv / daily_audience.csv (+ fallback xlsx)
│   ├── features.py                 # extraction déterministe des variables (réplique exacte du scoreur)
│   ├── archetypes.py               # définition et encodage des bras (hook × format)
│   ├── scorer.py                   # BRIQUE 1
│   ├── bandit.py                   # BRIQUE 2
│   ├── changepoint.py              # BRIQUE 3
│   ├── reward.py                   # BRIQUE 4
│   ├── orchestrator.py             # boucle de décision articulant les 4 briques
│   ├── backtest.py                 # rejoue l'historique, mesure le regret vs statique
│   └── cli.py                      # typer: score / recommend / backtest / monitor
├── tests/
│   ├── test_data_io.py
│   ├── test_features.py
│   ├── test_scorer.py
│   ├── test_bandit.py
│   ├── test_changepoint.py
│   ├── test_reward.py
│   └── test_orchestrator.py
└── docs/
    ├── architecture.md             # le schéma des 4 briques et leurs flux
    └── parameters.md               # justification de chaque paramètre par les données
```

---

## 4. Spécification des 4 briques

### 4.1 BRIQUE 1 — Scoreur de queue (`scorer.py`, `features.py`)

**Rôle.** Étant donné un brouillon (texte + format prévu), renvoyer `P(queue)` ∈ [0,1] : probabilité d'appartenir au top 20 % d'engagement. Sert de **filtre de production**.

**Ne pas ré-entraîner.** Charger `data/scorer_model.json`, dont le schéma est :
```json
{
  "features": ["log_len","tags","emoji","hashtag","question_body",
               "hook_contrarian","hook_question","hook_data"],
  "mean":  [...], "scale": [...],
  "coef":  [...], "intercept": -1.55,
  "auc": 0.90, "base_rate": 0.201, "eng_q80": 98, "eng_median": 40
}
```

**Extraction des features (`features.py`) — règles DÉTERMINISTES, à respecter à l'identique** (elles doivent reproduire celles qui ont servi à l'entraînement, sinon le modèle est invalide) :

| feature | règle |
|---|---|
| `log_len` | `ln(1 + nombre_de_caractères)` |
| `tags` | nb d'occurrences du motif regex `[A-ZÀ-Ý][a-zà-ÿ]+\s+[A-ZÀ-Ý]` (artistes tagués) |
| `emoji` | nb de caractères dans la plage emoji Unicode `[\U0001F300-\U0001FAFF\u2600-\u27BF]` |
| `hashtag` | 1 si le texte contient `#`, sinon 0 |
| `question_body` | 1 si le texte contient `?`, sinon 0 |
| `hook_contrarian` | 1 si la 1re ligne (≤200 car.) matche `^\s*(Le mythe\|L.illusion\|Et si\|Le march\|Pourquoi\|Personne\|On a \|L.inflation\|Je ne crois\|Arr[êe]tez\|En \d{4})` (insensible casse) |
| `hook_question` | 1 si la 1re ligne se termine par `?` |
| `hook_data` | 1 si la 1re ligne contient un chiffre |

**Scoring.** Standardiser chaque feature `(x - mean) / scale`, calculer le log-odds `intercept + Σ coef·x_std`, appliquer la sigmoïde.

**API attendue.**
```python
class TailScorer:
    def __init__(self, model_path: str) -> None: ...
    def features(self, text: str) -> dict[str, float]: ...
    def score(self, text: str) -> float:                  # P(queue)
    def explain(self, text: str) -> list[Contribution]:   # contribution signée par feature
    def verdict(self, text: str) -> Literal["publish","optimize","rework"]:  # seuils 0.55 / 0.30
```

**Test d'acceptation.** Sur les 139 posts de `data/posts.csv` (colonne `text`), recalculer le score et vérifier que la précision@20 % ≥ 0,55 (référence mesurée : 0,64, seuil de queue `eng_score ≥ 98`). Vérifier que `features()` est byte-identique à une implémentation de référence sur 5 cas fournis dans le test.

---

### 4.2 BRIQUE 2 — Bandit à escompte (`bandit.py`, `archetypes.py`)

**Rôle.** À chaque tour (= prochain post à produire), recommander **l'archétype** à tester en priorité, en équilibrant exploitation du meilleur archétype courant et exploration, **sous escompte** pour suivre la dérive et éviter l'essoufflement.

**Bras = archétypes (`archetypes.py`).** Ensemble **restreint** (4 à 6 bras max — petit volume, ne pas fragmenter). Croisement minimal : `hook ∈ {contrarian, data, question, statement}` (mêmes règles que `features.py`). Optionnellement enrichir avec `format ∈ {image, video}` si le volume le permet, mais **commencer tabulaire à 4 bras** (hook seul) et ne raffiner que si les tests le justifient. Fournir `archetype_of(text, format) -> ArmId`.

**Algorithme : Thompson sampling Beta-Bernoulli à escompte.**
- Récompense par bras (succès = post en queue, ou récompense métier de la brique 4 — voir orchestration).
- Posterior Beta(α_a, β_a) par bras `a`.
- **Escompte géométrique** : avant chaque mise à jour, `α_a ← γ·α_a + α0`, `β_a ← γ·β_a + β0` (priors α0=β0=1). `γ` réglé pour une **demi-vie d'oubli de ~6 à 10 semaines** ; avec ~1 post/jour, `γ ≈ 0.985` par tour (config, défaut 0.985, justifié dans `docs/parameters.md`).
- Sélection : échantillonner `θ_a ~ Beta(α_a, β_a)`, recommander `argmax θ_a`.
- Variante **fenêtre glissante** (W ≈ 35 tours) en option, plus interprétable.

**API attendue.**
```python
class DiscountedThompsonBandit:
    def __init__(self, arms: list[ArmId], gamma: float = 0.985,
                 prior: tuple[float,float] = (1.0,1.0), seed: int|None = None) -> None: ...
    def recommend(self) -> ArmId: ...
    def update(self, arm: ArmId, reward: float) -> None: ...   # reward ∈ [0,1]
    def posterior(self) -> dict[ArmId, tuple[float,float]]: ...
    def reset_arm(self, arm: ArmId) -> None: ...               # appelé par la brique 3
```

**Garde-fou anti-essoufflement.** L'escompte gère la dérive lente. Documenter que la fatigue d'un archétype se traduit par une baisse de son θ posterior, donc une réallocation naturelle vers d'autres bras — le vérifier dans le backtest.

**Test d'acceptation.** Sur un environnement synthétique non stationnaire (bras gagnant change à mi-parcours), le bandit à escompte doit atteindre un **regret cumulé strictement inférieur** à celui d'un Thompson sampling stationnaire.

---

### 4.3 BRIQUE 3 — Détecteur de rupture (`changepoint.py`)

**Rôle.** Surveiller en ligne la dynamique (λ(t) = nouveaux abonnés/jour, lissé 14 j, depuis `daily_audience.csv`) et **lever une alarme** au changement de régime, ce qui déclenche : (a) ré-apprentissage du scoreur, (b) reset partiel des posteriors du bandit.

**Algorithme : CUSUM bilatéral** sur λ(t).
- `S+ = max(0, S+ + (x_t - μ0 - k))` et `S- = max(0, S- - (x_t - μ0 + k))`.
- `μ0` = moyenne de référence (calibration initiale) ; `k` ≈ 0,5·σ ; seuil d'alarme `h` ≈ 4–5·σ. Exposer `k`, `h` en config.
- À l'alarme : renvoyer `ChangePoint(date, direction, statistic)` et réinitialiser S+/S-.
- Variante **BOCPD** en option ; CUSUM par défaut.

**Validation sur données réelles.** Retrouver la rupture connue autour du **25/05/2026** sur `daily_audience.csv` (colonne `new_followers`). Test d'acceptation principal de la brique.

**API attendue.**
```python
class CusumDetector:
    def __init__(self, mu0: float, k: float, h: float) -> None: ...
    def update(self, x: float, t: date) -> ChangePoint | None: ...
    def calibrate(self, series: pd.Series) -> None: ...   # fixe mu0, sigma sur fenêtre initiale
```

---

### 4.4 BRIQUE 4 — Fonction de récompense métier (`reward.py`)

**Rôle.** Transformer le résultat observé d'un post en **récompense alignée sur l'objectif commercial** (leads qualifiés), pas sur l'engagement brut. Brique anti-Goodhart.

**Principe.**
```
reward = w_tail · tail_indicator          # le post a-t-il atteint la queue (reach)
       + w_lead · lead_signal             # conversion (commentaires mot-clé, clics CTA, leads CRM)
       − w_brand · brand_penalty          # pénalité hors-marque (anti-Goodhart)
```
- `tail_indicator` ∈ {0,1} : observé (`eng_score ≥ 98`).
- `lead_signal` ∈ [0,1] : **le vrai objectif**. En prod, branché sur le CRM (leads qualifiés attribués au post) ou, à défaut, proxy documenté (CTA mot-clé déclenché, nb de MP qualifiés). Source branchable (interface `LeadSource`).
- `brand_penalty` ∈ [0,1] : pénalise les mécaniques hors-marque (appât à commentaires pur, provocation, sur-tagging > 15). Heuristiques configurables.
- Poids défaut `w_tail=0.3, w_lead=0.6, w_brand=0.4` — la conversion domine le reach, la marque est une contrainte forte. Config + justification dans `docs/parameters.md`.

**API attendue.**
```python
class BusinessReward:
    def __init__(self, weights: RewardWeights, lead_source: LeadSource) -> None: ...
    def compute(self, post: PostOutcome) -> float: ...   # ∈ [0,1] après clip
class LeadSource(Protocol):
    def leads_for(self, post_id: str) -> float: ...      # branchable CRM ; proxy par défaut
```

**Garde-fou.** Documenter : si `lead_source` n'est pas branché sur des données réelles, le système optimise un proxy, pas l'objectif réel — le signaler dans les logs et le README.

---

## 5. Orchestration (`orchestrator.py`)

Boucle de décision par tour (un tour = un post) :

```
1.  bandit.recommend()                    → archétype cible du prochain post
2.  [humain rédige un brouillon dans cet archétype]
3.  scorer.score(draft)                   → P(queue)
        - si verdict == "rework"          → renvoyer au rédacteur (ne pas publier)
        - si "optimize"/"publish"         → continuer
4.  [post publié]
5.  observer l'issue (engagement, leads)  → PostOutcome
6.  reward.compute(outcome)               → r ∈ [0,1]
7.  bandit.update(archétype, r)           → met à jour les croyances (avec escompte)
8.  changepoint.update(λ_t)               → si ChangePoint :
        - flag « ré-apprendre le scoreur » (ré-entraînement hors-ligne du logistique)
        - bandit.reset_arm(...) sur les bras concernés (oubli accéléré)
```

Classe `GrowthSystem` avec mode **interactif** (un tour à la fois) et mode **batch** (rejouer une séquence, pour le backtest). État sérialisable (JSON) pour persistance.

Le système ne rédige rien : à l'étape 2, il **émet une consigne d'archétype + les suggestions du scoreur**, et attend le brouillon humain.

---

## 6. Backtest (`backtest.py`) — la preuve

Rejouer les **139 posts dans l'ordre chronologique** (`posts.csv` est déjà trié : `post_index` 0 = plus ancien) et comparer trois politiques :

- **P0 — statique** : publie tout, archétype au hasard (baseline « intuition »).
- **P1 — bandit stationnaire** (sans escompte).
- **P2 — bandit à escompte** (le système).

Récompense cumulée via la brique 4 (à défaut de leads réels, proxy documenté = `tail_indicator` pondéré, depuis `eng_score ≥ 98`). Produire :
- courbe de récompense cumulée des 3 politiques,
- **regret cumulé** P2 vs P0,
- évolution des θ posterior par archétype (réallocation au passage de la rupture),
- moment de l'alarme CUSUM (depuis `daily_audience.csv`), superposé sur l'axe temps.

**Critère de succès** : P2 ≥ P1 ≥ P0 en récompense cumulée, et réallocation visible vers « contrarian » en période récente. Sinon, investiguer (`γ`, définition des bras) — ne pas maquiller le résultat.

Sortie : `backtest_results.png` + résumé chiffré dans le `README`.

---

## 7. CLI (`cli.py`, via typer)

```
growth score    --draft path.txt --format image      # P(queue) + verdict + suggestions
growth recommend                                      # archétype à tester au prochain tour
growth update    --arm contrarian --tail 1 --leads 2  # enregistre une issue, met à jour bandit
growth monitor   --audience data/daily_audience.csv   # passe la série au CUSUM, affiche alarmes
growth backtest                                       # rejoue l'historique, génère le rapport
```

---

## 8. Tests & critères d'acceptation (bloquants)

- `pytest` vert, couverture ≥ 85 % sur `src/`.
- `mypy --strict` et `ruff` sans erreur.
- **Brique 1** : précision@20 % ≥ 0,55 sur `posts.csv` ; `features()` conforme aux 5 cas de référence.
- **Brique 2** : regret(escompte) < regret(stationnaire) sur l'environnement synthétique non stationnaire.
- **Brique 3** : alarme CUSUM levée dans une fenêtre ±10 jours autour du 25/05/2026 sur `daily_audience.csv`.
- **Brique 4** : un post « appât hors-marque » reçoit une récompense strictement inférieure à un post « queue + lead + on-brand » de même engagement brut.
- **Backtest** : P2 ≥ P1 ≥ P0 en récompense cumulée.

---

## 9. Garde-fous (non négociables)

- **Pas de génération de texte.** Le système score/alloue/surveille/récompense. Il n'écrit pas de post.
- **Appartenance, pas magnitude.** Ne jamais prétendre prédire l'engagement chiffré. Tout repose sur le binaire « queue / pas queue ».
- **Petit échantillon.** 139 posts : régulariser, valider en croisé, ne pas multiplier bras ni features. Simplicité interprétable > sophistication.
- **Goodhart.** Récompense ancrée sur le lead métier sous contrainte de marque ; signaler quand on optimise un proxy faute de CRM.
- **Reproductibilité.** Toute exécution stochastique prend une `seed` ; backtest déterministe à seed fixée.
- **Honnêteté du backtest.** Si le système ne bat pas la baseline, le dire et investiguer, ne pas ajuster a posteriori.

---

## 10. Ordre de construction suggéré (jalons)

1. `data_io.py` + `features.py` + `scorer.py` + tests (réutilise le modèle fourni — gain immédiat).
2. `archetypes.py` + `bandit.py` + test synthétique non stationnaire.
3. `changepoint.py` + validation sur la rupture réelle.
4. `reward.py` + interface `LeadSource` (proxy par défaut).
5. `orchestrator.py` — assemblage.
6. `backtest.py` — la preuve.
7. `cli.py`, `README.md`, `docs/` — finition.

Valider chaque jalon par ses tests avant de passer au suivant.

---

## 11. Données dans le repo

### 11.1 Fichiers prêts à l'emploi (à placer dans `data/`)

- **`posts.csv`** — 139 publications, fourni. Colonnes :
  `post_index` (0 = plus ancien), `days_ago` (ancienneté en jours, ordre chronologique), `publishDate_raw` (date relative LinkedIn brute, ex. `2w`), `format`, `likes`, `comments`, `shares`, `eng_score` (= likes + 3·comments + 2·shares), `permalink`, `text` (texte intégral — **indispensable** pour les features).
- **`daily_audience.csv`** — 365 jours, fourni. Colonnes : `date` (AAAA-MM-JJ), `impressions`, `interactions`, `new_followers`.
- **`scorer_model.json`** — déjà en ta possession (schéma §4.1). Place-le dans `data/`.

> Seuil de queue : `eng_score ≥ 98` (quantile 80 % du profil) ; médiane 40. Aussi dans `scorer_model.json` (`eng_q80`, `eng_median`).

### 11.2 Cohérence à vérifier au démarrage (`data_io.py`)

- `posts.csv` : 139 lignes, `post_index` strictement croissant et aligné sur `days_ago` décroissant. Jusqu'à ~9 posts anciens ont un `text` vide (posts sans légende capturée, engagement ≈ 0) : les traiter comme texte vide, ils scorent légitimement bas (vrais négatifs) et ne doivent pas être filtrés.
- `daily_audience.csv` : 365 lignes, `date` continue sans trou, somme `impressions` ≈ 184 778, somme `new_followers` ≈ 2 395.
- `scorer_model.json` : 8 features, longueurs `mean/scale/coef` = 8.

### 11.3 Sources brutes (optionnel, dans `data/raw/`)

Deux classeurs xlsx d'origine sont disponibles pour régénérer ou enrichir les CSV. **Ne pas les utiliser par défaut** (les CSV fournis sont la source de vérité) ; `data_io.py` peut exposer une fonction de (re)génération à partir d'eux :

- **`AggregateAnalytics_Timothee_Roy_2025-06-20_2026-06-19.xlsx`** — analytics natifs LinkedIn. Feuilles : `DÉCOUVERTE` (totaux), `ENGAGEMENT` (date, impressions, interactions/jour), `MEILLEURS POSTS`, `ABONNÉS` (date, nouveaux abonnés/jour), `DONNÉES DÉMOGRAPHIQUES` (localisation, secteur, séniorité, taille d'entreprise, poste — utile si tu veux pondérer la récompense par segment d'audience). → source de `daily_audience.csv` (jointure `ENGAGEMENT` + `ABONNÉS` sur la date).
- **`InRealArt_LinkedIn_Positioning_Audit.xlsx`** — classeur d'audit éditorial. Feuilles : `Timothée Roy`, `LELOLUCE`, `Page InRealArt`, `Synthèse & Positionnement`, `Patricia Allouche`, `Annelise Stern`, `Pierre Chandès`, `Insights Audience & Vente`, `Hervé Hoint-Lecoq`. La feuille `Timothée Roy` contient le tableau post-par-post, mais **le texte y est tronqué à ~160 caractères** : pour les features, **utiliser le `text` intégral de `posts.csv`**, jamais l'extrait du classeur. Les autres feuilles sont du contexte d'analyse, non requises.

> Important : si tu régénères `posts.csv` depuis le classeur d'audit, le texte sera tronqué et le scoreur deviendra invalide. Le `posts.csv` fourni (texte intégral) est la seule source correcte pour les features.

Si l'un des fichiers de §11.1 manque, **demande son chemin** ; ne fabrique pas de données de substitution silencieusement.
