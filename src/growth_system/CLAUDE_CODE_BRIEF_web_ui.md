# Brief Claude Code — Interface web de l'OS de croissance (intégration extension + 4 briques)

> À coller dans Claude Code à la racine du repo `growth_system` existant (celui qui contient déjà le package Python, `data/posts.csv`, `data/daily_audience.csv`, `data/scorer_model.json`, et les 4 briques testées). Tu ajoutes ici une **interface web locale** qui pilote tout depuis le navigateur, y compris l'ingestion des posts via l'extension Chrome. Lis l'intégralité avant de coder.

---

## 0. Objectif

Construire une **application web locale** (backend + frontend) qui expose **tous les outils du système depuis une seule interface** :

1. **Mettre à jour le profil de Timothée** en récupérant ses posts LinkedIn via l'extension Chrome fournie, sans manipuler de fichier à la main.
2. **Scorer un brouillon** (brique 1) — coller le texte, obtenir P(queue), verdict, suggestions.
3. **Recommander l'archétype** du prochain post et **enregistrer le résultat** d'un post publié (brique 2).
4. **Surveiller la dynamique** d'audience et voir les alarmes de rupture (brique 3).
5. **Saisir les leads** manuellement (brique 4, sans CRM) et **lancer le backtest**.

Contrainte absolue : **tout doit être faisable depuis l'interface web.** Aucune commande terminale requise pour l'usage quotidien.

Le système ne rédige aucun post. Il scrape, score, alloue, surveille, récompense.

---

## 1. Architecture cible

```
┌──────────────────────────────────────────────────────────┐
│                   NAVIGATEUR (Chrome)                      │
│                                                            │
│  ┌────────────────────┐         ┌──────────────────────┐  │
│  │  Interface web      │         │  Extension Chrome     │  │
│  │  (localhost:8000)   │         │  (li-posts-export)    │  │
│  │                     │         │  scrape via voyager   │  │
│  │  Onglets:           │         └──────────┬───────────┘  │
│  │  - Profil/MAJ       │                    │ POST posts    │
│  │  - Scoreur          │ ◀──────────────────┘ (bridge)     │
│  │  - Bandit           │                                    │
│  │  - Monitoring       │                                    │
│  │  - Leads/Backtest   │                                    │
│  └─────────┬───────────┘                                    │
└────────────┼────────────────────────────────────────────────┘
             │ HTTP (REST)
             ▼
┌──────────────────────────────────────────────────────────┐
│         BACKEND LOCAL  FastAPI  (localhost:8000)           │
│  enveloppe le package growth_system (4 briques)            │
│  + endpoint /api/ingest (reçoit les posts scrapés)         │
│  + persistance: data/posts.csv, growth_state.json,         │
│    data/leads.csv                                          │
└──────────────────────────────────────────────────────────┘
```

Stack : **FastAPI** (backend, réutilise le package `growth_system` déjà écrit) + **frontend léger** (HTML/CSS/JS vanilla ou Vue 3 sans build lourd ; pas de framework massif). Tout tourne en local, lancé par une seule commande `growth serve` (à ajouter à la CLI `typer` existante) qui démarre uvicorn et ouvre le navigateur.

---

## 2. Le problème d'intégration de l'extension (à lire avant de coder)

L'extension fournie (`li-posts-export v2.6.2`, Manifest V3) :
- scrape les posts d'un profil via l'API **voyager** de LinkedIn (l'utilisateur entre une URL de profil) ;
- exporte en **CSV / XLSX / JSON** par téléchargement navigateur ;
- colonnes produites : `fullName, publishDate, likes, comments, shares, permalink, text, videoUrl, documentUrl, profileUrl` — **format déjà compatible** avec le pipeline (c'est la source des `LI_POSTS_*.csv` historiques).

**Contrainte technique incontournable** : une page web locale **ne peut pas lancer programmatiquement une extension Chrome** (sandbox navigateur, pas d'API pour ça). Il faut donc un **pont**. Deux options ; implémente l'**Option A** par défaut, garde l'**Option B** en repli documenté.

### Option A (par défaut, recommandée) — Ingestion par dépôt de fichier + dossier surveillé

L'interface web propose un panneau « Mettre à jour le profil » qui :
1. affiche l'URL du profil de Timothée (préremplie, configurable) avec un bouton « Ouvrir le profil + extension » qui ouvre l'URL LinkedIn dans un nouvel onglet ;
2. rappelle à l'utilisateur de cliquer sur l'extension et de lancer l'export **JSON** (format le plus fiable) ;
3. fournit une **zone de glisser-déposer** (et un sélecteur de fichier) où l'utilisateur dépose le fichier exporté ;
4. envoie le fichier à `POST /api/ingest` qui le normalise et met à jour `data/posts.csv`.

C'est robuste, ne dépend d'aucune modification de l'extension, et reste « tout depuis l'interface web » (le seul geste hors-UI est le clic sur l'extension, inévitable).

**Bonus** (si simple) : surveiller le dossier `~/Downloads` (ou un dossier configuré) côté backend ; dès qu'un fichier `LI_POSTS_*.json|csv|xlsx` apparaît, proposer son ingestion en un clic dans l'UI (watcher `watchdog`). Ne pas auto-ingérer sans confirmation.

### Option B (repli, documentée non activée par défaut) — Pont par POST depuis l'extension

L'extension est tierce et propriétaire : **ne pas la décompiler ni la republier**. Documenter seulement, dans `docs/extension_bridge.md`, comment un utilisateur avancé pourrait ajouter un petit **userscript** (Tampermonkey) ou une extension maison minimale qui POSTe le JSON scrapé vers `http://localhost:8000/api/ingest`. Ne pas implémenter de modification de l'extension fournie.

---

## 3. Backend — endpoints REST (FastAPI, `src/growth_system/web/api.py`)

Tous les endpoints enveloppent le package existant. Aucune logique métier nouvelle : réutilise `TailScorer`, `DiscountedThompsonBandit`, `CusumDetector`, `BusinessReward`, `GrowthSystem`, `backtest`.

| Méthode | Route | Rôle |
|---|---|---|
| `POST` | `/api/ingest` | Reçoit un fichier (JSON/CSV/XLSX) ou un payload JSON de posts scrapés. Normalise, déduplique, met à jour `data/posts.csv`. Renvoie un résumé (n nouveaux, n total, plage de dates). |
| `GET` | `/api/profile` | État du profil : nb de posts, dernier post, date de dernière MAJ, stats résumées. |
| `POST` | `/api/score` | Body `{text, format}`. Renvoie `{p_queue, verdict, contributions[], suggestions[]}`. |
| `GET` | `/api/recommend` | Archétype recommandé + posteriors (alpha,beta,theta) par bras. |
| `POST` | `/api/update` | Body `{arm, tail, leads}`. Met à jour bandit + reward, persiste `growth_state.json` et `data/leads.csv`. |
| `GET` | `/api/monitor` | Lance le CUSUM sur `daily_audience.csv`, renvoie la série λ(t), les alarmes (dates), le flag « ré-apprendre ». |
| `GET` | `/api/leads` | Liste des leads saisis (depuis `data/leads.csv`). |
| `POST` | `/api/leads` | Ajoute/édite une entrée lead `{post_id, qualified_contacts}`. |
| `POST` | `/api/backtest` | Lance le backtest, renvoie les courbes (séries JSON) + chemin du PNG. |
| `GET` | `/api/health` | Vérifie présence des 3 fichiers data + intégrité (cf. §11.2 du brief précédent). |

Règles :
- CORS limité à `localhost` (l'extension/userscript du pont POSTe depuis le navigateur).
- Toute écriture de fichier est **atomique** (écrire dans un tmp puis renommer) pour ne pas corrompre `posts.csv`.
- Validation pydantic stricte sur tous les bodies.

---

## 4. Normalisation à l'ingestion (`/api/ingest` — critique)

L'export de l'extension doit devenir un `posts.csv` valide pour le scoreur. Étapes :

1. **Lire** le fichier (détecter JSON/CSV/XLSX). Filtrer sur `fullName == "Timothée Roy"` (ne garder que les posts propres, pas le fil).
2. **Mapper** les colonnes : `text → text`, `publishDate → publishDate_raw`, `likes/comments/shares → idem`, `permalink → permalink`, `videoUrl/documentUrl/images → déduire `format` (Video / Document-Carousel / Image / Text only)`.
3. **Recalculer** `eng_score = likes + 3·comments + 2·shares`.
4. **Parser** `publishDate_raw` (formats relatifs `1d/2w/3mo/1y`) en `days_ago`, puis trier chronologiquement et réindexer `post_index` (0 = plus ancien).
5. **Dédupliquer** par `permalink` (ou hash du texte si permalink absent) : fusionner avec l'existant, garder la version la plus récente des métriques.
6. **Écrire** `data/posts.csv` atomiquement. Conserver une copie horodatée dans `data/snapshots/posts_AAAAMMJJ-HHMM.csv` (historique des MAJ).
7. **Re-scorer** tout le corpus en mémoire et renvoyer le nouveau résumé (précision@20 %, nb publish/rework).

**Garde-fou** : ne jamais écraser `posts.csv` si l'ingestion produit < 50 posts (probable scrape partiel) ; avertir l'utilisateur et demander confirmation explicite via l'UI.

⚠️ Le texte doit être **intégral** (pas tronqué). Si l'export ne contient que des extraits, refuser l'ingestion avec un message clair (le scoreur serait invalidé).

---

## 5. Frontend — interface (5 onglets)

Une SPA légère, une seule page, navigation par onglets. Design sobre, dense, orienté usage quotidien. Palette dark/orange (#1A1A1A / #E8540A) cohérente avec les livrables existants.

### Onglet 1 — « Profil / Mise à jour »
- URL du profil (préremplie, éditable), bouton « Ouvrir profil + extension ».
- Instructions courtes (3 étapes) pour lancer l'export JSON depuis l'extension.
- Zone glisser-déposer + sélecteur de fichier → `POST /api/ingest`.
- Après ingestion : résumé (nouveaux posts, total, plage de dates, précision@20 % recalculée). Bandeau de confirmation si < 50 posts.
- État du profil (`GET /api/profile`) : dernière MAJ, nb posts, mini-stats.

### Onglet 2 — « Scoreur de queue »
- Grande zone de texte (colle le brouillon) + sélecteur de format.
- Bouton « Scorer » → `POST /api/score`.
- Affichage : jauge P(queue) colorée (vert ≥0.55 / ambre / rouge), verdict, tableau des contributions signées par feature, liste de suggestions priorisées.
- Scoring en temps réel (debounce) souhaitable, comme l'outil HTML autonome déjà livré (`scoreur_de_queue.html`) — tu peux t'en inspirer pour l'UX.

### Onglet 3 — « Bandit / Archétype »
- Bouton « Recommander » → archétype du jour + tableau des posteriors (α, β, θ̄) par bras, trié.
- Formulaire « Enregistrer un résultat » : sélecteur d'archétype, toggle `tail` (0/1), champ `leads` (entier), bouton « Mettre à jour » → `POST /api/update`.
- Rappel visuel : « sans saisie du résultat, le système n'apprend pas. »

### Onglet 4 — « Monitoring »
- Graphe de λ(t) (nouveaux abonnés/jour, lissé 14 j) + impressions/jour, depuis `GET /api/monitor`.
- Marqueurs verticaux aux dates d'alarme CUSUM. Bandeau si une alarme récente impose un ré-entraînement.
- Mini-tableau : totaux (impressions, abonnés), taux d'engagement, dernière alarme.

### Onglet 5 — « Leads & Backtest »
- **Leads** : tableau éditable (`post_id`, `qualified_contacts`) lu/écrit via `/api/leads`. C'est la source de récompense métier sans CRM (saisie manuelle). Explication courte : 1 lead = 1 conversation qualifiée (MP, commentaire mot-clé, RDV).
- **Backtest** : bouton « Lancer » → `POST /api/backtest` → affiche les courbes (récompense cumulée P0/P1/P2, regret, évolution des θ, alarme CUSUM superposée) et un résumé chiffré. Avertissement explicite si `leads.csv` est vide (récompense = proxy reach).

---

## 6. CLI — ajout (`growth serve`)

Ajouter à la CLI typer existante :
```
growth serve [--host 127.0.0.1] [--port 8000] [--open/--no-open]
```
Démarre uvicorn, sert le frontend (fichiers statiques montés sur `/`), et ouvre le navigateur par défaut. C'est la **seule commande** que l'utilisateur lance ; tout le reste se fait dans l'UI.

---

## 7. Persistance & état

- `data/posts.csv` — corpus (mis à jour par l'ingestion, snapshots horodatés dans `data/snapshots/`).
- `growth_state.json` — état du bandit (posteriors), sérialisé après chaque `/api/update`.
- `data/leads.csv` — saisie manuelle des leads (colonnes `post_id, qualified_contacts`).
- `data/daily_audience.csv` — série d'audience (MAJ manuelle ou future ingestion analytics ; hors périmètre ici).

Tout doit survivre à un redémarrage du serveur.

---

## 8. Tests & critères d'acceptation (bloquants)

- `pytest` vert ; les tests existants des 4 briques ne régressent pas.
- **Ingestion** : un export JSON de l'extension (fixture fournie ou échantillon réel) produit un `posts.csv` valide (139+ lignes, texte intégral, `eng_score` recalculé, dédupliqué, trié). Test sur fichier fixture.
- **Garde-fou ingestion** : un export < 50 posts ou à texte tronqué est refusé (test).
- **API** : chaque endpoint répond avec le schéma attendu (tests FastAPI `TestClient`).
- **Atomicité** : une ingestion interrompue ne corrompt pas `posts.csv` (test : tmp + rename).
- **E2E léger** : un script de test démarre le serveur, ingère une fixture, score un brouillon, recommande, met à jour, lance le monitoring — tout via HTTP.

---

## 9. Garde-fous (non négociables)

- **Ne pas décompiler, modifier ni republier l'extension Chrome fournie** (tierce, propriétaire). L'intégration se fait par ingestion de son export (Option A) ; le pont POST (Option B) est documenté pour un composant maison, pas une altération de l'extension.
- **Local only.** Le serveur n'écoute que sur `127.0.0.1`. Aucune donnée ne sort de la machine. CORS restreint à localhost.
- **Pas de génération de texte.** L'UI ne rédige aucun post.
- **Texte intégral obligatoire.** Refuser toute ingestion à texte tronqué (scoreur invalidé sinon).
- **Honnêteté de la récompense.** Tant que `leads.csv` est vide, l'UI et le backtest affichent clairement « optimisation sur proxy de reach, pas sur le métier ».
- **Écritures atomiques** sur tous les fichiers de données.
- **Pas de scraping côté backend.** Le backend n'appelle jamais LinkedIn directement (c'est le rôle de l'extension, qui gère l'authentification de l'utilisateur). Le backend ne fait qu'ingérer ce que l'extension a exporté.

---

## 10. Ordre de construction (jalons)

1. `web/api.py` : endpoints `/api/health`, `/api/profile`, `/api/score`, `/api/recommend`, `/api/update` (enveloppent le package existant) + tests TestClient.
2. `/api/ingest` + normalisation + garde-fous + snapshots + tests fixture.
3. `/api/monitor`, `/api/leads`, `/api/backtest`.
4. Frontend : les 5 onglets, branchés sur les endpoints.
5. `growth serve` (uvicorn + static + ouverture navigateur).
6. `docs/extension_bridge.md` (Option B documentée) + `README` mis à jour avec le flux web.
7. Test E2E léger.

Valide chaque jalon par ses tests avant le suivant. Commence par le jalon 1 : il rend l'UI utile immédiatement (scorer/recommander/mettre à jour) même avant l'ingestion automatisée.

---

## 11. Format de l'export extension (référence pour la normalisation)

L'extension `li-posts-export` exporte des fichiers nommés `LI_POSTS_urn_li_fsd_profile_<id>_<n>.{json,csv,xlsx}` avec, par post, au minimum :
`fullName, subtitle, publishDate, likes, comments, shares, permalink, text, profileUrl` et, selon le post, `videoUrl, documentUrl, images`.
- `publishDate` est **relatif** (`1d`, `2w`, `3mo`, `1y`) — le parser doit le convertir en `days_ago`.
- Le fil peut contenir des posts d'autres auteurs : **filtrer sur `fullName == "Timothée Roy"`**.
- `text` est l'élément critique : vérifier qu'il est intégral (longueur plausible, pas coupé à ~160 caractères).

Échantillons réels disponibles : les fichiers `LI_POSTS_*.csv` déjà présents dans l'historique du projet servent de fixtures de test pour la normalisation.
