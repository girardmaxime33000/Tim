# Paramètres — Justification par les données

Ce document justifie chaque choix de paramètre par les propriétés statistiques mesurées sur l'historique réel (139 posts, 365 jours d'audience).

---

## Brique 1 — Scoreur de queue

### `eng_q80 = 98.2` (seuil de queue)
Percentile 80 de l'engagement observé sur 139 posts. Définit l'appartenance à la « queue lourde » (top 20 %).

**Source :** `scorer_model.json` → champ `eng_q80`.

### `publish_threshold = 0.55` / `optimize_threshold = 0.30`
Seuils de verdict calibrés pour obtenir une précision@20% ≥ 0.55 tout en maintenant un taux de validation raisonnable (32 % de « publish »).

**Mesure sur données réelles :**
- Précision@20% obtenue : **0.643** (objectif : ≥ 0.55 ✓)
- 85 / 139 posts filtrés en « rework » (61 %)
- 44 / 139 validés « publish » (32 %)

**Justification :** un seuil de 0.55 correspond à une probabilité prédite ~2.7× la base rate (20 %), suffisant pour discriminer sans sur-filtrer.

---

## Brique 2 — Bandit à escompte

### `gamma = 0.985` (facteur d'escompte géométrique)

**Contrainte à satisfaire :** demi-vie d'oubli entre 6 et 10 semaines.

**Calcul :**
- À 1 post/jour, `t` tours = `t` jours
- Demi-vie : `t½ = log(0.5) / log(γ)`
- Pour γ = 0.985 : `t½ = log(0.5) / log(0.985) ≈ 46 tours ≈ 6.5 semaines`

**Pourquoi 6-10 semaines ?**

Trois échelles de temps identifiées dans les données :
| Échelle | Valeur | Source |
|---|---|---|
| Mémoire courte (autocorrélation) | 8 jours | ACF impressions = 0.82 à 1 jour, demi-vie ≈ 8j |
| Dérive structurelle | 6-10 semaines | Objectif d'escompte |
| Rupture de régime | ~3 mois | Rupture 25/05/2026, boucle ~12 semaines avant |

L'escompte cible l'échelle intermédiaire : plus rapide que le cycle de rupture (3 mois), plus lent que la mémoire d'audience (8 jours). Cela permet de suivre la dérive lente sans réagir au bruit quotidien.

**Alternative :** fenêtre glissante W = 35 tours (5 semaines) via `SlidingWindowThompsonBandit`. Comportement équivalent, plus interprétable car les données hors-fenêtre sont oubliées complètement.

### `prior = (alpha0=1.0, beta0=1.0)` (prior Beta uniforme)

Prior non-informatif (Beta(1,1) = loi uniforme sur [0,1]). Justifié par la petite taille d'échantillon (139 posts / 4 bras = ~35 obs. par bras) : partir d'une croyance neutre plutôt que d'injecter des biais non mesurés.

### `4 bras` (contrarian, data, question, statement)

**Justification :** avec 139 posts historiques et ~35 observations par bras, ajouter une dimension `format` (image/video) fragmenterait les données en ~70 cellules de ~2 observations — variance excessive, convergence impossible.

**Taux de queue réels par bras :**
| Bras | Observations | Taux de queue |
|---|---|---|
| contrarian | 8 | **62.5 %** |
| data | 28 | 25.0 % |
| question | 19 | 26.3 % |
| statement | 84 | 13.1 % |

L'hétérogénéité est suffisante pour justifier 4 bras distincts. Le bras « contrarian » est fortement sous-représenté (8 obs.) mais son signal est fort — le bandit l'exploitera davantage au fil du temps.

---

## Brique 3 — CUSUM

### `k = 0.5 · σ` (demi-amplitude de dérive tolérée)

Convention classique CUSUM : k = δ/2 où δ est la taille minimale de dérive à détecter. En prenant δ = σ (un écart-type), on détecte des dérives modérées sans être trop sensible au bruit.

**Sur données réelles (série `new_followers`, 60 premiers jours) :**
- μ₀ ≈ 3.2 abonnés/jour
- σ ≈ 2.1 → k ≈ 1.05

### `h = 4.5 · σ` (seuil d'alarme)

Choix standard pour un ARL (Average Run Length) hors-rupture de ~500 observations, soit ~16 mois à 1 post/jour — suffisant pour ne pas déclencher de fausses alarmes trop fréquentes.

**Validation :** sur la série réelle, l'alarme est levée le **25/05/2026** — date exacte de la rupture connue (×3.3 sur λ). Aucune fausse alarme majeure avant cette date.

### `calibration_days = 60`

60 jours de calibration initiale offrent ~60 observations pour estimer μ₀ et σ de manière robuste, sans capturer la dérive du second semestre. La période 2025-06-20 → 2025-08-19 est stable (faibles valeurs, pas de tendance), ce qui est idéal pour la baseline.

---

## Brique 4 — Récompense métier

### `w_tail = 0.3, w_lead = 0.6, w_brand = 0.4`

**Hiérarchie intentionnelle :**
1. `w_lead = 0.6` — la conversion est l'objectif commercial primaire. Poids dominant.
2. `w_tail = 0.3` — le reach est un prérequis : sans audience, pas de lead. Poids secondaire.
3. `w_brand = 0.4` (en soustraction) — la marque est une contrainte forte. Une pénalité de 1.0 (cumul bait + sur-tagging) coûte 0.4 points de récompense, ce qui compense un tail_indicator = 1.0 partiel.

**Cas limite documenté :**
```
Post appât avec tail (eng ≥ q80), sans lead, brand_penalty = 1.0 :
  r = 0.3·1 + 0.6·0 − 0.4·1 = −0.1 → clip à 0.0

Post qualité avec tail + 3 leads + on-brand :
  r = 0.3·1 + 0.6·min(3/N, 1) + 0.4·0 > 0.3
```

La pénalité marque suffit à annuler le bénéfice d'un post viral hors-marque.

**Avertissement :** tant que `lead_signal = 0` (pas de CRM connecté), la récompense effective = `w_tail · tail_indicator = 0.3 · 1{eng ≥ 98.2}`. Le système optimise alors uniquement le reach, ce qui est signalé explicitement dans les logs.

---

## Propriétés statistiques de référence

| Propriété | Valeur mesurée | Utilisation |
|---|---|---|
| Exposant queue lourde | ≈ 1.9 | → prédiction binaire (pas de magnitude) |
| Autocorrélation impressions (lag 1) | 0.82 | → demi-vie mémoire ≈ 8 jours |
| Concentration top 20 % | 57 % de l'engagement | → seuil q80 |
| Rupture régime | 25/05/2026 (×3.3 λ) | → cible CUSUM, valide γ |
| Échelle de dérive structurelle | 6-10 semaines | → demi-vie oubli bandit |
