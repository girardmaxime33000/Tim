# Extension bridge — Option B (documentation, non activée par défaut)

> **Garde-fou absolu :** ne pas décompiler, modifier ni republier l'extension
> `li-posts-export`. Ce document décrit uniquement comment un composant *maison*
> peut envoyer les données scrapées au backend local. L'extension fournie est
> tierce et propriétaire — elle n'est pas modifiée.

---

## Contexte

L'interface web (Option A, défaut) intègre les exports via glisser-déposer :
l'utilisateur lance l'export depuis l'extension, dépose le fichier dans l'UI,
et `POST /api/ingest` normalise tout.

L'Option B décrit comment automatiser cette étape en ajoutant un composant
maison (userscript ou mini-extension) qui envoie directement les données
JSON à `http://localhost:8000/api/ingest` sans manipulation de fichier.

---

## Architecture Option B

```
LinkedIn (navigateur)
      │
      │ Scrape via voyager API
      ▼
Extension li-posts-export  (inchangée)
      │
      │ Exporte JSON en mémoire
      ▼
Userscript / mini-extension maison  ←── COMPOSANT MAISON UNIQUEMENT
      │
      │ POST http://localhost:8000/api/ingest
      ▼
Backend Growth System (127.0.0.1:8000)
```

---

## Option B1 — Userscript Tampermonkey

Un userscript peut intercepter le téléchargement déclenché par l'extension
et renvoyer le contenu à l'API locale avant qu'il atteigne le disque.

### Principe

L'extension li-posts-export appelle `URL.createObjectURL(blob)` puis
déclenche un `<a download>`. Un userscript injecté sur les pages LinkedIn
peut surveiller `URL.createObjectURL` et intercepter le Blob correspondant.

### Code de référence (Tampermonkey, à adapter)

```javascript
// ==UserScript==
// @name         Growth System Bridge
// @namespace    http://localhost:8000
// @version      1.0
// @match        https://www.linkedin.com/*
// @grant        GM_xmlhttpRequest
// @connect      localhost
// ==/UserScript==

(function () {
  'use strict';

  // Intercept URL.createObjectURL to detect export blobs
  const origCreate = URL.createObjectURL.bind(URL);
  URL.createObjectURL = function (blob) {
    const url = origCreate(blob);

    // Only forward CSV/JSON blobs that look like a LI_POSTS export
    if (blob.type === 'text/csv' || blob.type === 'application/json') {
      blob.text().then(text => {
        // Quick check: must contain 'permalink' and 'fullName'
        if (!text.includes('permalink') || !text.includes('fullName')) return;

        const filename = 'LI_POSTS_bridge.' + (blob.type === 'text/csv' ? 'csv' : 'json');
        const formData = new FormData();
        formData.append('file', new Blob([text], { type: blob.type }), filename);
        formData.append('author', 'Timothée Roy');  // adapter si besoin

        GM_xmlhttpRequest({
          method: 'POST',
          url: 'http://localhost:8000/api/ingest',
          data: formData,
          onload: (r) => {
            if (r.status === 200) {
              const body = JSON.parse(r.responseText);
              console.log('[Growth Bridge] Ingestion OK —', body.n_posts, 'posts,', body.n_new, 'nouveaux');
            } else {
              console.error('[Growth Bridge] Erreur ingestion:', r.responseText);
            }
          },
          onerror: () => console.error('[Growth Bridge] Backend inaccessible — serveur démarré ?'),
        });
      });
    }

    return url;
  };
})();
```

### Installation

1. Installer [Tampermonkey](https://www.tampermonkey.net/) dans Chrome.
2. Créer un nouveau script, coller le code ci-dessus, adapter `author`.
3. Démarrer le backend (`growth serve`) **avant** de lancer l'export LinkedIn.
4. Lancer l'export depuis l'extension — le bridge intercepte et ingère automatiquement.

### Limites

- Dépend de l'implémentation interne de `URL.createObjectURL` dans l'extension,
  qui peut changer sans préavis lors d'une mise à jour.
- Tampermonkey doit avoir accès à `localhost` (autoriser dans les permissions
  du script : `@connect localhost`).
- Ne fonctionne que si le backend est déjà démarré.

---

## Option B2 — Mini-extension Chrome maison

Une extension Chrome minimale (Manifest V3) peut déclarer un
`content_script` sur `linkedin.com` avec les mêmes droits qu'un userscript,
mais avec une permission `host_permissions` explicite sur `localhost`.

### Structure minimale

```
growth-bridge/
├── manifest.json
├── content.js
└── background.js   (facultatif — pour les fetch cross-origin en MV3)
```

**`manifest.json`**

```json
{
  "manifest_version": 3,
  "name": "Growth System Bridge",
  "version": "1.0",
  "description": "Envoie les exports li-posts-export à l'API locale.",
  "permissions": ["scripting"],
  "host_permissions": [
    "https://www.linkedin.com/*",
    "http://localhost:8000/*",
    "http://127.0.0.1:8000/*"
  ],
  "content_scripts": [{
    "matches": ["https://www.linkedin.com/*"],
    "js": ["content.js"],
    "run_at": "document_start"
  }]
}
```

**`content.js`** — même logique que le userscript, en remplaçant
`GM_xmlhttpRequest` par `fetch('http://localhost:8000/api/ingest', ...)`.

En Manifest V3, les fetch cross-origin depuis un content script sont
autorisés si `localhost` est déclaré dans `host_permissions`.

### Installation (mode développeur)

1. Ouvrir `chrome://extensions` → activer le **Mode développeur**.
2. Cliquer **Charger l'extension non empaquetée** → sélectionner le dossier `growth-bridge/`.
3. Démarrer `growth serve`, ouvrir LinkedIn, lancer l'export.

---

## Endpoint `/api/ingest` — rappel du contrat

```
POST http://localhost:8000/api/ingest
Content-Type: multipart/form-data

file      : fichier CSV, JSON ou XLSX
author    : "Timothée Roy"  (filtre fullName, optionnel)
dry_run   : "false"         (mettre "true" pour valider sans écrire)
```

Réponse (HTTP 200) :

```json
{
  "n_raw": 252,
  "n_posts": 133,
  "n_new": 5,
  "duplicates_removed": 114,
  "tail_rate": 0.226,
  "date_range": "1–2555 jours",
  "snapshot": "data/snapshots/posts_2026-06-21.csv",
  "destination": "data/posts.csv"
}
```

Erreur (HTTP 422) si garde-fou déclenché :

```json
{ "detail": "Seulement 12 posts après filtrage — minimum requis : 50." }
```

---

## Garde-fous rappelés

| Règle | Raison |
|---|---|
| Ne pas modifier l'extension li-posts-export | Tierce, propriétaire, Manifest V3 signé |
| Serveur local uniquement (127.0.0.1) | Aucune donnée ne sort de la machine |
| CORS restreint à localhost | Le backend refuse les requêtes d'origines tierces |
| Pas de scraping côté backend | Le backend n'appelle jamais LinkedIn directement |
| Écriture atomique | `.tmp` → rename, snapshot horodaté avant écrasement |
