"""Script de migration ponctuel : dédoublonnage de data/leads.csv.

Usage :
    python scripts/deduplicate_leads.py [--apply]

Sans --apply : affiche le rapport des conflits, ne touche pas leads.csv.
Avec --apply : crée un snapshot puis écrit le fichier dédoublonné.

Règle de résolution : pour un groupe de lignes ayant le même activity_id canonique,
on garde la ligne avec le qualified_contacts le plus élevé.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Ajouter src/ au path pour importer le package sans installation
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from growth_system.leads_attribution import normalize_post_id

LEADS_CSV = _REPO_ROOT / "data" / "leads.csv"
SNAPSHOT_DIR = _REPO_ROOT / "data" / "snapshots"


def main() -> None:
    parser = argparse.ArgumentParser(description="Dédoublonner data/leads.csv par post_id canonique.")
    parser.add_argument("--apply", action="store_true", help="Écrire le fichier corrigé (snapshot créé avant).")
    args = parser.parse_args()

    if not LEADS_CSV.exists():
        print("Aucun fichier data/leads.csv trouvé. Rien à faire.")
        return

    df = pd.read_csv(LEADS_CSV, dtype={"post_id": str})
    if df.empty:
        print("data/leads.csv est vide. Rien à faire.")
        return

    # Normaliser les post_id
    df["post_id_raw"] = df["post_id"].astype(str)
    df["post_id_canonical"] = df["post_id_raw"].apply(normalize_post_id)

    # Détecter les groupes avec plusieurs variantes de clé brute
    conflicts: list[dict] = []
    clean_rows: list[dict] = []

    for canonical, group in df.groupby("post_id_canonical"):
        raw_variants = group["post_id_raw"].unique().tolist()
        if len(raw_variants) > 1 or any(r != canonical for r in raw_variants):
            # Groupe à normaliser
            conflicts.append({
                "canonical": canonical,
                "variants": raw_variants,
                "rows": group[["post_id_raw", "qualified_contacts"]].to_dict("records"),
                "n_rows": len(group),
            })
        # Résolution : garder max qualified_contacts, post_id = forme canonique
        best_qc = float(group["qualified_contacts"].max())
        clean_rows.append({"post_id": canonical, "qualified_contacts": best_qc})

    # --- Rapport ---
    total = len(df)
    n_conflict_groups = len(conflicts)
    n_conflict_rows = sum(c["n_rows"] for c in conflicts)

    print(f"\n=== Rapport dédoublonnage leads.csv ===")
    print(f"Lignes total        : {total}")
    print(f"Post_ids uniques (canoniques) : {len(df['post_id_canonical'].unique())}")
    print(f"Groupes en conflit  : {n_conflict_groups}")
    print(f"Lignes concernées   : {n_conflict_rows}")

    if conflicts:
        print("\n--- Détail des conflits ---")
        for c in conflicts:
            print(f"\n  Canonique : {c['canonical']}")
            for r in c["rows"]:
                marker = " ← brut ≠ canonique" if r["post_id_raw"] != c["canonical"] else ""
                print(f"    post_id_raw={r['post_id_raw']!r}  qualified_contacts={r['qualified_contacts']}{marker}")
    else:
        print("\nAucun doublon détecté — toutes les clés sont déjà sous forme canonique.")

    if not args.apply:
        print("\n[Mode aperçu] Aucun fichier modifié. Relancez avec --apply pour appliquer.")
        return

    # --- Snapshot ---
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = SNAPSHOT_DIR / f"leads_pre_dedup_{ts}.csv"
    shutil.copy2(LEADS_CSV, snapshot_path)
    print(f"\nSnapshot créé : {snapshot_path}")

    # --- Écriture atomique ---
    out_df = pd.DataFrame(clean_rows)
    tmp_path = LEADS_CSV.with_suffix(".tmp")
    out_df.to_csv(tmp_path, index=False)
    tmp_path.rename(LEADS_CSV)

    n_after = len(out_df)
    print(f"data/leads.csv réécrit : {total} → {n_after} lignes ({total - n_after} doublons supprimés).")


if __name__ == "__main__":
    main()
