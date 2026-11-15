"""One-off cleanup of data left over from the first full run. Nothing is deleted.

Two problems are fixed here, and everything removed from the active files is copied to
archive/legacy_v1/ first:

1. data/dopant_energies.csv still holds the 20 rows from the very first pipeline version.
   Their total_energy column is a per-atom number, so the formation energy computed from it
   is meaningless (values up to ~1600 eV). They were never used for training, but they
   distorted the summary statistics.
2. generate_defects.py used a unit-cell atom index on the 3x3x3 supercell, so the "2a_1" SiC
   structures are the same site as "2b_0" (identical formation energies to ~1e-6 eV). Those
   rows are archived and have to be regenerated with the fixed generator and relaxed again.

Dry run by default. Pass --apply to change files.
"""
import argparse
import os
import shutil
import sys
from typing import List, Optional

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import CONFIG
from src.utils.log import get_logger

logger = get_logger("clean_data")

ARCHIVE = "archive/legacy_v1"
LEGACY_COLUMNS = ["total_energy", "energy_per_atom"]
# Filenames written by the buggy generator whose second site was not actually a second site.
DUPLICATE_SITE_SUFFIX = "_2a_1.cif"

NOTES = """# legacy_v1: data set aside after the site-index fix

Created by `python -m src.data_pipeline.clean_data --apply`. Nothing here was deleted from
the project, only moved out of the active files.

- `dopant_energies_before_cleanup.csv`: the full CSV exactly as it was.
- `legacy_rows.csv`: 20 rows from the first pipeline version. `total_energy` in those rows is
  a per-atom value, so their `e_form_ev` is not meaningful.
- `duplicate_site_rows.csv`: rows for `*_2a_1` structures. The old generator replaced a
  translated copy of the 2b site instead of the 2a site, so these repeat the 2b energies.
  The structures have to be regenerated and relaxed again.
- `processed_graphs_before_site_fix.pt`: graph file that still contains the duplicate sites.
- `duplicate_site_cifs/`: unrelaxed or relaxed CIFs for the duplicate sites, if present.
"""


def _copy_once(src: str, name: str) -> None:
    """Copy src into the archive unless an archived copy exists (keeps the original safe on re-runs)."""
    dest = os.path.join(ARCHIVE, name)
    os.makedirs(ARCHIVE, exist_ok=True)
    if os.path.exists(dest):
        logger.info("Archive copy %s already exists, leaving it untouched", dest)
    else:
        shutil.copyfile(src, dest)


def _archive_csv(df: pd.DataFrame, name: str, apply: bool) -> None:
    path = os.path.join(ARCHIVE, name)
    logger.info("Archive %d row(s) -> %s", len(df), path)
    if apply and len(df) and not os.path.exists(path):
        os.makedirs(ARCHIVE, exist_ok=True)
        df.to_csv(path, index=False)


def clean_energies(csv_path: str, apply: bool) -> Optional[pd.DataFrame]:
    """Split the energies CSV into active rows and archived rows."""
    if not os.path.exists(csv_path):
        logger.warning("No energies CSV at %s", csv_path)
        return None
    df = pd.read_csv(csv_path)

    has_new_schema = df["e_total_ev"].notna() if "e_total_ev" in df.columns else pd.Series(False, index=df.index)
    legacy = df[~has_new_schema]
    current = df[has_new_schema]

    dup_mask = current["filename"].astype(str).str.endswith(DUPLICATE_SITE_SUFFIX)
    duplicates = current[dup_mask]
    active = current[~dup_mask].drop(columns=[c for c in LEGACY_COLUMNS if c in current.columns])

    logger.info(
        "Energies: %d total, %d legacy, %d duplicate-site, %d kept",
        len(df), len(legacy), len(duplicates), len(active),
    )
    if apply:
        _copy_once(csv_path, "dopant_energies_before_cleanup.csv")
    _archive_csv(legacy, "legacy_rows.csv", apply)
    _archive_csv(duplicates, "duplicate_site_rows.csv", apply)
    if apply:
        active.to_csv(csv_path, index=False)
    return active


def move_duplicate_cifs(directories: List[str], apply: bool) -> None:
    """Move duplicate-site CIFs out of the working folders."""
    dest = os.path.join(ARCHIVE, "duplicate_site_cifs")
    for directory in directories:
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory)):
            if name.endswith(DUPLICATE_SITE_SUFFIX):
                logger.info("Move %s/%s -> %s", directory, name, dest)
                if apply:
                    os.makedirs(os.path.join(dest, os.path.basename(directory)), exist_ok=True)
                    shutil.move(
                        os.path.join(directory, name),
                        os.path.join(dest, os.path.basename(directory), name),
                    )


def flag_manifest(manifest_path: str, apply: bool) -> None:
    """Mark duplicate-site manifest rows so the next relaxation run picks them up again."""
    if not os.path.exists(manifest_path):
        return
    df = pd.read_csv(manifest_path)
    mask = df["filename"].astype(str).str.endswith(DUPLICATE_SITE_SUFFIX) & (df["status"] != "candidate")
    logger.info("Manifest: %d row(s) flagged needs_regeneration", int(mask.sum()))
    if apply and mask.any():
        df.loc[mask, "status"] = "needs_regeneration"
        df.to_csv(manifest_path, index=False)


def clean_graphs(graph_path: str, apply: bool) -> None:
    """Drop duplicate-site graphs from the stored graph file and attach rich descriptors."""
    if not os.path.exists(graph_path):
        return
    import torch  # imported here so the CSV cleanup works without torch

    from src.data_pipeline.structures_to_graphs import attach_descriptors

    graphs = torch.load(graph_path, weights_only=False)
    if graphs and not hasattr(graphs[0], "dopant_z"):
        logger.warning(
            "%s is in the first-version graph format (no dopant_z). Leaving it alone, "
            "rebuild it with structures_to_graphs.", graph_path,
        )
        return
    kept = [g for g in graphs if not str(getattr(g, "filename", "")).endswith(DUPLICATE_SITE_SUFFIX)]
    logger.info("Graphs: %d -> %d after dropping duplicate sites", len(graphs), len(kept))
    if apply:
        _copy_once(graph_path, "processed_graphs_before_site_fix.pt")
        attach_descriptors(kept, CONFIG.references_csv_path)
        torch.save(kept, graph_path)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Archive stale rows and duplicate-site data.")
    parser.add_argument("--apply", action="store_true", help="Write changes (default is a dry run).")
    args = parser.parse_args(argv)

    clean_energies(CONFIG.energies_csv_path, args.apply)
    move_duplicate_cifs([CONFIG.unrelaxed_dir, CONFIG.relaxed_dir, CONFIG.candidates_dir], args.apply)
    flag_manifest(CONFIG.manifest_path, args.apply)
    clean_graphs(CONFIG.graph_data_path, args.apply)

    if args.apply:
        with open(os.path.join(ARCHIVE, "NOTES.md"), "w", encoding="utf-8") as fh:
            fh.write(NOTES)
        print(f"Done. Archived material is in {ARCHIVE}/")
    else:
        print("Dry run only. Re-run with --apply to make these changes.")


if __name__ == "__main__":
    main()
