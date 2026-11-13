"""Generate unrelaxed defect supercells and candidates across symmetry-inequivalent sites."""
import argparse
import os
import sys
from typing import Any, Dict, List, Optional
import pandas as pd
from pymatgen.core import Element, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import CONFIG
from src.data_pipeline.calculate_energies import write_csv_atomically
from src.utils.log import get_logger

logger = get_logger("generate_defects")


def generate_doped_structures(
    base_cif_path: str,
    target_element: str,
    dopants: List[str],
    output_prefix: str,
    output_dir: str = "data/unrelaxed",
    status: str = "unrelaxed",
    ood_flag: bool = False,
    relaxed_hosts_dir: str = "data/relaxed/hosts",
) -> List[Dict[str, Any]]:
    """Generate 3x3x3 defect supercells across all symmetry-inequivalent sites.

    Args:
        base_cif_path: Path to the raw host CIF structure.
        target_element: Element symbol to replace with dopants.
        dopants: List of dopant element symbols.
        output_prefix: Name of the host crystal (for example SiC or GaN).
        output_dir: Directory where generated CIFs are written.
        status: Label assigned to generated structures (unrelaxed or candidate).
        ood_flag: Boolean flag indicating out-of-distribution candidate status.
        relaxed_hosts_dir: Directory to check for pre-relaxed host CIFs.

    Returns:
        List of manifest metadata records for generated structures.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Prefer relaxed host structure when available, otherwise fall back to raw
    relaxed_cif = os.path.join(relaxed_hosts_dir, f"{output_prefix}.cif")
    if os.path.exists(relaxed_cif):
        host_cif = relaxed_cif
        logger.info("Using relaxed host structure: %s", host_cif)
    else:
        host_cif = base_cif_path
        logger.warning(
            "Relaxed host structure not found at %s. Using raw host from %s.",
            relaxed_cif,
            base_cif_path,
        )

    base_struct = Structure.from_file(host_cif)

    # 2. Enumerate symmetry-inequivalent sites of target element
    sga = SpacegroupAnalyzer(base_struct)
    symm = sga.get_symmetrized_structure()

    target_sites: List[Dict[str, Any]] = []
    target_site_counter = 0

    for group_idx, indices in enumerate(symm.equivalent_indices):
        first_idx = indices[0]
        site = base_struct[first_idx]
        site_element = site.species.elements[0].symbol
        if site_element == target_element:
            wyckoff = symm.wyckoff_symbols[group_idx]
            site_name = f"{wyckoff}_{target_site_counter}"
            target_sites.append(
                {
                    "base_site_idx": first_idx,
                    "wyckoff": wyckoff,
                    "site": site_name,
                    "site_index": target_site_counter,
                }
            )
            target_site_counter += 1

    logger.info(
        "Host %s: identified %d symmetry-inequivalent site(s) for target %s: %s",
        output_prefix,
        len(target_sites),
        target_element,
        [s["site"] for s in target_sites],
    )

    records: List[Dict[str, Any]] = []

    # 3. Generate supercell and substitute at each inequivalent site
    for site_info in target_sites:
        sub_idx = site_info["base_site_idx"]
        site_name = site_info["site"]
        wyckoff_sym = site_info["wyckoff"]

        for dopant in dopants:
            # Build fresh 3x3x3 supercell
            doped_struct = base_struct.copy()
            doped_struct.make_supercell([3, 3, 3])

            # Substitute single target site
            doped_struct.replace(sub_idx, dopant)

            filename = f"{output_prefix}_doped_{dopant}_{site_name}.cif"
            cif_path = os.path.join(output_dir, filename)
            doped_struct.to(fmt="cif", filename=cif_path)

            dopant_z = int(Element(dopant).Z)
            is_ood = bool(ood_flag or dopant == "U")

            record = {
                "filename": filename,
                "host": output_prefix,
                "site": site_name,
                "wyckoff": wyckoff_sym,
                "dopant": dopant,
                "dopant_z": dopant_z,
                "status": status,
                "ood_flag": is_ood,
                "path": cif_path,
                "paths": cif_path,
            }
            records.append(record)

    logger.info(
        "Generated %d defective structure(s) for host %s into %s.",
        len(records),
        output_prefix,
        output_dir,
    )
    return records


def generate_all_defects(
    unrelaxed_dir: str = "data/unrelaxed",
    candidates_dir: str = "data/candidates",
    relaxed_hosts_dir: str = "data/relaxed/hosts",
    raw_dir: str = "data/raw",
    manifest_path: str = "data/manifest.csv",
) -> pd.DataFrame:
    """Generate all training defects and active learning candidate supercells.

    Args:
        unrelaxed_dir: Destination folder for unrelaxed training structures.
        candidates_dir: Destination folder for candidate structures.
        relaxed_hosts_dir: Source folder for relaxed host structures.
        raw_dir: Source folder for raw host structures.
        manifest_path: Path to output manifest CSV.

    Returns:
        DataFrame containing all manifest records.
    """
    os.makedirs(unrelaxed_dir, exist_ok=True)
    os.makedirs(candidates_dir, exist_ok=True)

    all_records: List[Dict[str, Any]] = []

    # 1. Generate training defect supercells for hosts in config
    for host_name, host_def in CONFIG.hosts.items():
        raw_cif = os.path.join(raw_dir, f"{host_name}_base.cif")
        if not os.path.exists(raw_cif):
            raw_cif = host_def.cif_path

        training_records = generate_doped_structures(
            base_cif_path=raw_cif,
            target_element=host_def.cation_symbol,
            dopants=CONFIG.training_dopants,
            output_prefix=host_name,
            output_dir=unrelaxed_dir,
            status="unrelaxed",
            ood_flag=False,
            relaxed_hosts_dir=relaxed_hosts_dir,
        )
        all_records.extend(training_records)

    # 2. Generate candidate structures for entries in candidate_dopants (SiC sites)
    sic_def = CONFIG.hosts.get("SiC")
    if sic_def:
        raw_sic_cif = os.path.join(raw_dir, "SiC_base.cif")
        if not os.path.exists(raw_sic_cif):
            raw_sic_cif = sic_def.cif_path

        candidate_records = generate_doped_structures(
            base_cif_path=raw_sic_cif,
            target_element=sic_def.cation_symbol,
            dopants=CONFIG.candidate_dopants,
            output_prefix="SiC",
            output_dir=candidates_dir,
            status="candidate",
            ood_flag=True,
            relaxed_hosts_dir=relaxed_hosts_dir,
        )
        all_records.extend(candidate_records)

    # 3. Write data/manifest.csv atomically
    df_manifest = pd.DataFrame(all_records)
    write_csv_atomically(df_manifest, manifest_path)
    logger.info("Saved %d total entries to manifest at %s", len(df_manifest), manifest_path)
    return df_manifest


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entrypoint for defect structure generation."""
    parser = argparse.ArgumentParser(
        description="Generate unrelaxed defect supercells and candidate structures across inequivalent sites."
    )
    parser.add_argument(
        "--unrelaxed-dir",
        type=str,
        default="data/unrelaxed",
        help="Destination directory for unrelaxed training structures.",
    )
    parser.add_argument(
        "--candidates-dir",
        type=str,
        default="data/candidates",
        help="Destination directory for candidate structures.",
    )
    parser.add_argument(
        "--relaxed-hosts-dir",
        type=str,
        default="data/relaxed/hosts",
        help="Source directory for relaxed host structures.",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=CONFIG.raw_dir,
        help="Source directory for raw base structures.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/manifest.csv",
        help="Destination path for manifest CSV.",
    )
    args = parser.parse_args(argv)

    generate_all_defects(
        unrelaxed_dir=args.unrelaxed_dir,
        candidates_dir=args.candidates_dir,
        relaxed_hosts_dir=args.relaxed_hosts_dir,
        raw_dir=args.raw_dir,
        manifest_path=args.manifest,
    )


if __name__ == "__main__":
    main()
