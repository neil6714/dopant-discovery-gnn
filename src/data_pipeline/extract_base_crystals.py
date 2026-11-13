"""Download base host crystals and elemental reference structures from Materials Project."""
import argparse
import os
import sys
from typing import Any, Dict, List, Optional
import pandas as pd
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import CONFIG


def fetch_base_structure(
    material_id: str,
    name: str,
    output_dir: Optional[str] = None,
    force: bool = False,
    api_key: Optional[str] = None,
) -> str:
    """Download a base crystal structure from Materials Project if not already present.

    Args:
        material_id: Materials Project identifier (for example mp-11714).
        name: Base file identifier (for example SiC_base).
        output_dir: Target directory for storing raw CIF files.
        force: Whether to overwrite existing files on disk.
        api_key: Materials Project API key.

    Returns:
        Path to the saved CIF file.
    """
    if output_dir is None:
        output_dir = CONFIG.raw_dir
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{name}.cif")

    if not force and os.path.exists(file_path):
        print(f"File {file_path} already exists. Skipping download (use --force to overwrite).")
        return file_path

    if not api_key:
        api_key = os.getenv("MP_API_KEY")
    if not api_key:
        sys.exit("Error: MP_API_KEY environment variable is required to download from Materials Project.")

    from mp_api.client import MPRester

    print(f"Fetching {name} ({material_id})...")
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(material_ids=[material_id])
        if not docs:
            sys.exit(f"Error: No Materials Project record found for {material_id}.")
        structure = docs[0].structure
        structure.to(fmt="cif", filename=file_path)
        print(f"Saved structure to {file_path}")

    return file_path


def fetch_elemental_reference(
    element: str,
    output_dir: Optional[str] = None,
    force: bool = False,
    mpr: Optional[Any] = None,
) -> Dict[str, Any]:
    """Fetch the lowest energy_above_hull structure for a pure element.

    Args:
        element: Element chemical symbol.
        output_dir: Directory where reference CIFs are stored.
        force: Whether to overwrite existing files on disk.
        mpr: Active MPRester client instance.

    Returns:
        Dictionary containing element symbol, material_id, energy_above_hull, and file path.
    """
    if output_dir is None:
        output_dir = os.path.join(CONFIG.raw_dir, "references")
    os.makedirs(output_dir, exist_ok=True)
    cif_path = os.path.join(output_dir, f"{element}.cif")

    if not force and os.path.exists(cif_path):
        print(f"Reference for {element} already exists at {cif_path}. Skipping.")
        return {"element": element, "cif_path": cif_path, "skipped": True}

    if mpr is None:
        raise ValueError("An active MPRester client is required to fetch elemental references.")

    print(f"Fetching elemental reference for {element}...")
    docs = mpr.materials.summary.search(
        chemsys=element,
        num_elements=1,
        fields=["material_id", "structure", "energy_above_hull"],
    )
    if not docs:
        docs = mpr.materials.summary.search(
            elements=[element],
            num_elements=1,
            fields=["material_id", "structure", "energy_above_hull"],
        )

    if not docs:
        raise RuntimeError(f"No reference structure found on Materials Project for element {element}.")

    # Select the lowest energy above hull entry
    best_doc = min(docs, key=lambda d: getattr(d, "energy_above_hull", float("inf")) or 0.0)
    structure = best_doc.structure
    structure.to(fmt="cif", filename=cif_path)

    material_id_str = str(best_doc.material_id)
    e_hull = float(best_doc.energy_above_hull) if best_doc.energy_above_hull is not None else 0.0

    print(f"Saved {element} reference ({material_id_str}, e_hull={e_hull:.4f} eV) to {cif_path}")
    return {
        "element": element,
        "material_id": material_id_str,
        "energy_above_hull": e_hull,
        "cif_path": cif_path,
        "skipped": False,
    }


def main(argv: Optional[List[str]] = None) -> None:
    """Download base crystals and elemental references."""
    load_dotenv()
    api_key = os.getenv("MP_API_KEY")

    parser = argparse.ArgumentParser(
        description="Download base crystals and elemental reference structures from Materials Project."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files already exist on disk.",
    )
    args = parser.parse_args(argv)

    if not api_key:
        sys.exit("Error: MP_API_KEY environment variable is not set. Please set it in your environment or .env file.")

    from mp_api.client import MPRester

    raw_dir = CONFIG.raw_dir
    references_dir = os.path.join(raw_dir, "references")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(references_dir, exist_ok=True)

    # 1. Fetch base host structures
    for host_key, host_def in CONFIG.hosts.items():
        base_name = f"{host_key}_base"
        fetch_base_structure(
            material_id=host_def.material_id,
            name=base_name,
            output_dir=raw_dir,
            force=args.force,
            api_key=api_key,
        )

    # Note: The nitrogen chemical potential is not used in the cation-rich limit.
    # 2. Collect elements for references: all dopants plus host cations and anions
    elements = set()
    for dopant in CONFIG.training_dopants + CONFIG.candidate_dopants:
        elements.add(dopant)
    for host_def in CONFIG.hosts.values():
        elements.add(host_def.cation_symbol)
        elements.add(host_def.anion_symbol)

    sorted_elements = sorted(list(elements))
    sources_csv = os.path.join(references_dir, "sources.csv")
    existing_sources: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(sources_csv):
        try:
            df_existing = pd.read_csv(sources_csv)
            for _, row in df_existing.iterrows():
                el = str(row["element"]).strip()
                existing_sources[el] = {
                    "element": el,
                    "material_id": str(row["material_id"]),
                    "energy_above_hull": float(row["energy_above_hull"]),
                }
        except Exception:
            pass

    records = []
    with MPRester(api_key) as mpr:
        for el in sorted_elements:
            cif_path = os.path.join(references_dir, f"{el}.cif")
            if not args.force and os.path.exists(cif_path) and el in existing_sources:
                records.append(existing_sources[el])
                print(f"Reference for {el} already exists at {cif_path}. Skipping.")
                continue

            info = fetch_elemental_reference(
                element=el,
                output_dir=references_dir,
                force=args.force,
                mpr=mpr,
            )
            records.append(
                {
                    "element": info["element"],
                    "material_id": info.get("material_id", existing_sources.get(el, {}).get("material_id", "")),
                    "energy_above_hull": info.get(
                        "energy_above_hull",
                        existing_sources.get(el, {}).get("energy_above_hull", 0.0),
                    ),
                }
            )

    if records:
        df_sources = pd.DataFrame(records).drop_duplicates(subset=["element"], keep="last")
        df_sources.to_csv(sources_csv, index=False)
        print(f"Recorded elemental reference sources to {sources_csv}")


if __name__ == "__main__":
    main()
