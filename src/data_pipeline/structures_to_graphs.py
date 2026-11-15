"""Convert unrelaxed crystal structures into local cluster graph representations."""
import argparse
import os
import re
import sys
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from pymatgen.core import Element, Structure
import torch
from torch_geometric.data import Data

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import CONFIG
from src.utils.descriptors import rich_descriptor

# PyTorch 2.6+ compatibility for saving/loading PyG Data objects safely
try:
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([Data])
except Exception:
    pass

# Unrelaxed input structures are used so the model functions as a fast screening surrogate.


def get_valence_electron_count(element: Element) -> int:
    """Extract outer valence electron count from electronic structure string."""
    es = getattr(element, "electronic_structure", "")
    parts = es.split(".")
    valence_parts = [p for p in parts if not p.startswith("[")]
    val_count = 0
    for p in valence_parts:
        m = re.search(r"(\d+)$", p)
        if m:
            val_count += int(m.group(1))
    return val_count if val_count > 0 else int(element.group or 1)


def gaussian_rbf(
    distances: torch.Tensor,
    num_rbf: int = 8,
    rbf_min: float = 0.5,
    rbf_max: float = 6.0,
) -> torch.Tensor:
    """Expand scalar distances using Gaussian radial basis functions."""
    centers = torch.linspace(rbf_min, rbf_max, num_rbf, dtype=torch.float)
    sigma = (rbf_max - rbf_min) / float(max(1, num_rbf - 1))
    gamma = 1.0 / (2.0 * (sigma**2))
    diff = distances.unsqueeze(-1) - centers.unsqueeze(0)
    return torch.exp(-gamma * (diff**2))


def build_graph_legacy(cif_path: str, y_val: Optional[float] = None) -> Data:
    """Construct full-supercell graph using normalized atomic numbers and 5.0 A inverse distances."""
    struct = Structure.from_file(cif_path)
    filename = os.path.basename(cif_path)

    # 1. Node Features: Normalized atomic number Z / 100.0 (shape: [num_nodes, 1])
    node_features = torch.tensor(
        [[float(site.specie.Z) / 100.0] for site in struct],
        dtype=torch.float,
    )

    # 2. Edge Connectivity & Distances: Radial cutoff r <= 5.0 Angstroms
    edge_index_list = []
    edge_attr_list = []
    all_neighbors = struct.get_all_neighbors(r=5.0)

    for i, neighbors in enumerate(all_neighbors):
        for nb in neighbors:
            edge_index_list.append([i, nb.index])
            inv_dist = 1.0 / (float(nb.nn_distance) + 1e-5)
            edge_attr_list.append([inv_dist])

    if edge_index_list:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)

    # 3. Target label
    y = torch.tensor([float(y_val)], dtype=torch.float) if y_val is not None else None

    # 4. Construct Data object
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.material_id = filename
    return data


def build_graph_from_cif(
    cif_path: str,
    y_val: Optional[float] = None,
    manifest_meta: Optional[Dict[str, Any]] = None,
    cluster_radius: float = 6.0,
    edge_cutoff: float = 4.5,
    num_rbf: int = 8,
    rbf_min: float = 0.5,
    rbf_max: float = 6.0,
    references_csv: str = "data/references.csv",
) -> Data:
    """Construct defect-centered cluster graph representation with RBF edge features.

    Args:
        cif_path: Path to CIF file.
        y_val: Target energy label (e_form_ev).
        manifest_meta: Optional metadata dictionary from manifest.csv.
        cluster_radius: Radius in Angstroms around dopant to retain atoms.
        edge_cutoff: Cutoff distance in Angstroms for graph edges.
        num_rbf: Number of Gaussian RBF centers.
        rbf_min: Minimum distance center in Angstroms.
        rbf_max: Maximum distance center in Angstroms.
        references_csv: Elemental reference energies, used for the mu_dopant descriptor.

    Returns:
        PyG Data object with node features, edge indices, RBF edge attributes, a graph-level
        descriptor vector (desc), and metadata.
    """
    struct = Structure.from_file(cif_path)
    filename = os.path.basename(cif_path)

    # Determine host name from metadata or filename
    host = "SiC" if "SiC" in filename else ("GaN" if "GaN" in filename else "unknown")
    if manifest_meta and "host" in manifest_meta:
        host = str(manifest_meta["host"])

    # 1. Identify dopant site index
    dopant_symbol = manifest_meta.get("dopant") if manifest_meta else None
    dopant_indices: List[int] = []

    if dopant_symbol:
        dopant_indices = [
            i for i, s in enumerate(struct) if s.species.elements[0].symbol == dopant_symbol
        ]

    if not dopant_indices:
        host_species = ["Si", "C", "Ga", "N"]
        dopant_indices = [
            i for i, s in enumerate(struct) if s.species.elements[0].symbol not in host_species
        ]

    dopant_idx = dopant_indices[0] if dopant_indices else 0
    dopant_site = struct[dopant_idx]
    dopant_z = int(dopant_site.species.elements[0].Z)

    # 2. Extract cluster: dopant center plus neighbors within cluster_radius (including periodic images)
    neighbors = struct.get_neighbors(dopant_site, r=cluster_radius)
    cluster_sites = [dopant_site] + list(neighbors)

    # 3. Construct node features
    # Features: [element_idx, electronegativity, covalent_radius, group, period, valence, is_dopant, host_SiC, host_GaN]
    node_feat_list = []
    z_list = []

    for idx, site in enumerate(cluster_sites):
        el = site.species.elements[0]
        z_val = int(el.Z)
        z_list.append(z_val)

        elem_idx = float(z_val)
        electronegativity = float(el.X or 1.5) / 4.0
        covalent_radius = float(getattr(el, "atomic_radius", None) or 1.5) / 2.5
        group_val = float(el.group or 3) / 18.0
        period_val = float(el.row) / 7.0
        valence_electrons = float(get_valence_electron_count(el)) / 18.0
        is_dopant_flag = 1.0 if idx == 0 else 0.0

        host_onehot = [1.0 if host == "SiC" else 0.0, 1.0 if host == "GaN" else 0.0]

        node_feat_list.append(
            [
                elem_idx,
                electronegativity,
                covalent_radius,
                group_val,
                period_val,
                valence_electrons,
                is_dopant_flag,
            ]
            + host_onehot
        )

    x = torch.tensor(node_feat_list, dtype=torch.float)
    z = torch.tensor(z_list, dtype=torch.long)

    # 4. Construct edges within cluster (edge_cutoff = 4.5 A)
    coords = np.array([s.coords for s in cluster_sites])
    diff = coords[:, None, :] - coords[None, :, :]
    dist_matrix = np.linalg.norm(diff, axis=-1)

    edge_mask = (dist_matrix > 1e-4) & (dist_matrix <= edge_cutoff)
    edge_pairs = np.argwhere(edge_mask)

    if len(edge_pairs) > 0:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        edge_distances = torch.tensor(dist_matrix[edge_mask], dtype=torch.float)
        edge_attr = gaussian_rbf(
            edge_distances,
            num_rbf=num_rbf,
            rbf_min=rbf_min,
            rbf_max=rbf_max,
        )
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_rbf), dtype=torch.float)

    # 5. Target label y
    y = torch.tensor([float(y_val)], dtype=torch.float) if y_val is not None else None

    # 6. Metadata attributes
    site_str = str(manifest_meta.get("site", "unknown")) if manifest_meta else "unknown"
    ood_flag = (
        bool(manifest_meta.get("ood_flag", False))
        if manifest_meta
        else (dopant_symbol == "U" or dopant_z == 92)
    )
    converged_all = bool(manifest_meta.get("converged_all", True)) if manifest_meta else True

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.z = z
    data.filename = filename
    data.material_id = filename
    data.host = host
    data.dopant_z = dopant_z
    data.site = site_str
    data.ood_flag = ood_flag
    data.converged_all = converged_all

    # Graph-level dopant descriptors (same vector the ridge and GP baselines use)
    dopant_name = dopant_symbol or str(site_symbol(dopant_site))
    data.desc = torch.tensor(
        rich_descriptor(dopant_name, host, site_str, references_csv=references_csv),
        dtype=torch.float,
    ).unsqueeze(0)

    return data


def attach_descriptors(
    graphs: List[Data],
    references_csv: str = "data/references.csv",
    overwrite: bool = False,
) -> List[Data]:
    """Add the graph-level descriptor vector to graphs that were built before it existed.

    The descriptor only depends on dopant, host, site and the elemental reference energy,
    so stored graphs can be upgraded without the CIF files.
    """
    from pymatgen.core import Element

    for g in graphs:
        if getattr(g, "desc", None) is not None and not overwrite:
            continue
        dopant = Element.from_Z(int(g.dopant_z)).symbol
        vec = rich_descriptor(dopant, str(g.host), str(getattr(g, "site", "")), references_csv=references_csv)
        g.desc = torch.tensor(vec, dtype=torch.float).unsqueeze(0)
    return graphs


def site_symbol(site) -> str:
    """Element symbol of a pymatgen site."""
    return site.species.elements[0].symbol


def cif_to_graph(cif_path: str, energy_label: Optional[float] = None) -> Data:
    """Backward-compatible wrapper for build_graph_from_cif."""
    return build_graph_from_cif(cif_path, y_val=energy_label)


def process_cif_directory(
    unrelaxed_dir: str = "data/unrelaxed",
    csv_path: str = "data/dopant_energies.csv",
    manifest_path: str = "data/manifest.csv",
    output_path: str = "data/graph_data/processed_graphs.pt",
    processed_dir: Optional[str] = None,
) -> List[Data]:
    """Convert valid unrelaxed CIF files into PyG graphs, filtering out non-converged or OOD candidates.

    Args:
        unrelaxed_dir: Directory containing input unrelaxed CIF structures.
        csv_path: Path to dopant energies CSV containing e_form_ev.
        manifest_path: Path to defect manifest CSV.
        output_path: Destination path for graph dataset tensor file.
        processed_dir: Backward-compatible alias for unrelaxed_dir.

    Returns:
        List of constructed PyG Data objects.
    """
    input_dir = processed_dir if processed_dir is not None else unrelaxed_dir
    if not os.path.exists(input_dir):
        print(f"Error: Directory {input_dir} not found!")
        return []

    # 1. Load manifest metadata
    manifest_map: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(manifest_path):
        try:
            df_manifest = pd.read_csv(manifest_path)
            for _, row in df_manifest.iterrows():
                fname = str(row.get("filename", "")).strip()
                if fname:
                    manifest_map[fname] = row.to_dict()
        except Exception as exc:
            print(f"Warning: Could not read manifest at {manifest_path}: {exc}")

    # 2. Load energy labels and convergence statuses
    energy_map: Dict[str, float] = {}
    converged_map: Dict[str, bool] = {}
    ood_map: Dict[str, bool] = {}

    if os.path.exists(csv_path):
        try:
            df_energies = pd.read_csv(csv_path)
            if "filename" in df_energies.columns:
                df_energies = df_energies.drop_duplicates(subset=["filename"], keep="last")
                for _, row in df_energies.iterrows():
                    fname = str(row["filename"]).strip()

                    # Prefer e_form_ev, fallback to energy_per_atom if e_form_ev is not yet computed
                    e_target = row.get("e_form_ev")
                    if pd.isna(e_target):
                        e_target = row.get("energy_per_atom")

                    if not pd.isna(e_target):
                        try:
                            energy_map[fname] = float(e_target)
                        except (ValueError, TypeError):
                            pass

                    # Convergence check
                    if "converged_all" in row:
                        converged_map[fname] = bool(row["converged_all"])
                    elif "converged" in row:
                        converged_map[fname] = bool(row["converged"])
                    else:
                        converged_map[fname] = True

                    # OOD flag
                    if "ood_flag" in row:
                        ood_map[fname] = bool(row["ood_flag"])
        except Exception as exc:
            print(f"Warning: Could not read energy CSV at {csv_path}: {exc}")

    # 3. Filter and construct graphs
    files = sorted([f for f in os.listdir(input_dir) if f.endswith(".cif")])

    skip_missing_label = 0
    skip_non_converged = 0
    skip_ood = 0
    graphs: List[Data] = []

    print(f"Converting CIF structures in {input_dir} to graphs...")

    for fname in files:
        cif_path = os.path.join(input_dir, fname)
        meta = manifest_map.get(fname, {})

        # OOD check
        is_ood = ood_map.get(fname, bool(meta.get("ood_flag", False)))
        if is_ood or re.search(r"_doped_U_", fname):
            skip_ood += 1
            continue

        # Missing label check
        if fname not in energy_map:
            skip_missing_label += 1
            continue
        y_val = energy_map[fname]

        # Convergence check
        is_converged = converged_map.get(fname, bool(meta.get("converged", True)))
        if not is_converged:
            skip_non_converged += 1
            continue

        meta["converged_all"] = is_converged
        meta["ood_flag"] = is_ood

        try:
            graph = build_graph_from_cif(cif_path=cif_path, y_val=y_val, manifest_meta=meta)
            graphs.append(graph)
        except Exception as exc:
            print(f"Failed to process {fname}: {repr(exc)}")

    # Report filtering counts
    print(f"Skipped due to missing label: {skip_missing_label}")
    print(f"Skipped due to non-converged: {skip_non_converged}")
    print(f"Skipped due to ood_flagged:   {skip_ood}")
    print(f"Successfully processed {len(graphs)} graph(s).")

    if graphs:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        torch.save(graphs, output_path)
        print(f"Saved {len(graphs)} graph(s) to {output_path}")

    return graphs


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entrypoint for batch graph dataset generation."""
    parser = argparse.ArgumentParser(
        description="Convert unrelaxed CIF files and formation energies into graph tensors."
    )
    parser.add_argument(
        "--unrelaxed-dir",
        type=str,
        default="data/unrelaxed",
        help="Source directory for unrelaxed CIF structures.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=CONFIG.energies_csv_path,
        help="Path to dopant energies CSV.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=CONFIG.manifest_path,
        help="Path to defect manifest CSV.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=CONFIG.graph_data_path,
        help="Output path for processed graph dataset file.",
    )
    parser.add_argument(
        "--refresh-descriptors",
        action="store_true",
        help="Only add or update the descriptor vector on the existing graph file (no CIFs needed).",
    )
    args = parser.parse_args(argv)

    if args.refresh_descriptors:
        graphs = torch.load(args.output, weights_only=False)
        attach_descriptors(graphs, CONFIG.references_csv_path, overwrite=True)
        torch.save(graphs, args.output)
        print(f"Updated descriptors on {len(graphs)} graph(s) in {args.output}")
        return

    process_cif_directory(
        unrelaxed_dir=args.unrelaxed_dir,
        csv_path=args.csv,
        manifest_path=args.manifest,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
