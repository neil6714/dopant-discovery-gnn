"""Convert processed CIF files and energy labels into graph data."""
import os
import pandas as pd
import torch
from pymatgen.core import Structure
from torch_geometric.data import Data

# PyTorch 2.6+ compatibility for saving/loading PyG Data objects safely
try:
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([Data])
except Exception:
    pass


def build_graph_from_cif(cif_path, y_val=None):
    """Convert a CIF file into a PyG Data object with normalized node features and edge attributes.

    Args:
        cif_path (str): Path to the CIF file.
        y_val (float, optional): Ground truth energy or target label.

    Returns:
        Data: PyG Data object containing x, edge_index, edge_attr, y, and material_id.
    """
    struct = Structure.from_file(cif_path)
    filename = os.path.basename(cif_path)

    # 1. Node Features: Normalized atomic number Z / 100.0 (shape: [num_nodes, 1])
    node_features = torch.tensor(
        [[float(site.specie.Z) / 100.0] for site in struct],
        dtype=torch.float,
    )

    # 2. Edge Connectivity & Distances: Find neighbors within 5.0 Angstroms
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

    # 3. Target Label (y)
    if y_val is not None:
        y = torch.tensor([float(y_val)], dtype=torch.float)
    else:
        y = None

    # 4. Construct Data object and attach metadata
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.material_id = filename

    return data


def cif_to_graph(cif_path, energy_label=None):
    """Backward-compatible wrapper for build_graph_from_cif."""
    return build_graph_from_cif(cif_path, y_val=energy_label)


def process_cif_directory(
    processed_dir="data/processed",
    csv_path="data/dopant_energies.csv",
    output_path="data/graph_data/processed_graphs.pt",
):
    """Convert valid processed CIF files into PyG graphs, skipping unvalidated hypotheses."""
    if not os.path.exists(processed_dir):
        print(f"Error: {processed_dir} not found!")
        return []

    energy_map = {}
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if "filename" in df.columns and "energy_per_atom" in df.columns:
            df = df.drop_duplicates(subset=["filename"], keep="last")
            for _, row in df.iterrows():
                fname = str(row["filename"]).strip()
                try:
                    energy_map[fname] = float(row["energy_per_atom"])
                except (ValueError, TypeError):
                    continue

    files = sorted([f for f in os.listdir(processed_dir) if f.endswith(".cif")])
    graphs = []

    print(f"Converting CIF files in {processed_dir} to Graphs...")
    for filename in files:
        cif_path = os.path.join(processed_dir, filename)

        # Look up ground-truth energy label
        y_val = None
        if filename in energy_map:
            y_val = energy_map[filename]
        elif filename.endswith("_HYPOTHESIS.cif"):
            alt_name = filename.replace("_HYPOTHESIS.cif", ".cif")
            if alt_name in energy_map:
                y_val = energy_map[alt_name]

        # Exclude unvalidated hypotheses from training set
        if filename.endswith("_HYPOTHESIS.cif") and y_val is None:
            print(f"Skipping unvalidated hypothesis: {filename}")
            continue

        if y_val is None:
            print(f"Skipping {filename}: no ground-truth energy found in {csv_path}")
            continue

        try:
            graph = build_graph_from_cif(cif_path, y_val=y_val)
            graphs.append(graph)
            print(f"Processed {filename} -> y={y_val:.4f}")
        except Exception as e:
            print(f"Failed to process {filename}: {repr(e)}")

    if graphs:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        torch.save(graphs, output_path)
        print(f"Saved {len(graphs)} graphs to {output_path}")
    else:
        print("Warning: No graphs were processed.")

    return graphs


if __name__ == "__main__":
    process_cif_directory()