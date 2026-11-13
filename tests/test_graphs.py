"""Unit tests for crystal graph construction and node/edge attributes."""
import torch
from pymatgen.core import Lattice, Structure

from src.data_pipeline.structures_to_graphs import build_graph_from_cif


def test_graph_construction_and_attributes(tmp_path):
    """Verify node count, 8-channel edge RBF features, attached target y, and unique dopant node."""
    lattice = Lattice.cubic(15.0)
    # 4-atom cell with 1 dopant (Ti) and 3 host atoms within 2.0 A of the dopant
    struct = Structure(
        lattice,
        ["Ti", "Si", "C", "C"],
        [
            [0.0, 0.0, 0.0],
            [1.5 / 15.0, 0.0, 0.0],
            [0.0, 1.5 / 15.0, 0.0],
            [0.0, 0.0, 1.5 / 15.0],
        ],
    )
    cif_file = tmp_path / "toy_doped.cif"
    struct.to(fmt="cif", filename=str(cif_file))

    y_expected = 3.1415
    graph = build_graph_from_cif(
        str(cif_file),
        y_val=y_expected,
        cluster_radius=6.0,
        edge_cutoff=4.5,
    )

    # 1. Node count matches cluster atoms
    assert graph.num_nodes == 4

    # 2. Edge attributes shape is [E, 8]
    num_edges = graph.edge_index.size(1)
    assert graph.edge_attr.shape == (num_edges, 8)

    # 3. Target label is attached
    assert graph.y is not None
    assert torch.isclose(graph.y, torch.tensor([y_expected], dtype=torch.float))

    # 4. is_dopant indicator (column index 6 in node features) is set exactly once
    is_dopant_flags = graph.x[:, 6]
    assert (is_dopant_flags == 1.0).sum().item() == 1
