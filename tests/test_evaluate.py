"""Unit tests for Leave-One-Dopant-Out and Leave-One-Host-Out evaluation routines."""
import os
import shutil
import sys
import tempfile
import numpy as np
import torch
from torch_geometric.data import Data

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.evaluate import (
    compute_plain_verdict,
    evaluate_mean_predictor,
    evaluate_ridge,
    extract_dopant_descriptors,
    parse_sample_metadata,
    plot_parity_lodo,
    prepare_dataset_items,
    run_lodo_cv,
    run_loho_cv,
    train_legacy_gcn,
)


def create_dummy_graph(
    host: str = "SiC",
    dopant_symbol: str = "Ti",
    dopant_z: int = 22,
    site: str = "2b_0",
    y_val: float = 1.5,
) -> Data:
    """Create a minimal synthetic PyG Data object for testing."""
    filename = f"{host}_doped_{dopant_symbol}_{site}.cif"
    # Node features: 2 nodes, 9 features (elem_idx, 5 descriptors, is_dopant, 2 host one-hot)
    x = torch.tensor(
        [
            [float(dopant_z), 1.54, 1.45, 4.0, 4.0, 4.0, 1.0, 1.0 if host == "SiC" else 0.0, 1.0 if host == "GaN" else 0.0],
            [14.0, 1.90, 1.11, 14.0, 3.0, 4.0, 0.0, 1.0 if host == "SiC" else 0.0, 1.0 if host == "GaN" else 0.0],
        ],
        dtype=torch.float,
    )
    z = torch.tensor([dopant_z, 14], dtype=torch.long)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr = torch.ones((2, 8), dtype=torch.float)
    y = torch.tensor([y_val], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.z = z
    data.filename = filename
    data.material_id = filename
    data.host = host
    data.dopant_z = dopant_z
    data.site = site
    data.ood_flag = False
    data.converged_all = True
    return data


def test_extract_dopant_descriptors():
    """Verify descriptor dimension and expected elemental values."""
    desc = extract_dopant_descriptors("Ti", "SiC")
    assert desc.shape == (8,)
    assert desc[0] == 22.0  # Z of Ti
    assert desc[6] == 1.0  # SiC flag
    assert desc[7] == 0.0  # GaN flag

    desc_gan = extract_dopant_descriptors("V", "GaN")
    assert desc_gan.shape == (8,)
    assert desc_gan[0] == 23.0  # Z of V
    assert desc_gan[6] == 0.0
    assert desc_gan[7] == 1.0


def test_parse_sample_metadata():
    """Verify metadata extraction from graph attributes and filename patterns."""
    g = create_dummy_graph(host="SiC", dopant_symbol="Ti", dopant_z=22, site="2b_0")
    filename, host, dopant, site = parse_sample_metadata(g)
    assert host == "SiC"
    assert dopant == "Ti"
    assert site == "2b_0"
    assert filename == "SiC_doped_Ti_2b_0.cif"


def test_baselines_prediction():
    """Check mean predictor and ridge regression on synthetic target values."""
    y_train = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y_test = np.array([2.0, 2.5], dtype=np.float64)

    mean_eval = evaluate_mean_predictor(y_train, y_test)
    assert np.isclose(mean_eval.predictions[0], 2.0)
    assert mean_eval.mae >= 0.0

    x_train = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype=np.float64)
    x_test = np.array([[2.0, 3.0], [2.5, 3.5]], dtype=np.float64)
    ridge_eval = evaluate_ridge(x_train, y_train, x_test, y_test)
    assert len(ridge_eval.predictions) == len(y_test)
    assert ridge_eval.mae >= 0.0


def test_train_legacy_gcn():
    """Test training legacy isotropic GCN on minimal graph data."""
    graphs = [
        create_dummy_graph(dopant_symbol="Ti", dopant_z=22, y_val=1.2),
        create_dummy_graph(dopant_symbol="V", dopant_z=23, y_val=2.1),
    ]
    # Build legacy fallback data
    for g in graphs:
        g.x = g.z.float().unsqueeze(-1) / 100.0
        g.edge_attr = torch.ones((g.edge_index.size(1), 1), dtype=torch.float)

    model = train_legacy_gcn(graphs, epochs=2, batch_size=2, seed=42)
    assert model is not None
    assert hasattr(model, "y_mean")
    assert hasattr(model, "y_std")


def test_lodo_and_loho_cv_execution():
    """Test full cross validation pipeline with synthetic graphs."""
    dummy_graphs = [
        create_dummy_graph(host="SiC", dopant_symbol="Ti", dopant_z=22, site="2b_0", y_val=1.0),
        create_dummy_graph(host="SiC", dopant_symbol="Ti", dopant_z=22, site="2a_1", y_val=1.1),
        create_dummy_graph(host="GaN", dopant_symbol="Ti", dopant_z=22, site="2b_0", y_val=0.9),
        create_dummy_graph(host="SiC", dopant_symbol="V", dopant_z=23, site="2b_0", y_val=2.0),
        create_dummy_graph(host="SiC", dopant_symbol="V", dopant_z=23, site="2a_1", y_val=2.1),
        create_dummy_graph(host="GaN", dopant_symbol="V", dopant_z=23, site="2b_0", y_val=1.9),
    ]

    samples = prepare_dataset_items(dummy_graphs)
    assert len(samples) == 6

    # Run LODO with 2 epochs for rapid test execution
    df_lodo, preds_lodo, targets_lodo, summaries = run_lodo_cv(
        samples=samples,
        epochs=2,
        batch_size=2,
        seed=42,
        run_legacy=True,
    )

    assert not df_lodo.empty
    assert "mae_ev" in df_lodo.columns
    assert "defect_gnn" in df_lodo["model"].values
    assert "ridge_baseline" in df_lodo["model"].values
    assert len(targets_lodo) == 6

    # Test plain verdict generation
    verdict = compute_plain_verdict(summaries)
    assert "Verdict:" in verdict
    assert "—" not in verdict
    assert "--" not in verdict

    # Run LOHO
    df_loho, preds_loho, targets_loho = run_loho_cv(
        samples=samples,
        epochs=2,
        batch_size=2,
        seed=42,
        run_legacy=True,
    )
    assert not df_loho.empty
    assert "LOHO_exploratory" in df_loho["cv_type"].values

    # Test Parity Plot creation
    temp_dir = tempfile.mkdtemp()
    try:
        plot_path = os.path.join(temp_dir, "parity_test.png")
        plot_parity_lodo(targets_lodo, preds_lodo, save_path=plot_path)
        assert os.path.exists(plot_path)
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    print("Running test_extract_dopant_descriptors...")
    test_extract_dopant_descriptors()
    print("Running test_parse_sample_metadata...")
    test_parse_sample_metadata()
    print("Running test_baselines_prediction...")
    test_baselines_prediction()
    print("Running test_train_legacy_gcn...")
    test_train_legacy_gcn()
    print("Running test_lodo_and_loho_cv_execution...")
    test_lodo_and_loho_cv_execution()
    print("All tests passed successfully!")

