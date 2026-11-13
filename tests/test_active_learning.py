"""Unit tests for active learning loop, ensemble uncertainty, and pool benchmark."""
import os
import shutil
import sys
import tempfile
import torch
from torch_geometric.data import Data

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.active_learning import (
    ActiveLearningLoop,
    _instantiate_and_load_checkpoint,
    load_ensemble_models,
    predict_with_uncertainty,
    run_pool_benchmark,
)
from src.models.train_gnn import DefectGNN


def create_synthetic_graph(dopant_z: int = 22, y_val: float = 2.0, host: str = "SiC") -> Data:
    """Create minimal synthetic PyG graph for testing."""
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
    data.filename = f"{host}_doped_{dopant_z}.cif"
    data.host = host
    data.dopant_z = dopant_z
    data.site = "2b_0"
    data.ood_flag = False
    return data


def test_predict_with_uncertainty_and_scaling():
    """Verify ensemble prediction and unstandardized uncertainty in eV."""
    # Create two synthetic models with known scaling
    m1 = DefectGNN(in_features=9, hidden_dim=32, edge_dim=8)
    m1.y_mean = 5.0
    m1.y_std = 2.0

    m2 = DefectGNN(in_features=9, hidden_dim=32, edge_dim=8)
    m2.y_mean = 7.0
    m2.y_std = 2.0

    graph = create_synthetic_graph(y_val=6.0)

    mean_pred, std_unc = predict_with_uncertainty(graph, ensemble_models=[m1, m2])
    assert isinstance(mean_pred, float)
    assert isinstance(std_unc, float)
    assert std_unc >= 0.0


def test_checkpoint_dict_loading():
    """Verify loading and instantiating from checkpoint dictionary with standardization statistics."""
    temp_dir = tempfile.mkdtemp()
    try:
        model = DefectGNN(in_features=9, hidden_dim=32, edge_dim=8)
        ckpt = {
            "state_dict": model.state_dict(),
            "config": {"in_features": 9, "hidden_dim": 32, "edge_dim": 8},
            "y_mean": 3.45,
            "y_std": 1.12,
            "seed": 42,
        }
        loaded_model = _instantiate_and_load_checkpoint(ckpt)
        assert loaded_model.y_mean == 3.45
        assert loaded_model.y_std == 1.12

        # Save to disk as ensemble member
        path = os.path.join(temp_dir, "ensemble_model_0.pth")
        torch.save(ckpt, path)

        loaded_list = load_ensemble_models(model_dir=temp_dir, num_models=1)
        assert len(loaded_list) == 1
        assert loaded_list[0].y_mean == 3.45
    finally:
        shutil.rmtree(temp_dir)


def test_active_learning_loop_empty_models_fallback():
    """Verify that an empty self.models falls back to loading from disk instead of raising."""
    temp_dir = tempfile.mkdtemp()
    try:
        model = DefectGNN(in_features=9, hidden_dim=32, edge_dim=8)
        ckpt = {
            "state_dict": model.state_dict(),
            "config": {"in_features": 9, "hidden_dim": 32, "edge_dim": 8},
            "y_mean": 2.5,
            "y_std": 1.0,
            "seed": 42,
        }
        torch.save(ckpt, os.path.join(temp_dir, "ensemble_model_0.pth"))

        loop = ActiveLearningLoop(initial_data_path="nonexistent.pt", model_save_path=temp_dir)
        assert loop.models == []

        # Calling predict_with_uncertainty with empty models should fall back to disk
        graph = create_synthetic_graph()
        mean_pred, std_dev = loop.predict_with_uncertainty(graph)
        assert isinstance(mean_pred, float)
        assert len(loop.models) == 1
    finally:
        shutil.rmtree(temp_dir)


def test_train_ensemble_bootstrapping():
    """Verify train_ensemble runs with numpy Generator bootstrapping and saves checkpoint dicts."""
    temp_dir = tempfile.mkdtemp()
    try:
        graphs = [
            create_synthetic_graph(dopant_z=22, y_val=1.0),
            create_synthetic_graph(dopant_z=23, y_val=2.0),
            create_synthetic_graph(dopant_z=24, y_val=3.0),
            create_synthetic_graph(dopant_z=25, y_val=4.0),
        ]
        loop = ActiveLearningLoop(initial_data_path="dummy.pt", model_save_path=temp_dir)
        loop.training_pool = graphs

        trained_models = loop.train_ensemble(num_models=2, epochs=2, batch_size=2)
        assert len(trained_models) == 2
        assert os.path.exists(os.path.join(temp_dir, "ensemble_model_0.pth"))
        assert os.path.exists(os.path.join(temp_dir, "ensemble_model_1.pth"))

        # Verify saved checkpoint content
        ckpt0 = torch.load(os.path.join(temp_dir, "ensemble_model_0.pth"), weights_only=False)
        assert "state_dict" in ckpt0
        assert "y_mean" in ckpt0
        assert "y_std" in ckpt0
        assert "seed" in ckpt0
    finally:
        shutil.rmtree(temp_dir)


def test_run_pool_benchmark():
    """Verify pool benchmark runs both uncertainty and random acquisition, saving csv and plot."""
    temp_dir = tempfile.mkdtemp()
    try:
        graphs = [
            create_synthetic_graph(dopant_z=22, y_val=1.0),
            create_synthetic_graph(dopant_z=23, y_val=2.0),
            create_synthetic_graph(dopant_z=24, y_val=3.0),
            create_synthetic_graph(dopant_z=25, y_val=4.0),
            create_synthetic_graph(dopant_z=26, y_val=5.0),
            create_synthetic_graph(dopant_z=27, y_val=6.0),
        ]
        out_csv = os.path.join(temp_dir, "al_benchmark.csv")
        out_plot = os.path.join(temp_dir, "al_learning_curve.png")
        out_calib = os.path.join(temp_dir, "uncertainty_calibration.csv")

        df_bench, df_calib = run_pool_benchmark(
            strategy="both",
            seed_size=2,
            batch_size=2,
            budget=2,
            n_repeats=1,
            dataset=graphs,
            epochs=2,
            num_models=2,
            output_csv=out_csv,
            output_plot=out_plot,
            output_calibration_csv=out_calib,
        )

        assert os.path.exists(out_csv)
        assert os.path.exists(out_plot)
        assert os.path.exists(out_calib)
        assert not df_bench.empty
        assert not df_calib.empty
        assert "mae_ev" in df_bench.columns
        assert "spearman_corr" in df_calib.columns
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    print("Running test_predict_with_uncertainty_and_scaling...")
    test_predict_with_uncertainty_and_scaling()
    print("Running test_checkpoint_dict_loading...")
    test_checkpoint_dict_loading()
    print("Running test_active_learning_loop_empty_models_fallback...")
    test_active_learning_loop_empty_models_fallback()
    print("Running test_train_ensemble_bootstrapping...")
    test_train_ensemble_bootstrapping()
    print("Running test_run_pool_benchmark...")
    test_run_pool_benchmark()
    print("All active learning tests passed successfully!")
