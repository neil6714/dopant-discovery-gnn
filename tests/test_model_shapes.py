"""Unit tests for DefectGNN model forward output shapes and checkpoint round trips."""
import numpy as np
import torch
from torch_geometric.data import Batch, Data

from src.models.train_gnn import DefectGNN, load_checkpoint


def _create_synthetic_data(dopant_z: int = 22) -> Data:
    """Generate minimal synthetic graph for shape verification."""
    x = torch.tensor(
        [
            [float(dopant_z), 1.54, 1.45, 4.0, 4.0, 4.0, 1.0, 1.0, 0.0],
            [14.0, 1.90, 1.11, 14.0, 3.0, 4.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float,
    )
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr = torch.ones((2, 8), dtype=torch.float)
    y = torch.tensor([1.25], dtype=torch.float)
    z = torch.tensor([dopant_z, 14], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, z=z)


def test_defect_gnn_forward_shapes():
    """Verify DefectGNN batch forward pass produces [B, 1] output tensor."""
    g1 = _create_synthetic_data(dopant_z=22)
    g2 = _create_synthetic_data(dopant_z=23)
    batch = Batch.from_data_list([g1, g2])

    model = DefectGNN(in_features=9, hidden_dim=32, edge_dim=8)
    model.eval()

    with torch.no_grad():
        out = model(batch)

    assert out.shape == (2, 1)


def test_checkpoint_round_trip_reproduces_predictions(tmp_path):
    """Verify saving and loading model checkpoint preserves exact predictions."""
    g = _create_synthetic_data(dopant_z=22)
    batch = Batch.from_data_list([g])

    model = DefectGNN(in_features=9, hidden_dim=32, edge_dim=8)
    model.y_mean = 2.45
    model.y_std = 1.15
    model.eval()

    with torch.no_grad():
        pred_before = float(model(batch).view(-1).item() * model.y_std + model.y_mean)

    ckpt_path = tmp_path / "model_ckpt.pth"
    checkpoint = {
        "state_dict": model.state_dict(),
        "config": {"in_features": 9, "hidden_dim": 32, "edge_dim": 8},
        "y_mean": model.y_mean,
        "y_std": model.y_std,
        "seed": 42,
    }
    torch.save(checkpoint, str(ckpt_path))

    # Load back and verify reproduction
    loaded_ckpt = load_checkpoint(str(ckpt_path))
    new_model = DefectGNN(in_features=9, hidden_dim=32, edge_dim=8)
    new_model.load_state_dict(loaded_ckpt["state_dict"])
    new_model.y_mean = loaded_ckpt["y_mean"]
    new_model.y_std = loaded_ckpt["y_std"]
    new_model.eval()

    with torch.no_grad():
        pred_after = float(new_model(batch).view(-1).item() * new_model.y_std + new_model.y_mean)

    assert np.isclose(pred_before, pred_after, atol=1e-6)
