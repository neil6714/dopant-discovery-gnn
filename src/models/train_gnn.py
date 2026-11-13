"""Train graph neural networks for defect formation energy prediction."""
import argparse
import copy
import os
import sys
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import r2_score
import torch
import torch.nn.functional as F
from torch.nn import Embedding, Linear
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import CGConv, GCNConv, global_mean_pool
from torch.utils.data import random_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import CONFIG
from src.utils.log import get_logger
from src.utils.seed import set_seed

logger = get_logger("train_gnn")

# Safe globals for PyTorch 2.6+ weights_only compatibility when loading PyG Data objects
try:
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([Data])
except Exception:
    pass


class EvaluationMetrics(NamedTuple):
    """Container for model evaluation scores."""

    mae: float
    rmse: float
    r2: float
    predictions: np.ndarray


class DopantGNN(torch.nn.Module):
    """Legacy isotropic Graph Neural Network for dopant formation energy prediction."""

    def __init__(self, in_features=1, hidden_dim=64):
        super(DopantGNN, self).__init__()
        self.embedding = Linear(in_features, hidden_dim)
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        self.mlp = torch.nn.Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, edge_index=None, batch=None, edge_attr=None, edge_weight=None):
        if hasattr(x, "edge_index"):
            data = x
            x = data.x
            edge_index = data.edge_index
            batch = getattr(data, "batch", None)
            edge_attr = getattr(data, "edge_attr", None)
            edge_weight = getattr(data, "edge_weight", None)

        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if hasattr(self, "embedding") and hasattr(self.embedding, "in_features"):
            if self.embedding.in_features == 1 and x.size(-1) > 1:
                x = x[:, 0:1] / 100.0
        if x.dtype == torch.long:
            x = x.float() / 100.0

        h = F.relu(self.embedding(x))

        if edge_weight is None and edge_attr is not None:
            if edge_attr.dim() > 1 and edge_attr.size(-1) > 1:
                edge_weight = edge_attr[:, 0]
            else:
                edge_weight = edge_attr.squeeze(-1) if edge_attr.dim() > 1 else edge_attr

        if edge_weight is not None:
            h = F.relu(self.conv1(h, edge_index, edge_weight=edge_weight))
            h = F.relu(self.conv2(h, edge_index, edge_weight=edge_weight))
            h = F.relu(self.conv3(h, edge_index, edge_weight=edge_weight))
        else:
            h = F.relu(self.conv1(h, edge_index))
            h = F.relu(self.conv2(h, edge_index))
            h = F.relu(self.conv3(h, edge_index))

        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)

        h = global_mean_pool(h, batch)
        return self.mlp(h)


class DefectGNN(torch.nn.Module):
    """Crystal graph convolution network with learned embeddings and center readout."""

    def __init__(
        self,
        in_features: int = 9,
        hidden_dim: int = 64,
        edge_dim: int = 8,
        num_elements: int = 101,
        emb_dim: int = 32,
    ):
        super(DefectGNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.y_mean: float = 0.0
        self.y_std: float = 1.0

        self.element_embedding = Embedding(num_embeddings=num_elements, embedding_dim=emb_dim)

        # in_features has element_idx at [0] and (in_features - 1) continuous descriptors
        descriptor_dim = max(0, in_features - 1)
        self.node_proj = Linear(emb_dim + descriptor_dim, hidden_dim)

        # 3 Crystal Graph Convolution (CGConv) layers
        self.conv1 = CGConv(channels=hidden_dim, dim=edge_dim, batch_norm=True)
        self.conv2 = CGConv(channels=hidden_dim, dim=edge_dim, batch_norm=True)
        self.conv3 = CGConv(channels=hidden_dim, dim=edge_dim, batch_norm=True)

        # Readout: concatenates dopant node embedding with cluster mean pooling
        self.mlp = torch.nn.Sequential(
            Linear(2 * hidden_dim, hidden_dim),
            torch.nn.SiLU(),
            Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: Union[torch.Tensor, Data, Batch],
        edge_index: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through embedding, crystal convolutions, and dual readout."""
        batch_obj = None
        if hasattr(x, "edge_index"):
            batch_obj = x
            x = batch_obj.x
            edge_index = batch_obj.edge_index
            batch = getattr(batch_obj, "batch", None)
            edge_attr = getattr(batch_obj, "edge_attr", None)
            z = getattr(batch_obj, "z", None)

        # Extract atomic number for embedding
        if z is None:
            z = x[:, 0].long().clamp(0, 100)
        descriptors = x[:, 1:]

        # Project concatenated embedding and descriptors
        elem_emb = self.element_embedding(z)
        h = self.node_proj(torch.cat([elem_emb, descriptors], dim=-1))

        # 3 CGConv message passing steps
        h = F.silu(self.conv1(h, edge_index, edge_attr=edge_attr))
        h = F.silu(self.conv2(h, edge_index, edge_attr=edge_attr))
        h = F.silu(self.conv3(h, edge_index, edge_attr=edge_attr))

        # Resolve dopant node representation
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
            dopant_h = h[0:1]
        elif batch_obj is not None and hasattr(batch_obj, "ptr") and batch_obj.ptr is not None:
            dopant_h = h[batch_obj.ptr[:-1]]
        else:
            is_dop = x[:, 6] > 0.5
            num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
            if is_dop.sum() == num_graphs:
                dopant_h = h[is_dop]
            else:
                dopant_h = global_mean_pool(h, batch)

        cluster_mean = global_mean_pool(h, batch)
        readout = torch.cat([dopant_h, cluster_mean], dim=-1)
        return self.mlp(readout)


def load_checkpoint(checkpoint_path: str, map_location: str = "cpu") -> Dict[str, Any]:
    """Load model checkpoint safely using weights_only=True."""
    return torch.load(checkpoint_path, map_location=map_location, weights_only=True)


def evaluate(
    model: torch.nn.Module,
    graphs: List[Data],
    y_mean: Optional[float] = None,
    y_std: Optional[float] = None,
    batch_size: int = 4,
    device: Optional[torch.device] = None,
) -> EvaluationMetrics:
    """Evaluate model and return MAE, RMSE, R2, and predicted energies in eV."""
    if device is None:
        device = next(model.parameters()).device

    if y_mean is None:
        y_mean = getattr(model, "y_mean", 0.0)
    if y_std is None:
        y_std = getattr(model, "y_std", 1.0)

    model.eval()
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)

    all_preds: List[float] = []
    all_targets: List[float] = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out_std = model(batch)
            preds_ev = out_std.view(-1) * y_std + y_mean
            all_preds.extend(preds_ev.cpu().numpy().tolist())

            if batch.y is not None:
                all_targets.extend(batch.y.view(-1).cpu().numpy().tolist())

    preds_arr = np.array(all_preds, dtype=np.float64)

    if all_targets:
        targets_arr = np.array(all_targets, dtype=np.float64)
        mae = float(np.mean(np.abs(preds_arr - targets_arr)))
        rmse = float(np.sqrt(np.mean((preds_arr - targets_arr) ** 2)))
        if len(targets_arr) > 1 and np.var(targets_arr) > 1e-8:
            r2 = float(r2_score(targets_arr, preds_arr))
        else:
            r2 = 0.0
    else:
        mae = float("nan")
        rmse = float("nan")
        r2 = float("nan")

    return EvaluationMetrics(mae=mae, rmse=rmse, r2=r2, predictions=preds_arr)


def train_one_model(
    train_graphs: List[Data],
    val_graphs: Optional[List[Data]] = None,
    config: Optional[Any] = None,
    epochs: Optional[int] = None,
    lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    batch_size: Optional[int] = None,
    seed: Optional[int] = None,
    patience: int = 15,
    save_path: Optional[str] = None,
    hidden_dim: int = 64,
    device: Optional[torch.device] = None,
) -> Tuple[DefectGNN, Dict[str, Any]]:
    """Train DefectGNN with target standardisation, cosine schedule, and early stopping.

    Args:
        train_graphs: Training graph list.
        val_graphs: Validation graph list for early stopping.
        config: Optional Config instance.
        epochs: Number of training epochs.
        lr: Learning rate.
        weight_decay: Weight decay.
        batch_size: Batch size for DataLoader.
        seed: Random seed.
        patience: Epochs to wait for validation MAE improvement.
        save_path: Path to save checkpoint dict.
        hidden_dim: Model hidden layer dimension.
        device: Torch compute device.

    Returns:
        Tuple of (trained DefectGNN model, training history dictionary).
    """
    if config is not None:
        if epochs is None:
            epochs = getattr(config, "train_epochs", 50)
        if lr is None:
            lr = getattr(config, "learning_rate", 1e-3)
        if weight_decay is None:
            weight_decay = getattr(config, "weight_decay", 1e-4)
        if batch_size is None:
            batch_size = getattr(config, "batch_size", 4)
        if seed is None:
            seed = getattr(config, "seed", 42)

    epochs = epochs if epochs is not None else 50
    lr = lr if lr is not None else 1e-3
    weight_decay = weight_decay if weight_decay is not None else 1e-4
    batch_size = batch_size if batch_size is not None else 4
    seed = seed if seed is not None else 42

    set_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Target standardisation fitted strictly on the training split only
    train_targets = [g.y.item() for g in train_graphs if g.y is not None]
    if train_targets:
        train_y_t = torch.tensor(train_targets, dtype=torch.float)
        y_mean = float(train_y_t.mean().item())
        y_std = float(train_y_t.std().item()) if len(train_y_t) > 1 and train_y_t.std() > 1e-6 else 1.0
    else:
        y_mean = 0.0
        y_std = 1.0

    in_feat = train_graphs[0].x.size(-1) if train_graphs and train_graphs[0].x is not None else 9
    model = DefectGNN(in_features=in_feat, hidden_dim=hidden_dim, edge_dim=8)
    model.y_mean = y_mean
    model.y_std = y_std
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = torch.nn.SmoothL1Loss()

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)

    best_val_mae = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0

    history = {"train_loss": [], "val_mae": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            pred = model(batch)
            y_std_target = (batch.y.view(-1, 1) - y_mean) / y_std

            loss = criterion(pred, y_std_target)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            batches += 1

        scheduler.step()
        avg_train_loss = total_loss / max(1, batches)
        history["train_loss"].append(avg_train_loss)

        # Validation check for early stopping
        if val_graphs:
            val_eval = evaluate(model, val_graphs, y_mean=y_mean, y_std=y_std, device=device)
            val_mae = val_eval.mae
            history["val_mae"].append(val_mae)

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_weights = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping activated at epoch %d with val MAE %.4f eV", epoch, best_val_mae)
                    break
        else:
            best_weights = copy.deepcopy(model.state_dict())

    # Load best model checkpoint weights
    model.load_state_dict(best_weights)

    # Save checkpoint dict
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        checkpoint = {
            "state_dict": model.state_dict(),
            "config": {
                "in_features": in_feat,
                "hidden_dim": hidden_dim,
                "edge_dim": 8,
                "lr": lr,
                "weight_decay": weight_decay,
                "epochs": epochs,
            },
            "y_mean": float(y_mean),
            "y_std": float(y_std),
            "seed": int(seed),
        }
        torch.save(checkpoint, save_path)
        logger.info("Saved checkpoint to %s", save_path)

    return model, history


def train_model(
    data_path: str = "data/graph_data/processed_graphs.pt",
    save_path: str = "models/dopant_gnn.pth",
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 4,
    seed: int = 42,
    test_split: float = 0.2,
) -> DefectGNN:
    """Train DefectGNN on the graph dataset and report held-out test split evaluation.

    Args:
        data_path: Path to processed graphs tensor dataset.
        save_path: Destination path for checkpoint.
        epochs: Training epochs.
        lr: Learning rate.
        weight_decay: Weight decay.
        batch_size: Batch size.
        seed: Random seed.
        test_split: Fraction of dataset reserved for held-out evaluation.

    Returns:
        Trained DefectGNN model instance.
    """
    set_seed(seed)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    dataset = torch.load(data_path, weights_only=False)
    if len(dataset) < 2:
        raise ValueError(f"Dataset at {data_path} has fewer than 2 graphs.")

    test_len = max(1, int(len(dataset) * test_split))
    train_len = len(dataset) - test_len
    train_data, test_data = random_split(
        dataset,
        [train_len, test_len],
        generator=torch.Generator().manual_seed(seed),
    )

    train_list = [train_data[i] for i in range(len(train_data))]
    test_list = [test_data[i] for i in range(len(test_data))]

    print(f"Loaded {len(dataset)} graphs: {len(train_list)} train, {len(test_list)} test.")

    model, _ = train_one_model(
        train_graphs=train_list,
        val_graphs=test_list,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        seed=seed,
        save_path=save_path,
    )

    # Held-out evaluation
    metrics = evaluate(model, test_list, y_mean=model.y_mean, y_std=model.y_std)
    print(
        f"Held-out Test Scores | MAE: {metrics.mae:.4f} eV | RMSE: {metrics.rmse:.4f} eV | R2: {metrics.r2:.4f}"
    )

    return model


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entrypoint for training DefectGNN."""
    parser = argparse.ArgumentParser(
        description="Train DefectGNN model on graph dataset."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=CONFIG.graph_data_path,
        help="Path to graphs dataset.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=CONFIG.dopant_gnn_path,
        help="Path to output checkpoint file.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=CONFIG.train_epochs,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=CONFIG.learning_rate,
        help="Learning rate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=CONFIG.batch_size,
        help="Training batch size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=CONFIG.seed,
        help="Random seed.",
    )
    args = parser.parse_args(argv)

    if not os.path.exists(args.data_path):
        print(f"Dataset file {args.data_path} not found. Skipping execution.")
        return

    train_model(
        data_path=args.data_path,
        save_path=args.save_path,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
