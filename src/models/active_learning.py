"""Train an ensemble GNN and estimate prediction uncertainty."""
import os
import random
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.models.train_gnn import DopantGNN
from src.data_pipeline.structures_to_graphs import build_graph_from_cif

# Safe globals for PyTorch 2.6+ weights_only compatibility
try:
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([Data])
except Exception:
    pass


def predict_with_uncertainty(cif_or_graph, ensemble_models=None):
    """Predict energy and epistemic uncertainty using an ensemble of DopantGNN models.

    Args:
        cif_or_graph (str or Data): CIF file path or PyG Data object.
        ensemble_models (list[DopantGNN], optional): Trained ensemble models.

    Returns:
        tuple[float, float]: (mean_energy, std_uncertainty)
    """
    if ensemble_models is None:
        ensemble_models = []
        for i in range(5):
            path = f"models/ensemble_model_{i}.pth"
            if os.path.exists(path):
                m = DopantGNN()
                m.load_state_dict(torch.load(path, weights_only=False))
                ensemble_models.append(m)

        if not ensemble_models and os.path.exists("models/dopant_gnn.pth"):
            m = DopantGNN()
            m.load_state_dict(torch.load("models/dopant_gnn.pth", weights_only=False))
            ensemble_models.append(m)

    if not ensemble_models:
        raise ValueError("No ensemble models provided or found on disk in models/")

    if isinstance(cif_or_graph, str):
        graph = build_graph_from_cif(cif_or_graph)
    else:
        graph = cif_or_graph

    predictions = []
    batch = torch.zeros(graph.x.size(0), dtype=torch.long)
    edge_attr = getattr(graph, "edge_attr", None)

    for model in ensemble_models:
        model.eval()
        with torch.no_grad():
            pred = model(graph.x, graph.edge_index, batch=batch, edge_attr=edge_attr)
            predictions.append(float(pred.item()))

    mean_pred = sum(predictions) / len(predictions)
    if len(predictions) > 1:
        variance = sum((p - mean_pred) ** 2 for p in predictions) / (len(predictions) - 1)
        std_dev = variance ** 0.5
    else:
        std_dev = 0.0

    return float(mean_pred), float(std_dev)


class ActiveLearningLoop:
    """Active learning workflow using a bootstrapped GNN ensemble."""

    def __init__(self, initial_data_path="data/graph_data/processed_graphs.pt", model_save_path="models"):
        self.data_path = initial_data_path
        self.model_save_path = model_save_path
        if os.path.exists(initial_data_path):
            self.training_pool = torch.load(initial_data_path, weights_only=False)
        else:
            self.training_pool = []
        self.models = []

    def train_ensemble(self, num_models=5, epochs=30, batch_size=4, lr=1e-3, weight_decay=1e-4):
        """Train independent DopantGNN models with distinct seeds and bootstrapped sampling."""
        print(f"\n--- Training Bootstrapped Ensemble of {num_models} Models ---")
        os.makedirs(self.model_save_path, exist_ok=True)
        self.models = []
        n_samples = len(self.training_pool)

        if n_samples == 0:
            print("Warning: training_pool is empty.")
            return self.models

        for i in range(num_models):
            print(f"Training ensemble model {i + 1}/{num_models}...")
            torch.manual_seed(42 + i)
            random.seed(42 + i)

            model = DopantGNN()
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            criterion = torch.nn.SmoothL1Loss()

            # Bootstrapped sampling with replacement
            bootstrap_indices = [random.randint(0, n_samples - 1) for _ in range(n_samples)]
            bootstrap_subset = [self.training_pool[idx] for idx in bootstrap_indices]
            loader = DataLoader(bootstrap_subset, batch_size=batch_size, shuffle=True)

            model.train()
            for epoch in range(epochs):
                for data in loader:
                    optimizer.zero_grad()
                    edge_attr = getattr(data, "edge_attr", None)
                    out = model(data.x, data.edge_index, data.batch, edge_attr=edge_attr)
                    loss = criterion(out, data.y.view(-1, 1))
                    loss.backward()
                    optimizer.step()

            # Save individual model
            model_file = os.path.join(self.model_save_path, f"ensemble_model_{i}.pth")
            torch.save(model.state_dict(), model_file)
            self.models.append(model)

        print(f"Saved {num_models} ensemble models to {self.model_save_path}/")
        return self.models

    def predict_with_uncertainty(self, new_graph_or_cif):
        """Predicts with the ensemble and measures epistemic uncertainty."""
        return predict_with_uncertainty(new_graph_or_cif, ensemble_models=self.models)


if __name__ == "__main__":
    loop = ActiveLearningLoop()
    if loop.training_pool:
        loop.train_ensemble(num_models=5, epochs=30)
    else:
        print("Dataset not found at data/graph_data/processed_graphs.pt")