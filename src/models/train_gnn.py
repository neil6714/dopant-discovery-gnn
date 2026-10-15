"""Train the graph neural network on the generated graph dataset."""
import os
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

# Safe globals for PyTorch 2.6+ weights_only compatibility
try:
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([Data])
except Exception:
    pass


class DopantGNN(torch.nn.Module):
    """Edge-aware Graph Neural Network for dopant formation energy prediction."""

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
        # Support passing a PyG Data / Batch object directly as first argument
        if hasattr(x, "edge_index"):
            data = x
            x = data.x
            edge_index = data.edge_index
            batch = getattr(data, "batch", None)
            edge_attr = getattr(data, "edge_attr", None)
            edge_weight = getattr(data, "edge_weight", None)

        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if x.dtype == torch.long:
            x = x.float() / 100.0

        h = F.relu(self.embedding(x))

        # Use bond distance / edge attributes as edge_weight if not directly passed
        if edge_weight is None and edge_attr is not None:
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


def train_model(
    data_path="data/graph_data/processed_graphs.pt",
    save_path="models/dopant_gnn.pth",
    epochs=50,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=4,
    seed=42,
):
    """Train DopantGNN using AdamW and SmoothL1Loss, and save weights."""
    if seed is not None:
        torch.manual_seed(seed)

    print(f"Loading graph dataset from {data_path}...")
    dataset = torch.load(data_path, weights_only=False)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, optimizer (AdamW), and loss (SmoothL1Loss)
    model = DopantGNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.SmoothL1Loss()

    print("Starting training loop...\n")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for data in train_loader:
            optimizer.zero_grad()
            edge_attr = getattr(data, "edge_attr", None)
            out = model(data.x, data.edge_index, data.batch, edge_attr=edge_attr)
            loss = criterion(out, data.y.view(-1, 1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Average Loss: {total_loss / max(1, len(train_loader)):.4f}")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\nThe learned weights are saved to {save_path}")
    return model


if __name__ == "__main__":
    train_model()