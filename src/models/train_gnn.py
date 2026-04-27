import os
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

class DopantGNN(torch.nn.Module):
    def __init__(self):
        super(DopantGNN, self).__init__()

        self.conv1 = GCNConv(1, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 64)
        
        self.out_layer = Linear(64, 1)

    def forward(self, x, edge_index, batch):

        x = self.conv1(x, edge_index)
        x = F.relu(x) 
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        
        x = global_mean_pool(x, batch)
        
        
        return self.out_layer(x)

if __name__ == "__main__":
    print("Loading graph dataset...")
    dataset = torch.load("data/graph_data/processed_graphs.pt", weights_only=False)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    # Initialize the model, optimizer, and loss function
    model = DopantGNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = torch.nn.MSELoss() # Mean Squared Error
    
    print("Starting training loop...\n")
    # Train for 50 passes over the data
    for epoch in range(1, 51):
        model.train()
        total_loss = 0
        
        for data in train_loader:
            optimizer.zero_grad() 
           
            out = model(data.x, data.edge_index, data.batch)

            loss = criterion(out, data.y.view(-1, 1))

            loss.backward() # backpropogation
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Average Loss: {total_loss/len(train_loader):.4f}")
            

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/dopant_gnn.pth")
    print("\nThe learned weights are saved to models/dopant_gnn.pth")