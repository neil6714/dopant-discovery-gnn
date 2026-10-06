"""Train an ensemble GNN and estimate prediction uncertainty."""
import random
import torch_geometric
import torch

from src.models.train_gnn import DopantGNN


class ActiveLearningLoop:
    def __init__(self, initial_data_path, model_save_path):
        self.data_path = initial_data_path
        self.model_save_path = model_save_path
        self.training_pool = torch.load(initial_data_path, weights_only=False)
        self.models = []  # Ensemble of models
        
    def train_ensemble(self, num_models=5, epochs=30):
        """Trains 5 independent GNNs on slightly shuffled data."""
        print(f"\n--- Training Ensemble of {num_models} Models ---")
        self.models = []
        for i in range(num_models):
            print(f"Training model {i+1}/{num_models}...")
            model = DopantGNN()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
            criterion = torch.nn.MSELoss()
            
            # Shuffle the data
            random.shuffle(self.training_pool)
            # Use 80% of the data for this model
            subset_size = int(len(self.training_pool) * 0.8)
            train_subset = self.training_pool[:subset_size]
            
            for epoch in range(epochs):
                for data in train_subset: 
                    optimizer.zero_grad()
                    out = model(data.x, data.edge_index, batch=torch.zeros(data.x.size(0), dtype=torch.long))
                    loss = criterion(out, data.y.view(1, 1))
                    loss.backward()
                    optimizer.step()
            self.models.append(model)
            
    def predict_with_uncertainty(self, new_graph):
        """Predicts with the ensemble and measures disagreement."""
        predictions = []
        for model in self.models:
            model.eval() # Set to evaluation mode
            with torch.no_grad():
                pred = model(new_graph.x, new_graph.edge_index, batch=torch.zeros(new_graph.x.size(0), dtype=torch.long))
                predictions.append(pred.item())
        
        mean_pred = sum(predictions) / len(predictions)
        
        # Calculate variance
        variance = sum((x - mean_pred) ** 2 for x in predictions) / len(predictions)
        std_dev = variance ** 0.5
        
        print(f"Model predictions: {[round(p, 4) for p in predictions]}") 
        
        return mean_pred, std_dev

if __name__ == "__main__":
    print("Initializing Active Learning Loop...")
    loop = ActiveLearningLoop(
        initial_data_path="data/graph_data/processed_graphs.pt",
        model_save_path="models/ensemble_gnn.pth"
    )
    
    # Train the initial models
    loop.train_ensemble()
    
    print("\n--- Simulating Unseen Materials Space ---")
    fake_unseen_candidates = [
        torch_geometric.data.Data(x=torch.rand(10, 1)*50, edge_index=torch.tensor([[0,1,2,3],[1,2,3,0]]), y=None, name=f"Candidate_{i}") 
        for i in range(10)
    ]
    
    # 2. Predict & Calculate Uncertainty
    candidate_scores = []
    for candidate in fake_unseen_candidates:
        mean_energy, uncertainty = loop.predict_with_uncertainty(candidate)
        candidate_scores.append({
            "name": candidate.name,
            "predicted_energy": mean_energy,
            "uncertainty": uncertainty
        })
        
    # 3. Sort by UNCERTAINTY (High uncertainty goes to the top)
    candidate_scores.sort(key=lambda x: x['uncertainty'], reverse=True)
    
    print("\nTop 3 Materials the GNN is most uncertain about:")
    for i in range(3):
        item = candidate_scores[i]
        print(f"{item['name']} | Predicted E: {item['predicted_energy']:.2f} | Uncertainty: {item['uncertainty']:.4f}")
        
   