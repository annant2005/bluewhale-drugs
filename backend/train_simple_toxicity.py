import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit.Chem import rdmolops
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

def atom_features(mol, max_atoms=50):
    """Enhanced atom features including atomic number, degree, formal charge, etc."""
    feats = []
    for atom in mol.GetAtoms():
        feat = [
            atom.GetAtomicNum(),  # Atomic number
            atom.GetDegree(),      # Number of bonds
            atom.GetFormalCharge(), # Formal charge
            atom.GetImplicitValence(), # Implicit valence
            atom.GetIsAromatic(),  # Is aromatic
            atom.GetNumRadicalElectrons(), # Radical electrons
            atom.GetHybridization(), # Hybridization
        ]
        feats.append(feat)
    
    # Pad to max_atoms
    while len(feats) < max_atoms:
        feats.append([0] * 7)  # Zero padding
    feats = feats[:max_atoms]
    
    return torch.tensor(feats, dtype=torch.float)

def load_simple_dataset(csv_path, max_atoms=50):
    """Load dataset and convert 12 endpoints to single toxicity score"""
    df = pd.read_csv(csv_path)
    
    # Define toxicity endpoints
    toxicity_endpoints = [
        'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 
        'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 
        'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
    ]
    
    data_list = []
    skipped = 0
    
    for idx, row in df.iterrows():
        smiles = row['smiles']
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None or mol.GetNumAtoms() > max_atoms:
            skipped += 1
            continue
        
        # Get toxicity labels for all endpoints and calculate average
        toxicity_scores = []
        for endpoint in toxicity_endpoints:
            value = row.get(endpoint, np.nan)
            if pd.isna(value) or value == '':
                toxicity_scores.append(0.0)  # Non-toxic for missing data
            else:
                toxicity_scores.append(float(value))
        
        # Calculate overall toxicity score (average of all endpoints)
        overall_toxicity = np.mean(toxicity_scores)
        
        # Create features
        x = atom_features(mol, max_atoms)
        adj = rdmolops.GetAdjacencyMatrix(mol)
        edge_index = torch.tensor(np.array(np.nonzero(adj)), dtype=torch.long)
        
        # Create target tensor with single toxicity score
        y = torch.tensor([overall_toxicity], dtype=torch.float)
        
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
    
    print(f"Loaded {len(data_list)} molecules, skipped {skipped}")
    
    # Print statistics about the toxicity scores
    toxicity_scores = [data.y.item() for data in data_list]
    print(f"Average toxicity score: {np.mean(toxicity_scores):.4f}")
    print(f"Max toxicity score: {np.max(toxicity_scores):.4f}")
    print(f"Min toxicity score: {np.min(toxicity_scores):.4f}")
    print(f"Compounds with toxicity > 0.5: {sum(1 for s in toxicity_scores if s > 0.5)}")
    print(f"Compounds with toxicity > 0.3: {sum(1 for s in toxicity_scores if s > 0.3)}")
    
    return data_list

class SimpleToxicityGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(SimpleToxicityGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 1)  # Single output
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x, edge_index, batch=None):
        # Graph convolutions
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        # Global pooling
        if batch is not None:
            from torch_geometric.nn import global_mean_pool
            x = global_mean_pool(x, batch)
        else:
            x = torch.mean(x, dim=0)
        
        # Fully connected layers
        x = self.lin1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        
        return torch.sigmoid(x)  # Output between 0 and 1

def train_simple_gnn(data_list, epochs=50, batch_size=32, lr=0.001):
    """Train the simple toxicity GNN model"""
    
    # Split data
    train_data, val_data = train_test_split(data_list, test_size=0.2, random_state=42)
    
    # Create model
    model = SimpleToxicityGNN(num_node_features=7, hidden_channels=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"Training on {len(train_data)} samples, validating on {len(val_data)} samples")
    print(f"Model outputs: 1 toxicity score (0-1)")
    print(f"Batch size: {batch_size}")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            
            # Calculate loss (MSE for regression)
            loss = F.mse_loss(out.squeeze(), batch.y.squeeze())
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = F.mse_loss(out.squeeze(), batch.y.squeeze())
                val_loss += loss.item()
                
                # Store predictions and targets
                val_predictions.extend(out.squeeze().cpu().numpy())
                val_targets.extend(batch.y.squeeze().cpu().numpy())
        
        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Calculate accuracy (predictions > 0.5 vs targets > 0.5)
        val_pred_binary = [1 if p > 0.5 else 0 for p in val_predictions]
        val_target_binary = [1 if t > 0.5 else 0 for t in val_targets]
        accuracy = accuracy_score(val_target_binary, val_pred_binary)
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(val_targets, val_predictions))
        
        scheduler.step(val_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Accuracy: {accuracy:.4f}, RMSE: {rmse:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), "gnn_simple_toxicity.pth")
        else:
            patience_counter += 1
            if patience_counter >= 20:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Load best model
    model.load_state_dict(torch.load("gnn_simple_toxicity.pth"))
    
    # Final evaluation
    model.eval()
    final_predictions = []
    final_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            out = model(batch.x, batch.edge_index, batch.batch)
            final_predictions.extend(out.squeeze().cpu().numpy())
            final_targets.extend(batch.y.squeeze().cpu().numpy())
    
    final_pred_binary = [1 if p > 0.5 else 0 for p in final_predictions]
    final_target_binary = [1 if t > 0.5 else 0 for t in final_targets]
    final_accuracy = accuracy_score(final_target_binary, final_pred_binary)
    final_rmse = np.sqrt(mean_squared_error(final_targets, final_predictions))
    
    print(f"\nFinal Results:")
    print(f"Accuracy: {final_accuracy:.4f}")
    print(f"RMSE: {final_rmse:.4f}")
    
    # Save accuracy
    with open("simple_toxicity_accuracy.txt", "w") as f:
        f.write(f"{final_accuracy:.4f}")
    
    return model

if __name__ == "__main__":
    print("🚀 Training Simple Toxicity GNN Model")
    print("=" * 50)
    
    # Load dataset
    data_list = load_simple_dataset("../tox21.csv")
    
    # Train model
    model = train_simple_gnn(data_list, epochs=50, batch_size=32, lr=0.001)
    
    print("\n✅ Training completed!")
    print("Model saved as: gnn_simple_toxicity.pth")
    print("Accuracy saved as: simple_toxicity_accuracy.txt") 