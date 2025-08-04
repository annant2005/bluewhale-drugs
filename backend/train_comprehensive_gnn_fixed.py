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
from sklearn.metrics import accuracy_score

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

def load_comprehensive_dataset(csv_path, max_atoms=50):
    """Load dataset with all toxicity endpoints"""
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
        
        # Get toxicity labels for all endpoints
        toxicity_labels = []
        for endpoint in toxicity_endpoints:
            value = row.get(endpoint, np.nan)
            if pd.isna(value) or value == '':
                toxicity_labels.append(0.0)  # Non-toxic for missing data (FIXED!)
            else:
                toxicity_labels.append(float(value))
        
        # Create features
        x = atom_features(mol, max_atoms)
        adj = rdmolops.GetAdjacencyMatrix(mol)
        edge_index = torch.tensor(np.array(np.nonzero(adj)), dtype=torch.long)
        
        # Create target tensor with all toxicity endpoints
        y = torch.tensor(toxicity_labels, dtype=torch.float)
        
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
    
    print(f"Loaded {len(data_list)} molecules, skipped {skipped}")
    
    # Print some statistics about the data
    toxic_counts = []
    for data in data_list:
        toxic_count = (data.y > 0.5).sum().item()
        toxic_counts.append(toxic_count)
    
    print(f"Average toxic endpoints per molecule: {np.mean(toxic_counts):.2f}")
    print(f"Max toxic endpoints: {max(toxic_counts)}")
    print(f"Min toxic endpoints: {min(toxic_counts)}")
    
    return data_list, toxicity_endpoints

class ComprehensiveGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_outputs=12):
        super(ComprehensiveGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, num_outputs)
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
        
        # Global pooling - handle both single and batched inputs
        if batch is not None:
            # Batched: use scatter_mean for proper pooling
            from torch_geometric.nn import global_mean_pool
            x = global_mean_pool(x, batch)
        else:
            # Single molecule: use mean pooling
            x = torch.mean(x, dim=0)
        
        # Fully connected layers
        x = self.lin1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        
        return torch.sigmoid(x)

def train_comprehensive_gnn(data_list, toxicity_endpoints, epochs=30, batch_size=16, lr=0.001):
    """Train the comprehensive GNN model"""
    
    # Split data
    train_data, val_data = train_test_split(data_list, test_size=0.2, random_state=42)
    
    # Create model
    model = ComprehensiveGNN(num_node_features=7, hidden_channels=64, num_outputs=len(toxicity_endpoints))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"Training on {len(train_data)} samples, validating on {len(val_data)} samples")
    print(f"Model outputs: {len(toxicity_endpoints)} toxicity endpoints")
    print(f"Batch size: {batch_size}")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            
            # Reshape batch.y to match output shape
            target = batch.y.view(-1, len(toxicity_endpoints))
            
            # Calculate loss
            loss = F.binary_cross_entropy(out, target)
            
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
                target = batch.y.view(-1, len(toxicity_endpoints))
                loss = F.binary_cross_entropy(out, target)
                val_loss += loss.item()
                
                # Store predictions and targets
                val_predictions.extend((out > 0.5).float().cpu().numpy())
                val_targets.extend(target.cpu().numpy())
        
        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Calculate accuracy for each endpoint
        val_predictions = np.array(val_predictions)
        val_targets = np.array(val_targets)
        
        endpoint_accuracies = []
        for i in range(len(toxicity_endpoints)):
            # Only consider samples where we have valid labels (not 0.5)
            valid_mask = val_targets[:, i] != 0.5
            if valid_mask.sum() > 0:
                acc = accuracy_score(val_targets[valid_mask, i], val_predictions[valid_mask, i])
                endpoint_accuracies.append(acc)
        
        avg_accuracy = np.mean(endpoint_accuracies) if endpoint_accuracies else 0
        
        print(f"Epoch {epoch+1:3d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Avg Acc: {avg_accuracy:.3f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), "gnn_comprehensive.pth"))
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "gnn_comprehensive.pth")))
    
    # Final evaluation
    model.eval()
    final_predictions = []
    final_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            out = model(batch.x, batch.edge_index, batch.batch)
            target = batch.y.view(-1, len(toxicity_endpoints))
            final_predictions.extend(out.cpu().numpy())
            final_targets.extend(target.cpu().numpy())
    
    final_predictions = np.array(final_predictions)
    final_targets = np.array(final_targets)
    
    # Calculate final accuracies
    print("\nFinal Endpoint Accuracies:")
    print("-" * 50)
    total_accuracy = 0
    valid_endpoints = 0
    
    for i, endpoint in enumerate(toxicity_endpoints):
        valid_mask = final_targets[:, i] != 0.5
        if valid_mask.sum() > 0:
            acc = accuracy_score(final_targets[valid_mask, i], (final_predictions[valid_mask, i] > 0.5).astype(int))
            print(f"{endpoint:15s}: {acc:.3f}")
            total_accuracy += acc
            valid_endpoints += 1
    
    overall_accuracy = total_accuracy / valid_endpoints if valid_endpoints > 0 else 0
    print(f"{'Overall':15s}: {overall_accuracy:.3f}")
    
    # Save accuracy
    with open(os.path.join(os.path.dirname(__file__), "comprehensive_accuracy.txt"), "w") as f:
        f.write(f"{overall_accuracy:.4f}")
    
    print(f"\nModel saved as gnn_comprehensive.pth")
    print(f"Overall accuracy: {overall_accuracy:.3f}")
    
    return model, toxicity_endpoints

if __name__ == "__main__":
    # Load comprehensive dataset
    csv_path = os.path.join(os.path.dirname(__file__), "..", "tox21.csv")
    data_list, toxicity_endpoints = load_comprehensive_dataset(csv_path)
    
    # Train model with smaller batch size to avoid memory issues
    model, endpoints = train_comprehensive_gnn(data_list, toxicity_endpoints, epochs=30, batch_size=16) 