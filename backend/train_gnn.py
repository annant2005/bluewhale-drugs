import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
from rdkit import Chem
from rdkit.Chem import rdmolops
import pandas as pd
import numpy as np
import os

def atom_features(mol, max_atoms=10):
    # Simple feature: atomic number, padded/truncated to max_atoms
    feats = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    feats = feats[:max_atoms] + [0]*(max_atoms-len(feats))
    return torch.tensor([feats], dtype=torch.float).T  # shape (max_atoms, 1)

def load_dataset(csv_path, max_atoms=10):
    df = pd.read_csv(csv_path)
    data_list = []
    for idx, row in df.iterrows():
        smiles = row['smiles']
        label = row['label']
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        x = atom_features(mol, max_atoms)
        adj = rdmolops.GetAdjacencyMatrix(mol)
        edge_index = torch.tensor(np.array(np.nonzero(adj)), dtype=torch.long)
        # For demo, use label as toxicity and (1-label) as dummy solubility
        y = torch.tensor([label, 1-label], dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
    return data_list

class GNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_outputs=2):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, num_outputs)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.lin(x)
        return torch.sigmoid(x.mean(dim=0))

def train_gnn(data_list, epochs=20, batch_size=4):
    model = GNN(num_node_features=1, hidden_channels=16, num_outputs=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = F.binary_cross_entropy(out, batch.y.float().mean(dim=0))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")
    # Save in backend/gnn_trained.pth
    save_path = os.path.join(os.path.dirname(__file__), "gnn_trained.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved as {save_path}")

if __name__ == "__main__":
    data_list = load_dataset(os.path.join(os.path.dirname(__file__), "..", "data", "tox21_sample.csv"))
    train_gnn(data_list, epochs=20, batch_size=4)
