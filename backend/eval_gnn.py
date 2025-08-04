import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
from rdkit import Chem
from rdkit.Chem import rdmolops
import pandas as pd
import numpy as np
import os

MAX_ATOMS = 60

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

def atom_features(mol, max_atoms=MAX_ATOMS):
    feats = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    feats = feats[:max_atoms] + [0]*(max_atoms-len(feats))
    return torch.tensor([feats], dtype=torch.float).T

def load_dataset(csv_path, max_atoms=MAX_ATOMS):
    df = pd.read_csv(csv_path)
    data_list = []
    for idx, row in df.iterrows():
        smiles = row['smiles']
        label = row['label']
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        if mol.GetNumAtoms() > max_atoms:
            continue
        x = atom_features(mol, max_atoms)
        adj = rdmolops.GetAdjacencyMatrix(mol)
        edge_index = torch.tensor(np.array(np.nonzero(adj)), dtype=torch.long)
        y = torch.tensor([label, 1-label], dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
    return data_list

if __name__ == "__main__":
    data_list = load_dataset(os.path.join("data", "tox21_sr-p53.csv"))
    split_idx = int(0.8 * len(data_list))
    val_data = data_list[split_idx:]
    print(f"Validation set size: {len(val_data)}")
    model = GNN(num_node_features=1, hidden_channels=16, num_outputs=2)
    model_path = os.path.join("backend", "gnn_trained.pth")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    if val_data:
        val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch.x, batch.edge_index)
                pred_tox = (out[0] > 0.5).float().item()
                true_tox = batch.y[0].item()
                if pred_tox == true_tox:
                    correct += 1
                total += 1
        acc = correct / total if total > 0 else 0.0
        print(f"Validation Toxicity Accuracy: {acc:.2%}")
    else:
        print("No validation data available.") 