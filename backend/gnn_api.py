import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from fastapi import FastAPI
from pydantic import BaseModel
from rdkit import Chem
from rdkit.Chem import rdmolops
import numpy as np

app = FastAPI()

MAX_ATOMS = 50  # Increased from 10 to 50 for robust support

class MoleculeRequest(BaseModel):
    smiles: str

class GNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.lin(x)
        return torch.sigmoid(x.mean(dim=0))

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    num_atoms = mol.GetNumAtoms()
    if num_atoms > MAX_ATOMS:
        return None, None  # Too large for model
    x = np.eye(MAX_ATOMS, dtype=np.float32)
    x[:num_atoms, :num_atoms] = np.eye(num_atoms, dtype=np.float32)
    adj = rdmolops.GetAdjacencyMatrix(mol)
    edge_index = np.array(np.nonzero(adj))
    return torch.tensor(x), torch.tensor(edge_index, dtype=torch.long)

# Dummy model for demonstration (replace with your trained model)
model = GNN(num_node_features=MAX_ATOMS, hidden_channels=16)
model.eval()

@app.post("/predict")
def predict_property(request: MoleculeRequest):
    x, edge_index = smiles_to_graph(request.smiles)
    if x is None:
        return {"error": f"Invalid SMILES or molecule has more than {MAX_ATOMS} atoms"}
    with torch.no_grad():
        pred = model(x, edge_index)
    return {"prediction": float(pred.item())}
