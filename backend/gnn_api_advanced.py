import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from rdkit import Chem
from rdkit.Chem import rdmolops
import numpy as np
import pandas as pd
from typing import List
import os

app = FastAPI()

class MoleculeRequest(BaseModel):
    smiles: str

class BatchRequest(BaseModel):
    smiles_list: List[str]

class GNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_outputs=2, use_attention=False):
        super(GNN, self).__init__()
        if use_attention:
            self.conv1 = GATConv(num_node_features, hidden_channels)
            self.conv2 = GATConv(hidden_channels, hidden_channels)
        else:
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

# Model expects 2 outputs
MAX_ATOMS = 50  # Max atoms per molecule
ATOM_FEATURES = 1  # Only atomic number as feature
model = GNN(num_node_features=ATOM_FEATURES, hidden_channels=16, num_outputs=2)
model.eval()

MODEL_PATH = os.path.join(os.path.dirname(__file__), "gnn_trained.pth")
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()

def atom_features(mol, max_atoms=MAX_ATOMS):
    feats = [[atom.GetAtomicNum()] for atom in mol.GetAtoms()]
    feats = feats[:max_atoms] + [[0]]*(max_atoms-len(feats))
    return torch.tensor(feats, dtype=torch.float)

def smiles_to_graph(smiles, max_atoms=MAX_ATOMS):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() > max_atoms:
        return None, None
    x = atom_features(mol, max_atoms)
    adj = rdmolops.GetAdjacencyMatrix(mol)
    edge_index = torch.tensor(np.array(np.nonzero(adj)), dtype=torch.long)
    return x, edge_index

@app.post("/predict")
def predict_property(request: MoleculeRequest):
    try:
        x, edge_index = smiles_to_graph(request.smiles)
        if x is None or edge_index is None:
            return {"error": f"Invalid SMILES or molecule has more than {MAX_ATOMS} atoms", "toxicity": None, "solubility": None, "intoxicant": None, "prediction": None}
        with torch.no_grad():
            pred = model(x, edge_index)
        tox = float(pred[0].item()) if pred.shape[0] > 0 else None
        sol = float(pred[1].item()) if pred.shape[0] > 1 else None
        intoxicant = 1 if tox and tox > 0.5 else 0  # Dummy logic: if toxicity > 0.5, consider intoxicant
        return {
            "toxicity": tox,
            "solubility": sol,
            "intoxicant": intoxicant,
            "prediction": tox
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e), "toxicity": None, "solubility": None, "intoxicant": None, "prediction": None}

@app.post("/batch_predict")
def batch_predict(request: BatchRequest):
    results = []
    for smiles in request.smiles_list:
        x, edge_index = smiles_to_graph(smiles)
        if x is None:
            results.append({"smiles": smiles, "error": f"Invalid SMILES or >{MAX_ATOMS} atoms"})
            continue
        with torch.no_grad():
            pred = model(x, edge_index)
        results.append({"smiles": smiles, "toxicity": float(pred[0].item()), "solubility": float(pred[1].item())})
    return {"results": results}

@app.post("/csv_predict")
def csv_predict(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    smiles_list = df['smiles'].tolist()
    batch_results = []
    for smiles in smiles_list:
        x, edge_index = smiles_to_graph(smiles)
        if x is None:
            batch_results.append({"smiles": smiles, "error": f"Invalid SMILES or >{MAX_ATOMS} atoms"})
            continue
        with torch.no_grad():
            pred = model(x, edge_index)
        batch_results.append({"smiles": smiles, "toxicity": float(pred[0].item()), "solubility": float(pred[1].item())})
    return {"results": batch_results}

@app.post("/compare_models")
def compare_models(request: MoleculeRequest):
    x, edge_index = smiles_to_graph(request.smiles)
    if x is None:
        return {"error": f"Invalid SMILES or >{MAX_ATOMS} atoms"}
    with torch.no_grad():
        gnn_pred = model(x, edge_index)
    rf_pred = rf_model.predict(x.numpy())
    return {
        "gnn": {"toxicity": float(gnn_pred[0].item()), "solubility": float(gnn_pred[1].item())},
        "rf": {"toxicity": float(rf_pred[0]), "solubility": float(rf_pred[1])}
    }

@app.post("/explain")
def explain(request: MoleculeRequest):
    x, edge_index = smiles_to_graph(request.smiles)
    if x is None:
        return {"error": f"Invalid SMILES or >{MAX_ATOMS} atoms"}
    # Dummy: assign random importance to each atom
    importances = np.random.rand(x.shape[0]).tolist()
    return {"importances": importances}

@app.get("/accuracy")
def get_accuracy():
    acc_path = os.path.join(os.path.dirname(__file__), "accuracy.txt")
    if os.path.exists(acc_path):
        with open(acc_path, "r") as f:
            acc = f.read().strip()
        return {"accuracy": float(acc)}
    else:
        return {"accuracy": None, "error": "Accuracy not available. Train the model first."}
