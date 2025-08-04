import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from fastapi import FastAPI
from pydantic import BaseModel
from rdkit import Chem
from rdkit.Chem import rdmolops, Descriptors
import numpy as np
import os
from typing import List, Dict, Any

app = FastAPI()

class MoleculeRequest(BaseModel):
    smiles: str

class BatchRequest(BaseModel):
    smiles_list: List[str]

class ComprehensiveGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_outputs=12):
        super(ComprehensiveGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, num_outputs)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x, edge_index):
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
        x = torch.mean(x, dim=0)
        
        # Fully connected layers
        x = self.lin1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        
        return torch.sigmoid(x)

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

def smiles_to_graph(smiles, max_atoms=50):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() > max_atoms:
        return None, None
    x = atom_features(mol, max_atoms)
    adj = rdmolops.GetAdjacencyMatrix(mol)
    edge_index = torch.tensor(np.array(np.nonzero(adj)), dtype=torch.long)
    return x, edge_index

# Toxicity endpoints
TOXICITY_ENDPOINTS = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 
    'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 
    'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]

# Load the comprehensive model
model = ComprehensiveGNN(num_node_features=7, hidden_channels=64, num_outputs=len(TOXICITY_ENDPOINTS))
model_path = os.path.join(os.path.dirname(__file__), "gnn_comprehensive.pth")

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    print("✅ Loaded comprehensive GNN model")
else:
    print("⚠️ Comprehensive model not found, using untrained model")

def calculate_overall_toxicity(predictions: List[float]) -> Dict[str, Any]:
    """Calculate overall toxicity metrics from individual endpoint predictions"""
    
    # Calculate average toxicity across all endpoints
    avg_toxicity = np.mean(predictions)
    
    # Count toxic endpoints (prediction > 0.5)
    toxic_endpoints = sum(1 for p in predictions if p > 0.5)
    
    # Calculate toxicity score (weighted by endpoint importance)
    # Some endpoints are more critical than others
    endpoint_weights = {
        'NR-AR': 1.2,      # Androgen receptor
        'NR-AR-LBD': 1.2,  # Androgen receptor ligand binding
        'NR-AhR': 1.1,     # Aryl hydrocarbon receptor
        'NR-Aromatase': 1.0, # Aromatase
        'NR-ER': 1.1,      # Estrogen receptor
        'NR-ER-LBD': 1.1,  # Estrogen receptor ligand binding
        'NR-PPAR-gamma': 1.0, # PPAR gamma
        'SR-ARE': 1.0,     # Antioxidant response element
        'SR-ATAD5': 1.0,   # ATAD5
        'SR-HSE': 1.0,     # Heat shock response
        'SR-MMP': 1.0,     # Matrix metalloproteinase
        'SR-p53': 1.3      # p53 (important for DNA damage)
    }
    
    weighted_toxicity = sum(predictions[i] * endpoint_weights[endpoint] 
                           for i, endpoint in enumerate(TOXICITY_ENDPOINTS))
    weighted_toxicity /= len(TOXICITY_ENDPOINTS)
    
    # Determine overall toxicity classification
    if weighted_toxicity > 0.7:
        toxicity_class = "HIGH"
    elif weighted_toxicity > 0.5:
        toxicity_class = "MODERATE"
    else:
        toxicity_class = "LOW"
    
    return {
        "average_toxicity": float(avg_toxicity),
        "weighted_toxicity": float(weighted_toxicity),
        "toxic_endpoints_count": toxic_endpoints,
        "toxicity_class": toxicity_class,
        "endpoint_predictions": dict(zip(TOXICITY_ENDPOINTS, [float(p) for p in predictions]))
    }

@app.post("/predict")
def predict_property(request: MoleculeRequest):
    try:
        x, edge_index = smiles_to_graph(request.smiles)
        if x is None or edge_index is None:
            return {
                "error": f"Invalid SMILES or molecule has more than 50 atoms",
                "toxicity": None,
                "solubility": None,
                "intoxicant": None,
                "endpoints": None,
                "overall_toxicity": None
            }
        
        with torch.no_grad():
            predictions = model(x, edge_index)
            predictions = predictions.cpu().numpy()
        
        # Calculate overall toxicity
        overall_toxicity = calculate_overall_toxicity(predictions)
        
        # For backward compatibility, provide single toxicity score
        toxicity_score = overall_toxicity["weighted_toxicity"]
        
        # Estimate solubility based on molecular properties
        # This is a simplified estimation - in practice, you'd want a dedicated solubility model
        mol = Chem.MolFromSmiles(request.smiles)
        if mol:
            # Simple solubility estimation based on molecular weight and logP
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            
            # Simple rule: smaller, more polar molecules are more soluble
            if mw < 300 and logp < 2:
                solubility_score = 0.8
            elif mw < 500 and logp < 3:
                solubility_score = 0.6
            else:
                solubility_score = 0.4
        else:
            solubility_score = 0.5
        
        # Determine intoxicant status based on toxicity
        intoxicant = toxicity_score > 0.5
        
        return {
            "toxicity": toxicity_score,
            "solubility": solubility_score,
            "intoxicant": bool(intoxicant),
            "endpoints": overall_toxicity["endpoint_predictions"],
            "overall_toxicity": overall_toxicity,
            "prediction": toxicity_score
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "toxicity": None,
            "solubility": None,
            "intoxicant": None,
            "endpoints": None,
            "overall_toxicity": None
        }

@app.post("/batch_predict")
def batch_predict(request: BatchRequest):
    results = []
    for smiles in request.smiles_list:
        try:
            x, edge_index = smiles_to_graph(smiles)
            if x is None:
                results.append({
                    "smiles": smiles,
                    "error": "Invalid SMILES or too large molecule"
                })
                continue
            
            with torch.no_grad():
                predictions = model(x, edge_index)
                predictions = predictions.cpu().numpy()
            
            overall_toxicity = calculate_overall_toxicity(predictions)
            
            results.append({
                "smiles": smiles,
                "toxicity": overall_toxicity["weighted_toxicity"],
                "endpoints": overall_toxicity["endpoint_predictions"],
                "toxicity_class": overall_toxicity["toxicity_class"]
            })
            
        except Exception as e:
            results.append({
                "smiles": smiles,
                "error": str(e)
            })
    
    return {"results": results}

@app.get("/accuracy")
def get_accuracy():
    try:
        accuracy_path = os.path.join(os.path.dirname(__file__), "comprehensive_accuracy.txt")
        if os.path.exists(accuracy_path):
            with open(accuracy_path, "r") as f:
                accuracy = float(f.read().strip())
            return {"accuracy": accuracy}
        else:
            return {"accuracy": None, "message": "Accuracy file not found"}
    except Exception as e:
        return {"accuracy": None, "error": str(e)}

@app.get("/endpoints")
def get_endpoints():
    return {"endpoints": TOXICITY_ENDPOINTS}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 