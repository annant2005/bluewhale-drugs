from fastapi import APIRouter
from pydantic import BaseModel
from drugs.backend.gnn_api import smiles_to_graph, GNN, MAX_ATOMS
import torch
from drugs.backend.gnn_api_advanced import smiles_to_graph as adv_smiles_to_graph, model as adv_model, MAX_ATOMS as ADV_MAX_ATOMS
import numpy as np

# Dummy model for demonstration (replace with your trained model)
model = GNN(num_node_features=MAX_ATOMS, hidden_channels=16)
model.eval()

router = APIRouter()

class AgentRequest(BaseModel):
    goal: str
    smiles: str = None

@router.post("/agent")
def agentic_action(request: AgentRequest):
    # Simple rule-based planning for demonstration
    if "batch" in request.goal.lower() and request.smiles:
        # Assume comma-separated SMILES for batch
        smiles_list = [s.strip() for s in request.smiles.split(",") if s.strip()]
        results = []
        for smiles in smiles_list:
            x, edge_index = adv_smiles_to_graph(smiles)
            if x is None:
                results.append({"smiles": smiles, "error": f"Invalid SMILES or >{ADV_MAX_ATOMS} atoms"})
                continue
            with torch.no_grad():
                pred = adv_model(x, edge_index)
            results.append({"smiles": smiles, "toxicity": float(pred[0].item()), "solubility": float(pred[1].item())})
        return {"agent_action": "batch_prediction", "results": results}
    elif "explain" in request.goal.lower() and request.smiles:
        x, edge_index = adv_smiles_to_graph(request.smiles)
        if x is None:
            return {"error": f"Invalid SMILES or >{ADV_MAX_ATOMS} atoms"}
        importances = np.random.rand(x.shape[0]).tolist()
        return {"agent_action": "explanation", "importances": importances}
    elif "predict" in request.goal.lower() and request.smiles:
        x, edge_index = adv_smiles_to_graph(request.smiles)
        if x is None:
            return {"error": f"Invalid SMILES or molecule has more than {ADV_MAX_ATOMS} atoms"}
        with torch.no_grad():
            pred = adv_model(x, edge_index)
        tox = float(pred[0].item()) if pred.shape[0] > 0 else None
        sol = float(pred[1].item()) if pred.shape[0] > 1 else None
        intoxicant = 1 if tox and tox > 0.5 else 0
        return {"agent_action": "prediction", "toxicity": tox, "solubility": sol, "intoxicant": intoxicant}
    return {"error": "Unknown goal or missing parameters. Please specify a valid goal and input."} 