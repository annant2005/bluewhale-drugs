import torch
import os
from backend.gnn_api_comprehensive import ComprehensiveGNN, smiles_to_graph

# Check if model file exists
model_path = "backend/gnn_comprehensive.pth"
print(f"Model file exists: {os.path.exists(model_path)}")

if os.path.exists(model_path):
    print(f"Model file size: {os.path.getsize(model_path)} bytes")
    
    # Try to load the model
    try:
        model = ComprehensiveGNN(num_node_features=7, hidden_channels=64, num_outputs=12)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        print("✅ Model loaded successfully!")
        
        # Test with nicotine
        x, edge_index = smiles_to_graph("CN1CCCC1C2=CN=CC=C2")
        if x is not None:
            with torch.no_grad():
                predictions = model(x, edge_index)
                predictions = predictions.cpu().numpy()
                print(f"Nicotine predictions: {predictions}")
                print(f"Average prediction: {predictions.mean():.3f}")
                print(f"Max prediction: {predictions.max():.3f}")
                print(f"Min prediction: {predictions.min():.3f}")
        else:
            print("❌ Could not convert nicotine SMILES to graph")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
else:
    print("❌ Model file not found!") 