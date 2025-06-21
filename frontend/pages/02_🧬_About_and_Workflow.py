import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

st.markdown("""
<style>
.gradient-header {
    background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
    color: white;
    padding: 1.2rem 1rem 0.7rem 1rem;
    border-radius: 12px;
    font-size: 2.1rem;
    font-weight: 700;
    margin-bottom: 1.2rem;
    box-shadow: 0 2px 12px rgba(106,17,203,0.08);
}
.gradient-section {
    background: linear-gradient(90deg, #f8fafc 0%, #e0e7ff 100%);
    border-radius: 10px;
    padding: 1.2rem 2.5rem 1.2rem 2.5rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 1px 8px rgba(37,117,252,0.07);
    max-width: 1100px;
    margin-left: auto;
    margin-right: auto;
}
.gradient-divider {
    height: 4px;
    border: none;
    background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
    margin: 1.5rem 0 1.5rem 0;
    border-radius: 2px;
}
.step-box {
    background: linear-gradient(90deg, #f7971e 0%, #ffd200 100%);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.7rem;
    font-size: 1.1rem;
    box-shadow: 0 1px 6px rgba(247,151,30,0.10);
    color: #3a2c00;
}
.tech-box {
    display: inline-block;
    background: linear-gradient(90deg, #43e97b 0%, #38f9d7 100%);
    color: #222;
    border-radius: 6px;
    padding: 0.4rem 0.9rem;
    margin: 0.2rem 0.5rem 0.2rem 0;
    font-weight: 600;
    font-size: 1.05rem;
}
.workflow-chart-section {
    background: linear-gradient(90deg, #fffbe7 0%, #ffe082 100%);
    border-radius: 10px;
    padding: 1.2rem 2.5rem 1.2rem 2.5rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 1px 8px rgba(255,224,130,0.13);
    max-width: 1100px;
    margin-left: auto;
    margin-right: auto;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="gradient-header">🧬 About & Workflow</div>', unsafe_allow_html=True)

st.markdown('<div class="gradient-section">', unsafe_allow_html=True)
st.markdown("""
### About This Project
This application predicts the toxicity, solubility, and intoxicant potential of chemical compounds using a Graph Neural Network (GNN) model. It also provides detailed chemical information and 2D/3D visualization for any molecule you enter.
""")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)

st.markdown('<div class="gradient-section">', unsafe_allow_html=True)
st.markdown("""
#### What is a SMILES string?
- **SMILES** (Simplified Molecular Input Line Entry System) is a compact, human-readable way to represent a molecule's structure as a line of text using short ASCII strings.
- Each atom is represented by its atomic symbol (e.g., C for carbon, O for oxygen), and bonds are implied by the order or explicitly denoted with symbols (e.g., = for double bond, # for triple bond).
- **Rings** are represented by numbers (e.g., benzene is `c1ccccc1`), and **branches** by parentheses (e.g., isopropanol is `CC(C)O`).
- SMILES can encode complex molecules, including stereochemistry (e.g., `F[C@H](Cl)Br` for a chiral center).
- **Why is SMILES important?**
    - It allows computers to store, search, and analyze chemical structures efficiently.
    - It is the standard input for cheminformatics tools, machine learning models, and chemical databases.
    - SMILES is used for virtual screening, property prediction, and molecular design in drug discovery and materials science.
- **Examples:**
    - Ethanol: `CCO`
    - Benzene: `c1ccccc1`
    - Aspirin: `CC(=O)Oc1ccccc1C(=O)O`
    - Glucose: `C(C1C(C(C(C(O1)O)O)O)O)O`
- You can enter either a SMILES string or a common compound name (like "aspirin") in this app. The app will resolve names to SMILES automatically using PubChem.
- **Note:** Not all names can be resolved, and invalid SMILES will be flagged by the app.
""")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)

st.markdown('<div class="gradient-section">', unsafe_allow_html=True)
st.markdown("""
#### What is a Graph Neural Network (GNN)?
- A **Graph Neural Network (GNN)** is a type of deep learning model specifically designed to operate on data structured as graphs, rather than regular grids (like images) or sequences (like text).
- **Why graphs?** Many real-world systems are naturally graphs: molecules (atoms as nodes, bonds as edges), social networks, protein interactions, etc.
- **How does a GNN work?**
    - Each node (atom) starts with a feature vector (e.g., atomic number, hybridization, charge).
    - The GNN iteratively updates each node's features by aggregating information from its neighbors (other atoms it's bonded to) using a process called "message passing."
    - After several rounds, the network learns a rich, context-aware representation of each atom and the whole molecule.
- **In this app:**
    - The GNN is trained on the Tox21 dataset, which contains thousands of molecules labeled for toxicity.
    - The model learns to predict toxicity, solubility, and intoxicant potential by analyzing the molecular graph structure.
    - The GNN can generalize to new molecules by recognizing patterns in how atoms are connected and what features they have.
- **Advantages of GNNs in chemistry:**
    - Naturally handle variable-sized, complex molecular structures.
    - Capture both local (atom-level) and global (molecule-level) chemical information.
    - Outperform traditional machine learning on many molecular property prediction tasks.
- **Types of GNN layers:**
    - Graph Convolutional Networks (GCN)
    - Graph Attention Networks (GAT)
    - Message Passing Neural Networks (MPNN)
- **Limitations:**
    - GNNs require large, high-quality datasets for training.
    - They may not capture all quantum or 3D effects unless explicitly modeled.
    - Interpretability can be challenging, but tools like attention maps and feature importance help.
- **Summary:**
    - GNNs are revolutionizing cheminformatics and drug discovery by enabling accurate, scalable, and flexible molecular property prediction.
""")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)

# --- Workflow Section with improved color ---
st.markdown('<div class="gradient-section">', unsafe_allow_html=True)
st.markdown("""
### Workflow
""")
workflow_steps = [
    ("Input", "You enter a SMILES string or compound name."),
    ("Name Resolution", "If a name is entered, the app uses PubChem to find the corresponding SMILES."),
    ("Validation", "The app checks the SMILES for validity and provides feedback if there are issues."),
    ("Visualization", "The app displays 2D and 3D structures of the molecule with interactive viewers."),
    ("Prediction", "The backend GNN model receives the SMILES, converts it to a graph, and predicts toxicity, solubility, and intoxicant status."),
    ("Details", "The app fetches detailed chemical information from PubChem and displays it alongside the predictions."),
    ("Explanation", "Each prediction is explained in detail, including model limitations and interpretation guidance."),
    ("Batch Mode", "You can upload a CSV of compounds for batch predictions, with downloadable results."),
    ("Industry Use", "Explore real-world use cases and best practices for deploying these predictions in pharma and chemical industries.")
]
for i, (title, desc) in enumerate(workflow_steps, 1):
    st.markdown(f'<div class="step-box"><b>Step {i}: {title}</b><br>{desc}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)

st.markdown('<div class="gradient-section">', unsafe_allow_html=True)
st.markdown("""
### Technologies Used
""")
techs = [
    ("Streamlit", "for the interactive web interface"),
    ("RDKit", "for chemical informatics, SMILES parsing, and 2D/3D visualization"),
    ("py3Dmol", "for interactive 3D molecular visualization"),
    ("PubChemPy", "for fetching compound data from PubChem"),
    ("PyTorch & PyTorch Geometric", "for the GNN backend model"),
    ("FastAPI", "for serving the backend model as an API")
]
for name, desc in techs:
    st.markdown(f'<span class="tech-box">{name}</span> <span style="color:#555;font-size:0.98rem;">{desc}</span><br>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)

st.info("All predictions are model-based and should be used as guidance only. Always consult experimental data and experts for critical decisions.")
