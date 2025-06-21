import streamlit as st
import requests
from rdkit import Chem
from rdkit.Chem import Draw

st.set_page_config(page_title="Drug Discovery with GNNs", layout="centered")

st.title("💊 Drug Discovery using Graph Neural Networks (GNNs)")
st.markdown("""
This app predicts properties of drug molecules using AI.  
Enter a molecule in SMILES format (e.g., `CCO` for ethanol, `C1=CC=CC=C1` for benzene).
""")

smiles = st.text_input("Enter SMILES string:", "CCO")

if st.button("Predict"):
    if smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            st.image(Draw.MolToImage(mol, size=(300, 300)), caption="Molecule Structure")
            with st.spinner("Predicting..."):
                response = requests.post("http://localhost:8000/predict", json={"smiles": smiles})
                if response.ok:
                    result = response.json()
                    if "prediction" in result:
                        st.success(f"Predicted Property Score: {result['prediction']:.3f}")
                    else:
                        st.error(result.get("error", "Unknown error"))
                else:
                    st.error("API error. Is the backend running?")
        else:
            st.error("Invalid SMILES string.")
    else:
        st.warning("Please enter a SMILES string.")
