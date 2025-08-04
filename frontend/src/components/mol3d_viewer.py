import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors
import py3Dmol

def show_molecule(smiles: str, highlight_atoms=None, view_mode='3D'):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("Invalid SMILES string.")
        return
    # Show molecular properties
    formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
    mw = Descriptors.MolWt(mol)
    st.markdown(f"**Formula:** {formula}  |  **Molecular Weight:** {mw:.2f} g/mol")
    if view_mode == '2D':
        if highlight_atoms:
            st.image(Draw.MolToImage(mol, size=(400, 400), highlightAtoms=highlight_atoms, highlightColor=(1,0,0)), caption="2D Structure (toxic atoms highlighted)")
        else:
            st.image(Draw.MolToImage(mol, size=(400, 400)), caption="2D Structure")
    else:
        mol = Chem.AddHs(mol)
        # Use ETKDGv3 for best 3D coordinates
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xf00d  # Deterministic
        embed_status = AllChem.EmbedMolecule(mol, params)
        if embed_status != 0:
            st.warning("3D embedding failed. Showing 2D structure instead.")
            st.image(Draw.MolToImage(mol, size=(400, 400)), caption="2D Structure")
            return
        # Try MMFF optimization for better geometry
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
            else:
                AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        mb = Chem.MolToMolBlock(mol)
        view = py3Dmol.view(width=400, height=400)
        view.addModel(mb, 'mol')
        if highlight_atoms:
            for idx in highlight_atoms:
                view.setStyle({'model': -1, 'serial': idx+1}, {"stick": {"color": "red"}, "sphere": {"color": "red", "opacity": 0.5}})
        else:
            view.setStyle({'stick': {}})
        view.setBackgroundColor('0xeeeeee')
        view.zoomTo()
        view.spin(True)
        mol_html = view._make_html()
        st.components.v1.html(mol_html, height=400, width=400)
