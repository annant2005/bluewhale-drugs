import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from src.components.mol3d_viewer import show_molecule
import numpy as np

st.title("📊 Batch Prediction & Analysis")
st.markdown("""
Upload a CSV file with a column named `smiles` or `name` to predict toxicity, solubility, and intoxicant status for multiple compounds at once.
""")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    smiles_col = None
    for col in df.columns:
        if col.lower() in ["smiles", "name"]:
            smiles_col = col
            break
    if not smiles_col:
        st.error("No 'smiles' or 'name' column found in the uploaded file.")
    else:
        st.success(f"Found column: {smiles_col}")
        results = []
        for i, val in enumerate(df[smiles_col]):
            if pd.isna(val):
                continue
            # Use the same name_to_smiles logic as main app
            from app_advanced import name_to_smiles
            smiles, _ = name_to_smiles(val)
            if not smiles:
                results.append({"Input": val, "Error": "Could not resolve to SMILES"})
                continue
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                results.append({"Input": val, "Error": "Invalid SMILES"})
                continue
            # Dummy: In real app, call backend API for prediction
            # Here, random values for demo
            toxicity = np.random.rand()
            solubility = np.random.rand()
            intoxicant = 'Yes' if toxicity > 0.5 else 'No'
            results.append({"Input": val, "SMILES": smiles, "Toxicity": toxicity, "Solubility": solubility, "Intoxicant": intoxicant})
        results_df = pd.DataFrame(results)
        st.dataframe(results_df)
        st.info("Batch prediction will use the same backend as the main page. For demo, results are placeholders.")
        # --- Add charts ---
        if not results_df.empty and 'Toxicity' in results_df and 'Solubility' in results_df:
            st.markdown("### 📈 Toxicity & Solubility Distribution")
            import matplotlib.pyplot as plt
            x = np.arange(len(results_df))
            width = 0.35
            fig, ax = plt.subplots(figsize=(8, 4))
            bars1 = ax.bar(x - width/2, results_df['Toxicity'], width, label='Toxicity', color='#1976d2', edgecolor='#222')
            bars2 = ax.bar(x + width/2, results_df['Solubility'], width, label='Solubility', color='#43a047', edgecolor='#222')
            ax.set_xlabel('Compound Index')
            ax.set_ylabel('Score')
            ax.set_title('Toxicity & Solubility Distribution', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([str(i+1) for i in x], rotation=0)
            ax.legend(loc='upper right')
            ax.grid(axis='y', linestyle='--', alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown("### 🧪 Intoxicant Class Distribution")
            intoxicant_counts = results_df['Intoxicant'].value_counts()
            # Enhanced donut-style pie chart
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            colors = ['#1976d2', '#e53935'] if 'Yes' in intoxicant_counts.index else ['#43a047', '#e53935']
            wedges, texts, autotexts = ax2.pie(
                intoxicant_counts,
                labels=intoxicant_counts.index,
                autopct='%1.0f%%',
                startangle=90,
                colors=colors,
                wedgeprops=dict(width=0.4, edgecolor='w'),
                textprops={'fontsize': 12}
            )
            ax2.set_title('Intoxicant Distribution', fontsize=13, fontweight='bold')
            ax2.legend(wedges, intoxicant_counts.index, title="Intoxicant", loc="center left", bbox_to_anchor=(1, 0.5))
            plt.setp(autotexts, size=13, weight="bold", color='white')
            ax2.axis('equal')
            st.pyplot(fig2)
