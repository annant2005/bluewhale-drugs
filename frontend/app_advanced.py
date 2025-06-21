import streamlit as st
import requests
from rdkit import Chem
from rdkit.Chem import Draw
from src.components.mol3d_viewer import show_molecule
import pubchempy as pcp
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Drug Toxicity Prediction", layout="wide")
# Add custom CSS for wider content and professional color sections
st.markdown("""
<style>
/* Make main container wider */
.css-18e3th9 { max-width: 1200px; }
/* Section headers with color */
.section-header { font-size: 1.5em; font-weight: bold; margin-top: 1em; margin-bottom: 0.5em; color: #222; }
.section-tox { background: #fff4f4; border-left: 6px solid #e53935; padding: 1em; border-radius: 8px; }
.section-sol { background: #f0f8ff; border-left: 6px solid #1e88e5; padding: 1em; border-radius: 8px; }
.section-intox { background: #f9fbe7; border-left: 6px solid #cddc39; padding: 1em; border-radius: 8px; }
.section-pubchem { background: #f3e5f5; border-left: 6px solid #8e24aa; padding: 1em; border-radius: 8px; }
.section-pubchem .section-header { color: #4a148c; }
.section-tox .section-header { color: #b71c1c; }
.section-sol .section-header { color: #1565c0; }
.section-intox .section-header { color: #827717; }
</style>
""", unsafe_allow_html=True)
st.title("💊 Drug Toxicity Prediction (GNN)")

def name_to_smiles(query):
    mol = Chem.MolFromSmiles(query)
    if mol:
        return query, None
    try:
        compounds = pcp.get_compounds(query, 'name')
        if compounds and compounds[0].isomeric_smiles:
            return compounds[0].isomeric_smiles, None
        else:
            return None, f"Could not resolve '{query}' to a SMILES string."
    except Exception as e:
        return None, f"Error resolving name: {e}"

def get_pubchem_cid(smiles):
    try:
        from pubchempy import get_compounds
        compounds = get_compounds(smiles, 'smiles')
        if compounds and compounds[0].cid:
            return compounds[0].cid
    except Exception:
        pass
    return None

def show_pubchem_details(cid):
    import pubchempy as pcp
    compound = pcp.Compound.from_cid(cid)
    st.markdown(f"**PubChem CID:** {compound.cid}")
    st.image(f"https://pubchem.ncbi.nlm.nih.gov/image/imgsrv.fcgi?cid={compound.cid}&t=l", caption="2D Structure")
    st.image(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{compound.cid}/PNG?record_type=3d", caption="3D Structure")
    if hasattr(compound, 'safety') and compound.safety:
        st.markdown(f"**Chemical Safety:** {compound.safety}")
    st.markdown(f"**Molecular Formula:** {compound.molecular_formula}")
    if hasattr(compound, 'canonical_smiles'):
        st.markdown(f"**SMILES:** `{compound.canonical_smiles}`")
    if compound.synonyms:
        st.markdown("**Synonyms:**")
        for syn in compound.synonyms[:5]:
            st.markdown(f"- {syn}")
        if len(compound.synonyms) > 5:
            st.markdown(f"...and {len(compound.synonyms)-5} more.")
    st.markdown(f"**Molecular Weight:** {compound.molecular_weight} g/mol")
    if hasattr(compound, 'record') and compound.record:
        if 'CreateDate' in compound.record:
            st.markdown(f"**Create:** {compound.record['CreateDate']}")
        if 'ModifyDate' in compound.record:
            st.markdown(f"**Modify:** {compound.record['ModifyDate']}")
    if hasattr(compound, 'description') and compound.description:
        st.markdown(f"**Description:** {compound.description}")
    if hasattr(compound, 'source') and compound.source:
        st.markdown(f"**Source:** {compound.source}")

# Query parameters for navigation
query_params = st.query_params
detail_cid = query_params.get('cid', [None])[0] if 'cid' in query_params else None

if detail_cid:
    show_pubchem_details(detail_cid)
    if st.button("Back to Results"):
        st.query_params.clear()
else:
    user_input = st.text_area("Enter one or more SMILES strings or compound names (one per line):", "CCO")
    view_mode = st.radio("View mode:", ["2D", "3D"], horizontal=True)
    if st.button("Predict"):
        queries = [q.strip() for q in user_input.splitlines() if q.strip()]
        results_table = []
        for query in queries:
            smiles, name_error = name_to_smiles(query)
            if smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    highlight_atoms = None
                    toxicity = None
                    cid = get_pubchem_cid(smiles)
                    with st.spinner(f"Predicting for {query}..."):
                        try:
                            response = requests.post("http://localhost:8000/predict", json={"smiles": smiles})
                            if response.ok:
                                result = response.json()
                                toxicity = result.get("toxicity")
                                solubility = result.get("solubility")
                                intoxicant = result.get("intoxicant")
                                if toxicity is not None and toxicity > 0.5:
                                    highlight_atoms = list(range(mol.GetNumAtoms()))
                                results_table.append({
                                    "Input": query,
                                    "SMILES": smiles,
                                    "Toxicity": toxicity,
                                    "Solubility": solubility,
                                    "Intoxicant": 'Yes' if intoxicant else 'No',
                                    "CID": cid
                                })
                                with st.expander(f"Details for {query}"):
                                    show_molecule(smiles, highlight_atoms=highlight_atoms, view_mode=view_mode)
                                    st.markdown(f"**SMILES:** `{smiles}`")
                                    if cid:
                                        st.markdown('<div class="section-header section-pubchem">PubChem Details</div>', unsafe_allow_html=True)
                                        show_pubchem_details(cid)
                                    else:
                                        st.warning("No PubChem CID found for this compound.")
                                    st.markdown('<div class="section-header section-tox">Prediction Explanations</div>', unsafe_allow_html=True)
                                    st.markdown('<div class="section-tox">', unsafe_allow_html=True)
                                    st.markdown("""
### Toxicity
- **Definition:** Toxicity is the degree to which a substance can damage an organism. In this context, it refers to the probability that the molecule is toxic to biological systems, as predicted by a Graph Neural Network (GNN) trained on the Tox21 dataset.
- **How to interpret:**
    - **Score close to 1:** High likelihood of toxicity. The molecule may cause adverse biological effects and should be handled with caution.
    - **Score close to 0:** Low likelihood of toxicity. The molecule is predicted to be relatively safe, but experimental validation is always recommended.
- **Model limitations:**
    - The prediction is based on known toxicological data and molecular structure. It does not account for all possible biological contexts, dosages, or long-term effects.
    - False positives/negatives are possible. Always consult experimental data for critical decisions.
""")
                                    st.markdown('</div>', unsafe_allow_html=True)
                                    st.markdown('<div class="section-header section-sol">Solubility</div>', unsafe_allow_html=True)
                                    st.markdown('<div class="section-sol">', unsafe_allow_html=True)
                                    st.markdown("""
- **Definition:** Solubility is the ability of a substance to dissolve in a solvent (here, water or biological fluids). The score estimates the probability that the molecule is soluble.
- **How to interpret:**
    - **Score close to 1:** The molecule is likely to dissolve well in water/biological fluids, which can affect absorption, distribution, and excretion in living organisms.
    - **Score close to 0:** The molecule is likely poorly soluble, which may limit its bioavailability or effectiveness as a drug.
- **Model limitations:**
    - The solubility score is a model estimate, not a direct measurement. Real-world solubility can be affected by pH, temperature, and other factors.
    - In this demo, the solubility score may be a placeholder and not based on experimental data.
""")
                                    st.markdown('</div>', unsafe_allow_html=True)
                                    st.markdown('<div class="section-header section-intox">Intoxicant</div>', unsafe_allow_html=True)
                                    st.markdown('<div class="section-intox">', unsafe_allow_html=True)
                                    st.markdown("""
- **Definition:** An intoxicant is a substance that can cause intoxication or adverse effects on the central nervous system. Here, it is a binary label based on the predicted toxicity score.
- **How to interpret:**
    - **Yes:** The model predicts the molecule is likely to be intoxicating (toxicity score > 0.5).
    - **No:** The molecule is predicted to be non-intoxicating (toxicity score ≤ 0.5).
- **Model limitations:**
    - This is a simplified threshold and does not reflect regulatory or clinical definitions of intoxicants.
    - Use this as a quick flag for potentially hazardous molecules, but always consult detailed toxicological and regulatory data for safety-critical applications.
""")
                                    st.markdown('</div>', unsafe_allow_html=True)
                                    st.markdown('<hr>', unsafe_allow_html=True)
                            else:
                                st.error(f"API error: {response.status_code} - {response.text}")
                        except Exception as e:
                            st.error(f"Request failed: {e}")
                else:
                    st.error(f"Invalid SMILES string for '{query}'.")
            else:
                st.error(name_error or f"Invalid input: {query}")
        if results_table:
            st.markdown("## Summary Table")
            df = pd.DataFrame(results_table)
            # Color formatting for toxicity and solubility
            def color_toxicity(val):
                if pd.isnull(val):
                    return ''
                if val > 0.7:
                    return 'background-color: #ffcccc; color: #b30000'  # Red
                elif val > 0.5:
                    return 'background-color: #fff2cc; color: #b36b00'  # Orange
                else:
                    return 'background-color: #ccffcc; color: #006600'  # Green
            def color_solubility(val):
                if pd.isnull(val):
                    return ''
                if val > 0.7:
                    return 'background-color: #ccffcc; color: #006600'  # Green
                elif val > 0.5:
                    return 'background-color: #fff2cc; color: #b36b00'  # Orange
                else:
                    return 'background-color: #ffcccc; color: #b30000'  # Red
            styled_df = df.style.applymap(color_toxicity, subset=['Toxicity']) \
                              .applymap(color_solubility, subset=['Solubility'])
            st.dataframe(styled_df, use_container_width=True)
            # --- Add charts ---
            if 'Toxicity' in df and 'Solubility' in df:
                st.markdown("### 📈 Toxicity & Solubility Distribution")
                x = np.arange(len(df))
                width = 0.35
                fig1, ax1 = plt.subplots(figsize=(8, 4))
                # Glossy gradient colors matching About & Workflow section
                from matplotlib.patches import Rectangle
                def glossy_bar(ax, x, height, width, color1, color2, label=None):
                    for i, h in enumerate(height):
                        bar = Rectangle((x[i] - width/2, 0), width, h, color=color1, zorder=2, label=label if i==0 else "")
                        ax.add_patch(bar)
                        # Overlay a glossy gradient
                        ax.add_patch(Rectangle((x[i] - width/2, 0), width, h, zorder=3, 
                            facecolor=color2, alpha=0.35, label=None))
                # Toxicity: purple gradient (#6a11cb to #2575fc)
                glossy_bar(ax1, x - width/2, df['Toxicity'], width, '#6a11cb', '#2575fc', label='Toxicity')
                # Solubility: green-cyan gradient (#43e97b to #38f9d7)
                glossy_bar(ax1, x + width/2, df['Solubility'], width, '#43e97b', '#38f9d7', label='Solubility')
                ax1.set_xlim(-1, len(df))
                ax1.set_xlabel('Compound Index')
                ax1.set_ylabel('Score')
                ax1.set_title('Toxicity & Solubility Distribution', fontsize=14, fontweight='bold')
                ax1.set_xticks(x)
                ax1.set_xticklabels([str(i+1) for i in x], rotation=0)
                ax1.legend(['Toxicity', 'Solubility'], loc='upper right')
                ax1.grid(axis='y', linestyle='--', alpha=0.4, zorder=1)
                for i, h in enumerate(df['Toxicity']):
                    ax1.text(x[i] - width/2, h + 0.01, f'{h:.2f}', ha='center', va='bottom', fontsize=9, color='#6a11cb', fontweight='bold')
                for i, h in enumerate(df['Solubility']):
                    ax1.text(x[i] + width/2, h + 0.01, f'{h:.2f}', ha='center', va='bottom', fontsize=9, color='#43e97b', fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig1)
            if 'Intoxicant' in df:
                st.markdown("### 🧪 Intoxicant Class Distribution")
                intoxicant_counts = df['Intoxicant'].value_counts()
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
