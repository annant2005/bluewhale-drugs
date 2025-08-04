import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

st.title("🏭 Real-World Industry Use Cases")
st.markdown("""
## How Can This Project Be Used in Industry?

This project leverages advanced machine learning (Graph Neural Networks) and cheminformatics to predict toxicity, solubility, and intoxicant potential of chemical compounds. Here are several detailed, real-world use cases for this technology in industry:

---
### 1. Pharmaceutical Drug Discovery
- **Early Toxicity Screening:** Rapidly filter out molecules with high predicted toxicity before expensive synthesis and biological testing.
- **Lead Optimization:** Guide medicinal chemists in modifying molecular structures to reduce toxicity and improve solubility, increasing the chance of clinical success.
- **Regulatory Submissions:** Provide computational evidence of safety for new drug candidates as part of regulatory filings (e.g., FDA, EMA).

---
### 2. Agrochemical Development
- **Pesticide Safety:** Screen new pesticide candidates for toxicity to humans, animals, and the environment before field trials.
- **Green Chemistry:** Design safer, more environmentally friendly agrochemicals by predicting and minimizing off-target toxicity.

---
### 3. Chemical Manufacturing & Materials Science
- **Process Safety:** Assess the toxicity of intermediates and byproducts in chemical manufacturing to ensure worker and environmental safety.
- **Material Additives:** Evaluate the safety of new additives in plastics, coatings, and packaging materials.

---
### 4. Cosmetics & Consumer Products
- **Ingredient Safety:** Screen new cosmetic ingredients for toxicity and solubility to comply with safety regulations and avoid harmful products.
- **Rapid Prototyping:** Enable R&D teams to quickly assess the safety of novel formulations before animal or human testing.

---
### 5. Environmental & Regulatory Compliance
- **Pollutant Risk Assessment:** Predict the toxicity of industrial chemicals and pollutants to support environmental risk assessments and remediation planning.
- **REACH & TSCA Compliance:** Help companies comply with chemical safety regulations (e.g., EU REACH, US TSCA) by providing computational toxicity data.

---
### 6. Academic & Contract Research
- **Virtual Screening:** Support academic labs and CROs in high-throughput virtual screening of chemical libraries for safety and drug-likeness.
- **Education:** Teach students and professionals about cheminformatics, machine learning, and molecular safety assessment.

---
### Why is this important?
- **Cost Savings:** Reduces the need for expensive and time-consuming lab experiments by prioritizing safe, promising compounds.
- **Ethical Impact:** Minimizes animal testing by providing reliable in silico predictions.
- **Speed:** Accelerates R&D cycles and time-to-market for new products.
- **Regulatory Readiness:** Provides documentation and evidence for regulatory submissions and audits.

---
**In summary:** This project empowers industry professionals to make safer, faster, and more informed decisions about chemicals and drugs, supporting innovation while protecting human health and the environment.
""")

# --- Add a chart for use case impact ---
use_cases = [
    "Pharma", "Agrochemical", "Manufacturing", "Cosmetics", "Environment", "Academic"
]
impact = [9, 7, 6, 5, 8, 4]
fig, ax = plt.subplots(figsize=(7,3))
bars = ax.bar(use_cases, impact, color=plt.cm.plasma(np.linspace(0,1,len(use_cases))))
ax.set_ylabel('Impact Score (1-10)')
ax.set_title('Industry Use Case Impact')
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'{yval}', ha='center', va='bottom')
st.pyplot(fig)
