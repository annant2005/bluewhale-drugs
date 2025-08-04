import pubchempy as pcp

# Example: Replace or expand this list with thousands of names for real use
names = [
    "paracetamol", "aspirin", "caffeine", "ibuprofen", "nicotine", "acetaminophen",
    "sodium chloride", "glucose", "ethanol", "water", "acetone", "benzene",
    "chloroform", "methane", "carbon dioxide", "ammonia", "hydrochloric acid",
    "sulfuric acid", "nitric acid", "sodium hydroxide", "potassium permanganate",
    # ...add hundreds/thousands more names here...
]

substance_dict = {}
for name in names:
    try:
        compounds = pcp.get_compounds(name, 'name')
        if compounds and compounds[0].isomeric_smiles:
            substance_dict[name.lower()] = compounds[0].isomeric_smiles
            print(f"{name}: {compounds[0].isomeric_smiles}")
        else:
            print(f"{name}: Not found")
    except Exception as e:
        print(f"{name}: Error - {e}")

# Save to a Python file
with open("frontend/src/components/substance_dictionary.py", "w") as f:
    f.write("SUBSTANCE_DICTIONARY = {\n")
    for k, v in substance_dict.items():
        f.write(f"    '{k}': '{v}',\n")
    f.write("}\n") 