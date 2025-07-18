# feature_extractor.py
from rdkit import Chem
from rdkit.Chem import Descriptors

def extract_features(smiles_list: list[str]) -> list[dict]:
    features = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            features.append({
                "SMILES": smi,
                "MolWt": Descriptors.MolWt(mol),
                "LogP": Descriptors.MolLogP(mol),
                "NumHAcceptors": Descriptors.NumHAcceptors(mol),
                "NumHDonors": Descriptors.NumHDonors(mol),
            })
    return features
