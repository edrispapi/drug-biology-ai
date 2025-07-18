from rdkit import Chem
from rdkit.Chem import Descriptors

def validate_smiles(smiles_list):
    return [s for s in smiles_list if Chem.MolFromSmiles(s)]

def extract_features(smiles_list):
    feats = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol:
            feats.append({
                "SMILES": s,
                "MolWt": Descriptors.MolWt(mol),
                "LogP": Descriptors.MolLogP(mol),
                "NumHAcceptors": Descriptors.NumHAcceptors(mol),
                "NumHDonors": Descriptors.NumHDonors(mol),
            })
    return feats
