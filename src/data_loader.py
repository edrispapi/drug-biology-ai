from Bio import SeqIO
from rdkit import Chem

def load_protein_sequence(fasta_path: str) -> str:
    record = SeqIO.read(fasta_path, "fasta")
    return str(record.seq)

def validate_smiles(smiles_list: list[str]) -> list[str]:
    valid_smiles = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            valid_smiles.append(s)
    return valid_smiles
