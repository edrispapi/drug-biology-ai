from fastapi import FastAPI
from pydantic import BaseModel
from src.data_loader import load_protein_sequence, validate_smiles
from src.drug_generator import DrugGenerator
from src.feature_extractor import extract_features
from src.qsar_predictor import QSARPredictor
from src.ranking import rank_candidates

app = FastAPI()

class InputData(BaseModel):
    fasta: str  # دنباله پروتئین به صورت رشته
    prompt: str # متن ورودی برای تولید ترکیبات

drug_generator = DrugGenerator()
tox_model = QSARPredictor()

@app.post("/predict_drugs")
def predict_drugs(data: InputData):
    seq = data.fasta
    prompt = data.prompt

    # تولید ترکیبات دارویی بر اساس prompt
    raw_smiles = drug_generator.generate_candidates(prompt)
    valid_smiles = validate_smiles(raw_smiles)

    # استخراج ویژگی
    features = extract_features(valid_smiles)

    # پیش‌بینی سمیت
    for feat in features:
        feat['Toxicity'] = tox_model.predict_toxicity(feat)

    # رتبه‌بندی
    ranked = rank_candidates(features)

    # بازگشت ۵ ترکیب برتر
    return {"top_candidates": ranked[:5]}
