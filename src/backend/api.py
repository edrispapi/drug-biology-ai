from fastapi import FastAPI
from pydantic import BaseModel
from src.bioinfo.sequence import load_fasta
from src.ai_models.llm_druggen import DrugGeneratorLLM
from src.chem_analysis.features import validate_smiles, extract_features
from src.ai_models.qsar_toxicity import QSARPredictor

app = FastAPI()
druggen = DrugGeneratorLLM()
qsar = QSARPredictor("models/toxicity_model.pkl")

class PredictRequest(BaseModel):
    fasta: str
    prompt: str

@app.post("/predict")
def predict(request: PredictRequest):
    # ۱. پس‌پردازش توالی هدف (در صورت نیاز)
    # ۲. تولید ترکیب با LLM
    candidates = druggen.generate(request.prompt)
    valids = validate_smiles(candidates)
    feats = extract_features(valids)
    # ۳. غربالگری سمیت با QSAR
    for f in feats:
        f['toxicity'] = qsar.predict(f)
    sorted_feats = sorted(feats, key=lambda x: (x['toxicity'], abs(x['LogP'] - 2)))
    return {"top_candidates": sorted_feats[:5]}
