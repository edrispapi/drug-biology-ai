# qsar_predictor.py
import pickle

class QSARPredictor:
    def __init__(self, model_path="models/toxicity_model.pkl"):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def predict_toxicity(self, features: dict) -> float:
        # فرض نمونه: ورودی مدل ویژگی‌های عددی
        input_vector = [
            features['MolWt'],
            features['LogP'],
            features['NumHAcceptors'],
            features['NumHDonors']
        ]
        score = self.model.predict([input_vector])[0]
        return score
