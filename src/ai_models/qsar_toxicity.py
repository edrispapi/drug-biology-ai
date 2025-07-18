import pickle

class QSARPredictor:
    def __init__(self, model_path):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
    def predict(self, features):
        x = [[features['MolWt'], features['LogP'], features['NumHAcceptors'], features['NumHDonors']]]
        return float(self.model.predict(x)[0])
