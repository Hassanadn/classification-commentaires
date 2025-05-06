import joblib
import yaml
import os
from src.features.text_processor import TextProcessor

class TextClassifier:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Charger le modèle
        model_path = "models/model_v1.pkl"
        self.model = joblib.load(model_path)
        
        # Charger le vectorizer
        self.text_processor = TextProcessor()
        self.text_processor.load()
    
    def predict(self, text):
        text_tfidf = self.text_processor.transform([text])
        prediction = self.model.predict(text_tfidf)
        probabilities = self.model.predict_proba(text_tfidf)
        return prediction[0], probabilities[0]