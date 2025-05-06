import pickle
import yaml
from transformers import BertForSequenceClassification, BertTokenizer
from src.features.feature_engineering import FeatureEngineer
import numpy as np

class Predictor:
    def __init__(self, config):
        self.config = config
        # Charger le modèle Random Forest et le vectorizer TF-IDF
        with open("models/random_forest_v1.pkl", "rb") as f:
            model_data = pickle.load(f)
            self.rf_model = model_data['model']  # SGDClassifier
            self.tfidf = model_data['vectorizer']  # TfidfVectorizer entraîné

        # Charger le modèle BERT
        self.bert_model = BertForSequenceClassification.from_pretrained("models/bert_v1")
        self.bert_tokenizer = BertTokenizer.from_pretrained("models/bert_v1")
        # Initialiser FeatureEngineer avec le vectorizer entraîné
        self.feature_engineer = FeatureEngineer(config["config_path"], tfidf=self.tfidf)

    def predict(self, text):
        # Prédiction Random Forest
        rf_features = self.feature_engineer.transform_tfidf([text])
        rf_pred = self.rf_model.predict(rf_features)[0]
        
        # Prédiction BERT
        bert_inputs = self.feature_engineer.transform_bert([text])
        outputs = self.bert_model(**bert_inputs)
        bert_pred = np.argmax(outputs.logits.detach().numpy(), axis=1)[0]
        return {"random_forest": int(rf_pred), "bert": int(bert_pred)}

