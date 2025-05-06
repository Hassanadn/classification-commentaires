from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
import yaml 

class TextProcessor:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.vectorizer = TfidfVectorizer(
            max_features=self.config['model']['max_features'],
            ngram_range=(self.config['model']['ngram_range'][0], 
                        self.config['model']['ngram_range'][1])
        )
        
    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)
    
    def transform(self, texts):
        return self.vectorizer.transform(texts)
    
    def save(self, path="models/vectorizer.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.vectorizer, path)
        
    def load(self, path="models/vectorizer.pkl"):
        self.vectorizer = joblib.load(path)