import yaml
import os
import logging
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer

# Configurer le logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FeatureEngineer:
    def __init__(self, config_path, tfidf=None):
        self.config = yaml.safe_load(open(config_path, 'r'))
        # Utiliser le vectorizer fourni ou en créer un nouveau
        self.tfidf = tfidf if tfidf is not None else TfidfVectorizer(max_features=5000)
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.config['model']['bert']['model_name'])
        logging.info("FeatureEngineer initialisé avec succès")

    def fit_tfidf(self, texts):
        """Ajuster le vectorizer TF-IDF sur les textes fournis."""
        self.tfidf.fit(texts)
        logging.info("TF-IDF vectorizer ajusté")
        return self

    def transform_tfidf(self, texts):
        """Transformer les textes en représentations TF-IDF."""
        return self.tfidf.transform(texts).toarray()

    def transform_bert(self, texts, max_length=128):
        """Transformer les textes en représentations BERT."""
        
        encodings = self.bert_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        logging.info(f"Textes transformés en encodings BERT (taille du lot : {len(texts)})")
        return encodings