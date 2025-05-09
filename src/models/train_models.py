import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
from src.data.load_data import DataLoader
from src.models.bert_model import BertTextClassifier
from src.models.random_forest_model import RandomForest

def train_models(config_path: str):
    """Train both Random Forest and BERT models."""
    data_loader = DataLoader(config_path)

    
    # Use data_generator for batches of data
    bert_classifier = BertTextClassifier(config_path)

    # Training BERT with chunks using data_generator
    for i, chunk_df in enumerate(data_loader.data_generator()):
        print(f"=== Entraînement sur le chunk {i+1} ===")
        bert_classifier.train(chunk_df, chunk_num=i+1)
    
    bert_classifier.save_model("models/bert_model")

if __name__ == "__main__":
    config_path = "config/config.yaml"
    train_models(config_path)
