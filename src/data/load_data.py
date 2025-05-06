import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
import os

def load_and_process_data(config_path="config/config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    data_path = config['data']['raw_path']
    df = pd.read_csv(data_path)
    
    # Nettoyage de base
    df.dropna(subset=['text'], inplace=True)
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df.dropna(subset=['label'], inplace=True)
    df['label'] = df['label'].astype(int)
    
    # Sauvegarde des données traitées
    processed_path = config['data']['processed_path']
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    
    # Split train/test
    X = df['text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config['data']['test_size'], 
        random_state=config['data']['random_state']
    )
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_process_data()
    print("Données chargées et traitées avec succès")