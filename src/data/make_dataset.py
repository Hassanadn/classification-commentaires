"""
Scripts pour télécharger ou générer des données
"""
import os
import pandas as pd
from pathlib import Path

def load_data(filepath):
    """
    Charge les données depuis un fichier
    """
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    elif filepath.endswith('.json'):
        return pd.read_json(filepath)
    else:
        raise ValueError(f"Format de fichier non supporté: {filepath}")

def preprocess_data(df):
    """
    Prétraite les données textuelles
    """
    # Exemple simple de prétraitement
    df['text_clean'] = df['text'].str.lower()
    return df

def split_data(df, test_size=0.2, val_size=0.1):
    """
    Divise les données en ensembles d'entraînement, validation et test
    """
    from sklearn.model_selection import train_test_split
    
    # D'abord, séparer les données de test
    train_val, test = train_test_split(df, test_size=test_size, random_state=42)
    
    # Ensuite, séparer les données d'entraînement et de validation
    train, val = train_test_split(train_val, test_size=val_size/(1-test_size), random_state=42)
    
    return train, val, test

def main():
    """
    Point d'entrée principal pour le prétraitement des données
    """
    # Définir les chemins
    project_dir = Path(__file__).resolve().parents[2]
    raw_data_path = os.path.join(project_dir, 'data', 'raw')
    processed_data_path = os.path.join(project_dir, 'data', 'processed')
    
    # Assurez-vous que les dossiers existent
    os.makedirs(raw_data_path, exist_ok=True)
    os.makedirs(processed_data_path, exist_ok=True)
    
    # Exemple: charger des données si elles existent
    try:
        data_file = os.path.join(raw_data_path, 'comments.csv')
        df = load_data(data_file)
        print(f"Données chargées avec succès: {len(df)} entrées")
        
        # Prétraiter les données
        df_processed = preprocess_data(df)
        
        # Diviser les données
        train, val, test = split_data(df_processed)
        
        # Sauvegarder les ensembles de données
        train.to_csv(os.path.join(processed_data_path, 'train.csv'), index=False)
        val.to_csv(os.path.join(processed_data_path, 'val.csv'), index=False)
        test.to_csv(os.path.join(processed_data_path, 'test.csv'), index=False)
        
        print("Prétraitement terminé et données sauvegardées")
        
    except FileNotFoundError:
        print(f"Fichier de données introuvable: {data_file}")
        print("Veuillez placer vos données dans le dossier data/raw")

if __name__ == "__main__":
    main()