import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
import sys

# Ajout du chemin src/ pour les imports relatifs
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def clean_and_standardize_data(input_path, output_path):
    logger.info(f"Chargement des données depuis {input_path}")
    try:
        data = pd.read_csv(input_path)
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données: {e}")
        raise
    
    # Affichage d'informations sur les données brutes
    logger.info(f"Dimensions des données brutes: {data.shape}")
    logger.info(f"Colonnes: {data.columns.tolist()}")
    logger.info(f"Valeurs manquantes par colonne: \n{data.isna().sum()}")
    
    # Vérification des colonnes
    required_columns = ["text", "label"]
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"La colonne {col} est manquante dans le dataset")
    
    # Nettoyage des données
    logger.info("Nettoyage des données...")
    
    # Suppression des lignes avec des valeurs manquantes dans les colonnes critiques
    initial_count = len(data)
    data = data.dropna(subset=["label"])
    dropped_count = initial_count - len(data)
    if dropped_count > 0:
        logger.warning(f"Suppression de {dropped_count} lignes avec des labels manquants")
    
    # Remplacer les valeurs NaN dans la colonne text par une chaîne vide
    missing_text = data["text"].isna().sum()
    if missing_text > 0:
        logger.warning(f"Remplacement de {missing_text} valeurs manquantes dans 'text' par une chaîne vide")
        data["text"] = data["text"].fillna("")
    
    # Vérifier et corriger les types de données
    # Standardisation des labels en strings
    data["label"] = data["label"].astype(str)
    logger.info(f"Types des labels après conversion: {data['label'].apply(type).unique()}")
    
    # S'assurer que 'text' est de type string
    data["text"] = data["text"].astype(str)
    logger.info(f"Types des textes après conversion: {data['text'].apply(type).unique()}")
    
    # Suppression des doublons
    initial_count = len(data)
    data = data.drop_duplicates(subset=["text"])
    dropped_count = initial_count - len(data)
    if dropped_count > 0:
        logger.info(f"Suppression de {dropped_count} lignes dupliquées")
    
    # Afficher quelques statistiques après nettoyage
    logger.info(f"Dimensions après nettoyage: {data.shape}")
    logger.info(f"Distribution des labels: \n{data['label'].value_counts()}")
    
    # Vérifier qu'il n'y a plus de valeurs NaN
    if data.isna().any().any():
        logger.warning(f"Il reste des valeurs manquantes après nettoyage: \n{data.isna().sum()}")
        # Remplir toutes les valeurs manquantes restantes
        data = data.fillna("")
    
    # Créer le répertoire de sortie si nécessaire
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Sauvegarde des données nettoyées
    data.to_csv(output_path, index=False)
    logger.info(f"Données nettoyées sauvegardées à {output_path}")
    
    return data

if __name__ == "__main__":
    # Chemins de fichiers
    input_path = os.path.join("data", "raw", "train.csv")
    output_path = os.path.join("data", "processed", "x_train_clean.csv")
    
    # Traitement des données
    clean_and_standardize_data(input_path, output_path)