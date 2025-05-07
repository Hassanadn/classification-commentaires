"""
Fichier: src/api/predict.py
Description: Script pour faire des prédictions avec le modèle Random Forest
"""

import os
import pickle
import pandas as pd
import numpy as np
import logging
import yaml
import json

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """Charger les paramètres de configuration"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_model(model_path):
    """Charger un modèle entraîné"""
    logger.info(f"Chargement du modèle depuis {model_path}")
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {e}")
        raise

class RandomForestPredictor:
    """Classe pour effectuer des prédictions avec un modèle Random Forest"""
    
    def __init__(self, model_path=None):
        """Initialisation du prédicteur"""
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'models', 'random_forest_model.pkl'
            )
        
        self.model = load_model(model_path)
        logger.info("Prédicteur initialisé avec succès")
    
    def preprocess_input(self, input_data):
        """Prétraiter les données d'entrée pour la prédiction"""
        # Si les données sont déjà un DataFrame, on les utilise directement
        if isinstance(input_data, pd.DataFrame):
            df = input_data
        # Si les données sont un dictionnaire, on les convertit en DataFrame
        elif isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        # Si les données sont une chaîne JSON, on les parse
        elif isinstance(input_data, str):
            try:
                data_dict = json.loads(input_data)
                if isinstance(data_dict, list):
                    df = pd.DataFrame(data_dict)
                else:
                    df = pd.DataFrame([data_dict])
            except json.JSONDecodeError as e:
                logger.error(f"Erreur lors du parsing JSON: {e}")
                raise ValueError("Les données d'entrée ne sont pas un JSON valide")
        else:
            logger.error("Format d'entrée non pris en charge")
            raise ValueError("Format d'entrée non pris en charge. Utilisez un DataFrame, un dictionnaire ou une chaîne JSON")
        
        # Vérification que toutes les colonnes requises sont présentes
        missing_features = set(self.model.feature_names_in_) - set(df.columns)
        if missing_features:
            logger.error(f"Colonnes manquantes dans les données d'entrée: {missing_features}")
            raise ValueError(f"Colonnes manquantes dans les données d'entrée: {missing_features}")
        
        # Ne garder que les colonnes utilisées par le modèle
        df = df[self.model.feature_names_in_]
        
        return df
    
    def predict(self, input_data):
        """Faire des prédictions avec le modèle"""
        logger.info("Exécution des prédictions")
        
        # Prétraitement des données d'entrée
        try:
            X = self.preprocess_input(input_data)
        except Exception as e:
            logger.error(f"Erreur lors du prétraitement des données: {e}")
            raise
        
        # Prédictions de classe
        y_pred = self.model.predict(X)
        
        # Si le modèle est un classifieur, on récupère aussi les probabilités
        if hasattr(self.model, 'predict_proba'):
            y_prob = self.model.predict_proba(X)
            
            # Création d'un dictionnaire de résultats avec classe et probabilités
            results = []
            for i in range(len(y_pred)):
                class_probs = {str(cls): float(prob) for cls, prob in zip(self.model.classes_, y_prob[i])}
                results.append({
                    'prediction': int(y_pred[i]) if isinstance(y_pred[i], (np.integer, np.floating)) else y_pred[i],
                    'probabilities': class_probs
                })
        else:
            # Création d'un dictionnaire de résultats avec seulement la classe prédite
            results = []
            for i in range(len(y_pred)):
                results.append({
                    'prediction': int(y_pred[i]) if isinstance(y_pred[i], (np.integer, np.floating)) else y_pred[i]
                })
        
        # Si une seule entrée, on renvoie directement le résultat sans liste
        if len(results) == 1:
            return results[0]
        
        return results
    
    def predict_proba(self, input_data):
        """Faire des prédictions de probabilité avec le modèle"""
        logger.info("Exécution des prédictions de probabilité")
        
        # Vérification que le modèle peut prédire des probabilités
        if not hasattr(self.model, 'predict_proba'):
            logger.error("Le modèle ne supporte pas la prédiction de probabilités")
            raise AttributeError("Le modèle ne supporte pas la prédiction de probabilités")
        
        # Prétraitement des données d'entrée
        try:
            X = self.preprocess_input(input_data)
        except Exception as e:
            logger.error(f"Erreur lors du prétraitement des données: {e}")
            raise
        
        # Prédictions de probabilité
        y_prob = self.model.predict_proba(X)
        
        # Création d'un dictionnaire de résultats avec probabilités
        results = []
        for i in range(len(y_prob)):
            class_probs = {str(cls): float(prob) for cls, prob in zip(self.model.classes_, y_prob[i])}
            results.append(class_probs)
        
        # Si une seule entrée, on renvoie directement le résultat sans liste
        if len(results) == 1:
            return results[0]
        
        return results

def batch_predict(data_path, output_path=None):
    """Effectuer des prédictions par lot sur un fichier de données"""
    logger.info(f"Prédictions par lot sur le fichier {data_path}")
    
    # Chargement des données
    try:
        data = pd.read_csv(data_path)
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données: {e}")
        raise
    
    # Initialisation du prédicteur
    predictor = RandomForestPredictor()
    
    # Colonnes d'entrée (toutes les colonnes sauf la cible si elle existe)
    config = load_config()
    target_column = config.get('model', {}).get('random_forest', {}).get('target_column', 'target')
    
    # Garder une copie des données complètes
    original_data = data.copy()
    
    # Supprimer la colonne cible si elle existe
    if target_column in data.columns:
        X = data.drop(target_column, axis=1)