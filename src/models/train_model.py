"""
Scripts pour entraîner les modèles
"""
import os
import pickle
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

def train_simple_model(X_train, y_train):
    """
    Entraîne un modèle simple (TF-IDF + RandomForest)
    """
    # Créer un pipeline avec TF-IDF et RandomForest
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Entraîner le pipeline
    pipeline.fit(X_train, y_train)
    
    return pipeline

def evaluate_model(model, X_test, y_test):
    """
    Évalue les performances du modèle
    """
    y_pred = model.predict(X_test)
    
    # Calculer les métriques
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'classification_report': report
    }

def main():
    """
    Point d'entrée principal pour l'entraînement du modèle
    """
    # Définir les chemins
    project_dir = Path(__file__).resolve().parents[2]
    processed_data_path = os.path.join(project_dir, 'data', 'processed')
    models_path = os.path.join(project_dir, 'models')
    
    # Assurez-vous que le dossier models existe
    os.makedirs(models_path, exist_ok=True)
    
    try:
        # Charger les données d'entraînement et de test
        train_data = pd.read_csv(os.path.join(processed_data_path, 'train.csv'))
        test_data = pd.read_csv(os.path.join(processed_data_path, 'test.csv'))
        
        # Séparer les features et les labels
        X_train = train_data['text_clean']
        y_train = train_data['sentiment']
        X_test = test_data['text_clean']
        y_test = test_data['sentiment']
        
        print("Entraînement du modèle...")
        model = train_simple_model(X_train, y_train)
        
        print("Évaluation du modèle...")
        metrics = evaluate_model(model, X_test, y_test)
        print(f"Précision: {metrics['accuracy']:.4f}")
        print(metrics['classification_report'])
        
        # Sauvegarder le modèle
        model_path = os.path.join(models_path, 'sentiment_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Modèle sauvegardé: {model_path}")
        
    except FileNotFoundError as e:
        print(f"Erreur: {e}")
        print("Assurez-vous d'avoir exécuté le script de prétraitement des données d'abord")

if __name__ == "__main__":
    main()