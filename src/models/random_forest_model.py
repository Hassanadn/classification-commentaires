import os
import yaml
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime

class RandomForest:
    # def __init__(self, config_path: str):
    #     with open(config_path, 'r') as f:
    #         self.config = yaml.safe_load(f)
    #     self._load_data()
    #     self._build_pipeline()
    #     self.model = None

    # def _load_data(self):
    #     # Chargement des données traitées depuis le chemin spécifié dans la config
    #     data_path = "C:/Users/pc/Desktop/classification-commentaires/data/processed/processed_data.csv"

    #     # self.config["data"]["processed_path"]
    #     if not os.path.isabs(data_path):
    #         data_path = os.path.join("data", "processed", data_path)
    #     self.data = pd.read_csv(data_path).dropna(subset=['text'])
    #     self.X = self.data['text']
    #     self.y = self.data['label']
    #     self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
    #         self.X, self.y, 
    #         test_size=self.config["data"]["test_size"], 
    #         random_state=self.config["data"]["random_state"]
    #     )

    # def _build_pipeline(self):
    #     # Conversion du ngram_range en tuple
    #     ngram_range = tuple(self.config["model"]["ngram_range"])
    #     self.pipeline = Pipeline([
    #         ('tfidf', TfidfVectorizer(
    #             ngram_range=ngram_range,
    #             max_features=self.config["model"]["max_features"]
    #         )),
    #         ('clf', RandomForestClassifier(
    #             n_estimators=self.config["model"]["n_estimators"],
    #             max_depth=self.config["model"]["max_depth"],
    #             random_state=42
    #         ))
    #     ])
        
    #     # Conversion des listes en tuples pour ngram_range dans param_grid
    #     param_grid = self.config["model"]["param_grid"].copy()
    #     if 'tfidf__ngram_range' in param_grid:
    #         param_grid['tfidf__ngram_range'] = [tuple(ng) for ng in param_grid['tfidf__ngram_range']]
        
    #     self.param_grid = param_grid

    # def train(self):
    #     print("🔍 GridSearchCV en cours...")
    #     grid_search = GridSearchCV(
    #         self.pipeline,
    #         self.param_grid,
    #         cv=3,
    #         scoring='accuracy',
    #         n_jobs=-1,
    #         verbose=2
    #     )
    #     grid_search.fit(self.X_train, self.y_train)
    #     self.model = grid_search.best_estimator_
    #     print("✅ Meilleurs paramètres :", grid_search.best_params_)

    # def evaluate(self):
    #     y_pred = self.model.predict(self.X_test)
    #     print("\n📊 Rapport de classification :")
    #     print(classification_report(self.y_test, y_pred))
    #     acc = accuracy_score(self.y_test, y_pred)
    #     print("🎯 Accuracy:", acc)
    #     cm = confusion_matrix(self.y_test, y_pred)
    #     return acc, cm

    # def save_model(self):
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     model_filename = self.config["model"]["model_filename"].format(timestamp=timestamp)
    #     os.makedirs("models", exist_ok=True)
    #     path = os.path.join("models", model_filename)
    #     joblib.dump(self.model, path)
    #     print(f"💾 Modèle sauvegardé à {path}")

    # def log_with_mlflow(self):
    #     mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])
    #     mlflow.set_experiment(self.config["mlflow"]["experiment_name"])

    #     with mlflow.start_run():
    #         acc, cm = self.evaluate()

    #         if self.config["mlflow"]["log_metrics"]:
    #             mlflow.log_metric("accuracy", acc)

    #         if self.config["mlflow"]["log_model"]:
    #             mlflow.sklearn.log_model(self.model, "model")

    #         if self.config["mlflow"]["log_artifacts"]:
    #             # Génération de la matrice de confusion
    #             plt.figure(figsize=(6, 5))
    #             sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    #             plt.title("Confusion Matrix")
    #             plt.xlabel("Predicted")
    #             plt.ylabel("Actual")
    #             plot_path = "confusion_matrix.png"
    #             plt.savefig(plot_path)
    #             mlflow.log_artifact(plot_path)
    #             os.remove(plot_path)

    # def run(self):
    #     self.train()
    #     self.save_model()
    #     self.log_with_mlflow()
    pass