import logging
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from src.features.text_processor import TextProcessor
from src.models.abstract_text_classification_model import TextClassificationModel
from src.utils.helper_functions import timer_decorator

class RandomForestTextClassifier(TextClassificationModel):
    def _init_(self, config_path: str):
        super()._init_(config_path)
        self.feature_engineer = TextProcessor(config_path)
        self.model = None  
        self.param_grid = self.config['model']['param_grid']
        self.vectorizer_params = self.config['model']['vectorizer_params'] 
        self.mlflow_config = self.config.get("mlflow", {})

    @timer_decorator
    def train(self, df: pd.DataFrame):
        # logging.info(f"Entraînement sur un lot de {len(df)} lignes")
        # texts, labels = df['text'].tolist(), df['label'].tolist()

        # # Pipeline : vectoriseur + classifieur
        # pipeline = Pipeline(steps=[
        #     ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        #     ('clf', RandomForestClassifier(random_state=42))
        # ])

        # grid_search = GridSearchCV(
        #     estimator=pipeline,
        #     param_grid=self.param_grid,
        #     cv=3, 
        #     n_jobs=-1,
        #     scoring="accuracy",
        #     verbose=2
        # )

        # grid_search.fit(texts, labels)
        # self.model = grid_search.best_estimator_
        # best_params = grid_search.best_params_
        # best_score = grid_search.best_score_

        # logging.info(f"Meilleurs paramètres : {best_params}")
        # logging.info(f"Meilleure précision (validation croisée) : {best_score:.4f}")

        # # MLflow logging
        # if self.mlflow_config.get("log_model", False):
        #     mlflow.set_tracking_uri(self.mlflow_config["tracking_uri"])
        #     experiment = mlflow.get_experiment_by_name(self.mlflow_config["experiment_name"])
        #     if experiment is None:
        #         experiment_id = mlflow.create_experiment(
        #             self.mlflow_config["experiment_name"],
        #             artifact_location=self.mlflow_config["artifact_location"]
        #         )
        #     else:
        #         experiment_id = experiment.experiment_id

        #     with mlflow.start_run(experiment_id=experiment_id):
        #         mlflow.log_params(best_params)
        #         mlflow.log_metric("cv_accuracy", best_score)
        #         mlflow.sklearn.log_model(self.model, "random_forest_v1")
        #         logging.info("Modèle loggé dans MLflow")
        pass
    def save_model(self, path):
        import pickle
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        logging.info(f"Modèle sauvegardé à {path}")