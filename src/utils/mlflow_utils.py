# src/utils/mlflow_utils.py
import os
import yaml
import mlflow
import mlflow.sklearn
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

class MLflowManager:
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        project_root = Path(__file__).resolve().parents[2]
        default_cfg = project_root / "config" / "mlflow_config.yaml"
        cfg_path = Path(config_path) if config_path else default_cfg

        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                self.mlflow_config = config.get("mlflow", {})
        except Exception as e:
            logger.warning(f"Impossible de charger la configuration MLflow: {e}. Utilisation des valeurs par défaut.")
            self.mlflow_config = {}

        # Configuration du tracking
        self.tracking_uri = self.mlflow_config.get("tracking_uri", "file:./mlruns")
        self.experiment_name = self.mlflow_config.get("experiment_name", "default")
        self.model_registry = self.mlflow_config.get("model_registry", "models")

        # Initialisation de MLflow
        mlflow.set_tracking_uri(self.tracking_uri)
        
    def setup_experiment(self) -> str:
        """Configure l'expérience MLflow et retourne l'ID d'expérience"""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                name=self.experiment_name,
                artifact_location=self.mlflow_config.get("artifacts_uri")
            )
            logger.info(f"Nouvelle expérience créée: {self.experiment_name} (ID: {experiment_id})")
        else:
            experiment_id = experiment.experiment_id
            logger.info(f"Utilisation de l'expérience existante: {self.experiment_name} (ID: {experiment_id})")
        
        mlflow.set_experiment(self.experiment_name)
        return experiment_id

    def log_model_params(self, params: Dict[str, Any]) -> None:
        """Log des paramètres du modèle"""
        for key, value in params.items():
            mlflow.log_param(key, value)
    
    def log_model_metrics(self, metrics: Dict[str, float]) -> None:
        """Log des métriques du modèle"""
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)
    
    def log_model(self, model, artifact_path="model", registered_model_name=None, **kwargs):
        """Log du modèle dans MLflow"""
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name,
            **kwargs
        )
    
    def log_artifact(self, local_path):
        """Log d'un artifact"""
        mlflow.log_artifact(local_path)
    
    def get_run_info(self, run_id):
        """Récupère les informations d'un run"""
        return mlflow.get_run(run_id)
    
    def search_runs(self, **kwargs):
        """Recherche des runs selon certains critères"""
        return mlflow.search_runs(**kwargs)
    
    def load_model(self, model_uri):
        """Charge un modèle depuis MLflow"""
        return mlflow.sklearn.load_model(model_uri)