# src/api/load_model.py
import sys
import logging
from pathlib import Path
import mlflow.sklearn

# Ajout du chemin src/ pour les imports
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.mlflow_utils import MLflowManager
from src.features.text_processor import TextProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self, run_id=None, model_name="RandomForest_TextClassifier", stage="Production"):
        self.mlflow_manager = MLflowManager()
        self.run_id = run_id
        self.model_name = model_name
        self.stage = stage
        self.model = None
        self.text_processor = None
    
    def load_from_run(self, run_id=None):
        """Charge un modèle à partir d'un run spécifique"""
        rid = run_id or self.run_id
        if not rid:
            logger.error("Aucun run_id spécifié")
            return False
        
        try:
            model_uri = f"runs:/{rid}/model"
            self.model = self.mlflow_manager.load_model(model_uri)
            logger.info(f"Modèle chargé depuis le run {rid}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle depuis le run {rid}: {e}")
            return False
    
    def load_from_registry(self, version=None):
        """Charge un modèle à partir du registre de modèles"""
        try:
            if version:
                model_uri = f"models:/{self.model_name}/{version}"
            else:
                model_uri = f"models:/{self.model_name}/{self.stage}"
            
            self.model = self.mlflow_manager.load_model(model_uri)
            logger.info(f"Modèle '{self.model_name}' chargé depuis le registre (stage: {self.stage})")
            return True
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle depuis le registre: {e}")
            return False
    
    def load_text_processor(self, processor_path):
        """Charge le text processor"""
        try:
            self.text_processor = TextProcessor()
            self.text_processor.load(processor_path)
            logger.info(f"Text processor chargé depuis {processor_path}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors du chargement du text processor: {e}")
            return False
    
    def predict(self, texts):
        """Fait une prédiction avec le modèle chargé"""
        if self.model is None:
            logger.error("Aucun modèle n'a été chargé")
            return None
        
        try:
            if self.text_processor:
                # Si le text processor est disponible
                X = self.text_processor.transform(texts)
                return self.model.predict(X)
            else:
                # Utilisation directe du modèle MLflow (qui peut inclure le pipeline complet)
                return self.model.predict(texts)
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {e}")
            return None

def get_best_model(metric="metrics.accuracy", ascending=False):
    """Récupère le meilleur modèle selon une métrique"""
    mlflow_manager = MLflowManager()
    mlflow_manager.setup_experiment()
    
    # Recherche des runs
    runs = mlflow_manager.search_runs()
    
    if runs.empty:
        logger.error("Aucun run trouvé")
        return None
    
    # Tri selon la métrique
    if metric in runs.columns:
        best_run = runs.sort_values(metric, ascending=ascending).iloc[0]
        run_id = best_run["run_id"]
        
        logger.info(f"Meilleur modèle trouvé: {run_id} avec {metric}={best_run[metric]}")
        return run_id
    else:
        logger.error(f"Métrique {metric} non trouvée dans les runs")
        return None

if __name__ == "__main__":
    # Exemple d'utilisation
    best_run_id = get_best_model()
    if best_run_id:
        loader = ModelLoader(run_id=best_run_id)
        if loader.load_from_run():
            prediction = loader.predict(["Ceci est un exemple de texte à classifier"])
            print(f"Prédiction: {prediction}")