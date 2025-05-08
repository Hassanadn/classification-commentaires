import os
import sys
import yaml
import wandb
from pathlib import Path
import logging
from typing import Dict, Any, Optional, List, Union

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WandbLogger:
    """
    Classe pour gérer l'intégration avec Weights & Biases
    """
    def __init__(self, config_path: str = None):
        """
        Initialise la connexion avec Weights & Biases
        
        Args:
            config_path: Chemin vers le fichier de configuration wandb
        """
        self.run = None
        self.config_path = config_path or os.path.join('config', 'wandb_config.yaml')
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Charge la configuration depuis le fichier YAML"""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
                
            # Remplacer les variables d'environnement
            wandb_config = config.get('wandb', {})
            for key, value in wandb_config.items():
                if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                    env_var = value[2:-1]
                    wandb_config[key] = os.environ.get(env_var, '')
                    
            return config
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration wandb: {e}")
            return {'wandb': {}}
            
    def setup(self) -> wandb.Run:
        """
        Configure et initialise wandb
        
        Returns:
            Une instance de wandb.Run
        """
        wandb_config = self.config.get('wandb', {})
        
        # Vérifier si la clé API est définie
        api_key = wandb_config.get('api_key')
        if not api_key:
            logger.warning("WANDB_API_KEY n'est pas définie. Utilisation de wandb en mode hors ligne.")
            os.environ['WANDB_MODE'] = 'offline'
        else:
            wandb.login(key=api_key)
            
        # Initialiser une nouvelle exécution
        self.run = wandb.init(
            project=wandb_config.get('project', 'default_project'),
            entity=wandb_config.get('entity'),
            name=wandb_config.get('experiment_name'),
            tags=wandb_config.get('tags', []),
            config=wandb_config,
            dir=os.environ.get('WANDB_DIR', './models/wandb')
        )
        
        # Activer le tracking du code si demandé
        if wandb_config.get('log_code', False):
            wandb.run.log_code(".")
            
        return self.run
        
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Enregistre des métriques dans wandb
        
        Args:
            metrics: Dictionnaire de métriques à enregistrer
            step: Étape/epoch actuelle (optionnel)
        """
        if self.run is None:
            self.setup()
            
        wandb.log(metrics, step=step)
        
    def log_model(self, model_path: str, model_name: str, metadata: Dict[str, Any] = None) -> None:
        """
        Enregistre un modèle entraîné dans wandb
        
        Args:
            model_path: Chemin vers le fichier du modèle sauvegardé
            model_name: Nom du modèle
            metadata: Métadonnées supplémentaires pour le modèle
        """
        if self.run is None:
            self.setup()
            
        artifact = wandb.Artifact(
            name=model_name,
            type='model',
            metadata=metadata or {}
        )
        artifact.add_file(model_path)
        self.run.log_artifact(artifact)
        
    def log_dataset(self, dataset_path: str, dataset_name: str, metadata: Dict[str, Any] = None) -> None:
        """
        Enregistre un jeu de données dans wandb
        
        Args:
            dataset_path: Chemin vers le fichier du dataset
            dataset_name: Nom du dataset
            metadata: Métadonnées supplémentaires pour le dataset
        """
        if self.run is None:
            self.setup()
            
        artifact = wandb.Artifact(
            name=dataset_name,
            type='dataset',
            metadata=metadata or {}
        )
        
        if os.path.isdir(dataset_path):
            artifact.add_dir(dataset_path)
        else:
            artifact.add_file(dataset_path)
            
        self.run.log_artifact(artifact)
        
    def finish(self) -> None:
        """Termine la session wandb"""
        if self.run:
            wandb.finish()