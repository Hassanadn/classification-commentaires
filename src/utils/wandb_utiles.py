import os
import yaml
from typing import Dict, Any, Optional, List
import wandb
from src.monitoring.wandb_logger import WandbLogger

def load_wandb_artifact(artifact_name: str, artifact_type: str, version: str = "latest") -> str:
    """
    Télécharge un artifact wandb et retourne son chemin local
    
    Args:
        artifact_name: Nom de l'artifact
        artifact_type: Type de l'artifact (model, dataset, etc.)
        version: Version de l'artifact à télécharger (default: "latest")
        
    Returns:
        Le chemin local de l'artifact téléchargé
    """
    artifact = wandb.use_artifact(f"{artifact_name}:{version}", type=artifact_type)
    return artifact.download()
    
def compare_runs(runs: List[str], metrics: List[str]) -> Dict[str, Any]:
    """
    Compare plusieurs exécutions wandb basées sur des métriques spécifiques
    
    Args:
        runs: Liste des IDs d'exécution à comparer
        metrics: Liste des métriques à comparer
        
    Returns:
        Un dictionnaire contenant les résultats de comparaison
    """
    api = wandb.Api()
    results = {}
    
    for run_id in runs:
        run = api.run(run_id)
        run_metrics = {}
        
        for metric in metrics:
            if metric in run.summary:
                run_metrics[metric] = run.summary[metric]
                
        results[run.name] = run_metrics
        
    return results

def sweep_configuration(config_dict: Dict[str, Any]) -> str:
    """
    Configure et démarre un sweep wandb pour l'optimisation d'hyperparamètres
    
    Args:
        config_dict: Configuration du sweep
        
    Returns:
        L'ID du sweep créé
    """
    sweep_id = wandb.sweep(config_dict)
    return sweep_id