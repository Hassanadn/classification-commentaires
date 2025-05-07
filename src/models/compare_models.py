# src/models/compare_models.py
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

# Ajout du chemin src/ pour les imports
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.mlflow_utils import MLflowManager

def compare_runs(metric_names=None, n_runs=10):
    """Compare les N derniers runs selon les métriques spécifiées"""
    if metric_names is None:
        metric_names = ["accuracy", "f1_score"]
    
    mlflow_manager = MLflowManager()
    mlflow_manager.setup_experiment()
    
    # Récupération des runs
    runs = mlflow_manager.search_runs(max_results=n_runs)
    
    if runs.empty:
        print("Aucun run trouvé pour cette expérience.")
        return
    
    # Extraction des paramètres et métriques intéressants
    columns = ["run_id", "start_time"]
    param_columns = ["n_estimators", "max_depth", "min_samples_split", "class_weight"]
    
    # Création d'un DataFrame pour l'analyse
    results = pd.DataFrame()
    results["run_id"] = runs["run_id"]
    results["start_time"] = pd.to_datetime(runs["start_time"])
    
    # Ajout des paramètres
    for param in param_columns:
        param_key = f"params.{param}"
        if param_key in runs.columns:
            results[param] = runs[param_key]
    
    # Ajout des métriques
    for metric in metric_names:
        metric_key = f"metrics.{metric}"
        if metric_key in runs.columns:
            results[metric] = runs[metric_key].astype(float)
    
    # Affichage du tableau comparatif
    print("\nComparaison des modèles:")
    print(results.sort_values("start_time", ascending=False))
    
    # Création d'un graphique pour comparer les métriques
    plt.figure(figsize=(12, 8))
    
    for i, metric in enumerate(metric_names):
        if metric in results.columns:
            plt.subplot(len(metric_names), 1, i+1)
            
            # Tri par date pour voir l'évolution
            sorted_results = results.sort_values("start_time")
            
            # Plot
            sns.lineplot(data=sorted_results, x="start_time", y=metric, marker="o", markersize=8)
            plt.title(f"Évolution de {metric}")
            plt.ylabel(metric)
            plt.grid(True)
            plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig("model_comparison.png")
    plt.close()
    
    return results

if __name__ == "__main__":
    compare_runs()