# src/api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import sys
from pathlib import Path
import logging

# Ajout du chemin src/ pour les imports
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.api.load_model import ModelLoader, get_best_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Text Classification API")

# Modèle global pour l'API
model_loader = None

class PredictionRequest(BaseModel):
    texts: List[str]

class PredictionResponse(BaseModel):
    predictions: List[str]
    confidence: Dict[str, float] = None

@app.on_event("startup")
async def startup_event():
    """Charge le modèle au démarrage de l'API"""
    global model_loader
    
    # Récupération du meilleur modèle
    best_run_id = get_best_model()
    if not best_run_id:
        logger.warning("Aucun modèle trouvé. Utilisation du modèle par défaut.")
        # Fallback sur un modèle par défaut
        model_loader = ModelLoader()
        model_loader.load_from_registry(stage="Production")
    else:
        # Chargement du meilleur modèle
        model_loader = ModelLoader(run_id=best_run_id)
        success = model_loader.load_from_run()
        if not success:
            logger.error("Échec du chargement du modèle.")
            raise Exception("Échec du chargement du modèle.")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Point d'entrée pour les prédictions"""
    global model_loader
    
    if model_loader is None or model_loader.model is None:
        raise HTTPException(status_code=500, detail="Modèle non disponible")
    
    try:
        predictions = model_loader.predict(request.texts)
        
        # Extraction des probabilités si disponibles
        confidence = {}
        if hasattr(model_loader.model, "predict_proba"):
            probas = model_loader.model.predict_proba(request.texts)
            for i, proba in enumerate(probas):
                max_proba_idx = proba.argmax()
                confidence[f"text_{i}"] = float(proba[max_proba_idx])
        
        return {
            "predictions": predictions.tolist() if hasattr(predictions, "tolist") else predictions,
            "confidence": confidence
        }
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Retourne des informations sur le modèle chargé"""
    global model_loader
    
    if model_loader is None or model_loader.model is None:
        raise HTTPException(status_code=500, detail="Modèle non disponible")
    
    return {
        "model_type": type(model_loader.model).__name__,
        "run_id": model_loader.run_id,
        "classes": model_loader.model.classes_.tolist() if hasattr(model_loader.model, "classes_") else None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)