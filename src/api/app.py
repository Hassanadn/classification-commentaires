"""
API FastAPI pour la classification des commentaires
"""
from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel

# Initialisation de l'API
app = FastAPI(
    title="API de Classification de Commentaires",
    description="API pour classifier les commentaires en positifs ou négatifs",
    version="0.1.0"
)

class CommentRequest(BaseModel):
    text: str

class CommentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de classification de commentaires"}

@app.post("/classify", response_model=CommentResponse)
def classify_comment(comment: CommentRequest):
    """
    Classifie un commentaire comme positif ou négatif
    """
    # Pour le moment, retourne un résultat fictif
    # Sera remplacé par l'appel au modèle réel
    return {
        "text": comment.text,
        "sentiment": "positif" if len(comment.text) % 2 == 0 else "négatif",
        "confidence": 0.85
    }