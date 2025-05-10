import os
import sys
from fastapi import FastAPI, Form, Request, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from transformers import BertForSequenceClassification, BertTokenizer
import numpy as np
from starlette.status import HTTP_303_SEE_OTHER
from src.features.feature_engineering import FeatureEngineer
from src.api.database import database, engine, metadata
from src.api.models import comments
from sqlalchemy import select

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

app = FastAPI()

# Initialisation du moteur de templates
templates = Jinja2Templates(directory="src/api")

# Chargement du modèle BERT et du tokenizer
bert_model = BertForSequenceClassification.from_pretrained("models/bert_model")
bert_tokenizer = BertTokenizer.from_pretrained("models/bert_model")

# Initialisation du FeatureEngineer
feature_engineer = FeatureEngineer("config/config.yaml")

# Sécurité basique HTTP
security = HTTPBasic()

# Création des tables
metadata.create_all(engine)

@app.on_event("startup")
async def startup():
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

# Route d'accueil (page HTML)
@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Route de prédiction
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, text: str = Form(...)):
    bert_inputs = feature_engineer.transform_bert([text])
    outputs = bert_model(**bert_inputs)
    bert_pred = np.argmax(outputs.logits.detach().numpy(), axis=1)[0] + 1
    sentiment = "POSITIF" if bert_pred == 2 else "NEGATIVE" if bert_pred == 1 else "NEUTRE"

    # Enregistrer le commentaire dans la base
    query = comments.insert().values(text=text, sentiment=sentiment)
    await database.execute(query)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": sentiment,
        "input_text": text
    })

# Route GET pour la page admin (authentification)
@app.get("/admin", response_class=HTMLResponse)
async def serve_admin(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})

# Route POST pour authentification admin simple
@app.post("/admin_login")
async def admin_login(request: Request, username: str = Form(...), password: str = Form(...)):
    # Identifiants codés en dur (à remplacer par une vraie base)
    if username == "admin" and password == "admin123":
        response = RedirectResponse(url="/admin_stats", status_code=HTTP_303_SEE_OTHER)
        # Ici vous pouvez ajouter un cookie ou token pour session
        return response
    else:
        return templates.TemplateResponse("admin.html", {
            "request": request,
            "success_message": "Nom d'utilisateur ou mot de passe incorrect."
        })

# Route GET pour la page admin_stats (affichage des statistiques)
@app.get("/admin_stats", response_class=HTMLResponse)
async def serve_admin_stats(request: Request):
    # Récupérer les statistiques simples (ex: nombre total de commentaires)
    #query = select([comments.c.sentiment, comments.c.id])
    query = select(comments.c.sentiment, comments.c.id)

    results = await database.fetch_all(query)

    # Calculer le nombre de commentaires par sentiment
    stats = {"POSITIF": 0, "NEGATIVE": 0, "NEUTRE": 0}
    for row in results:
        sentiment = row["sentiment"]
        if sentiment in stats:
            stats[sentiment] += 1

    return templates.TemplateResponse("admin_stats.html", {
        "request": request,
        "stats": stats
    })
from fastapi import Query

@app.get("/dashboard_filtered", response_class=HTMLResponse)
async def dashboard_filtered(request: Request, type: str = Query(...)):
    # Normaliser la valeur du type
    type_map = {
        "positif": "POSITIF",
        "negatif": "NEGATIVE",
        "neutre": "NEUTRE",
        "all": None
    }
    sentiment_filter = type_map.get(type.lower(), None)

    if sentiment_filter:
        query = select(comments).where(comments.c.sentiment == sentiment_filter)
    else:
        query = select(comments)

    results = await database.fetch_all(query)

    return templates.TemplateResponse("filtered_dash.html", {
        "request": request,
        "comments": results,
        "filter_type": type
    })

from fastapi import Path

@app.delete("/comments/{comment_id}")
async def delete_comment(comment_id: int = Path(..., description="ID du commentaire à supprimer")):
    query = comments.delete().where(comments.c.id == comment_id)
    result = await database.execute(query)
    if result:
        return {"message": "Commentaire supprimé avec succès."}
    else:
        return {"message": "Commentaire non trouvé."}
