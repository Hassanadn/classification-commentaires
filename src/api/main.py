import os
import sys
import warnings
from typing import Optional

from fastapi import (
    FastAPI, Request, Form, Response, Cookie, Depends, HTTPException, status, Query, Path
)
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic
from prometheus_fastapi_instrumentator import Instrumentator
from transformers import BertForSequenceClassification, BertTokenizer
from sqlalchemy import select
import numpy as np

from src.api.database import database, engine, metadata
from src.api.models import comments
from src.features.feature_engineering import FeatureEngineer
from starlette.status import HTTP_303_SEE_OTHER

# Ignore warnings
warnings.filterwarnings('ignore')

# Initialisation de l'app FastAPI
app = FastAPI()

# Prometheus metrics
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

# Templates & Static Files
templates = Jinja2Templates(directory="src/api/templates")
app.mount("/utils", StaticFiles(directory="src/utils"), name="utils")

# Authentification de base
security = HTTPBasic()

# Modèle BERT
bert_model = BertForSequenceClassification.from_pretrained("models/bert_model")
bert_tokenizer = BertTokenizer.from_pretrained("models/bert_model")

# Feature engineering
feature_engineer = FeatureEngineer("config/config.yaml")

# Création de la base
metadata.create_all(engine)

# Connexion à la base
@app.on_event("startup")
async def startup():
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

# Vérification de session
def is_authenticated(session_token: Optional[str]) -> bool:
    return session_token == "authenticated"

def get_current_user(session_token: Optional[str] = Cookie(None)):
    if not is_authenticated(session_token):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Non authentifié")

# Page d'accueil
@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Prédiction de sentiment
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, text: str = Form(...)):
    bert_inputs = feature_engineer.transform_bert([text])
    outputs = bert_model(**bert_inputs)
    bert_pred = np.argmax(outputs.logits.detach().numpy(), axis=1)[0] + 1
    sentiment = "POSITIF" if bert_pred == 2 else "NEGATIVE" if bert_pred == 1 else "NEUTRE"

    # Insertion en base
    query = comments.insert().values(text=text, sentiment=sentiment)
    await database.execute(query)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": sentiment,
        "input_text": text
    })

# Page admin
@app.get("/loging", response_class=HTMLResponse)
async def serve_admin(request: Request):
    return templates.TemplateResponse("loging.html", {"request": request})

# Connexion admin
@app.post("/loging")
async def admin_login(
    request: Request,
    response: Response,
    username: str = Form(...),
    password: str = Form(...)
):
    if username == "mlops" and password == "mlops":
        response = RedirectResponse(url="/dashboard", status_code=HTTP_303_SEE_OTHER)
        response.set_cookie(key="session_token", value="authenticated", httponly=True)
        return response
    else:
        return templates.TemplateResponse("loging.html", {
            "request": request,
            "success_message": "Nom d'utilisateur ou mot de passe incorrect."
        })

# Statistiques admin
@app.get("/dashboard", response_class=HTMLResponse)
async def serve_admin_stats(request: Request, session_token: Optional[str] = Cookie(None)):
    if not is_authenticated(session_token):
        return RedirectResponse(url="/loging", status_code=HTTP_303_SEE_OTHER)

    query = select(comments.c.sentiment, comments.c.id)
    results = await database.fetch_all(query)

    stats = {"POSITIF": 0, "NEGATIVE": 0, "NEUTRE": 0}
    for row in results:
        sentiment = row["sentiment"]
        if sentiment in stats:
            stats[sentiment] += 1

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "stats": stats
    })

# Filtrage des commentaires
@app.get("/filtered_dash", response_class=HTMLResponse)
async def dashboard_filtered(
    request: Request,
    type: str = Query(...),
    session_token: Optional[str] = Cookie(None)
):
    if not is_authenticated(session_token):
        return RedirectResponse(url="/loging", status_code=HTTP_303_SEE_OTHER)

    type_map = {
        "positif": "POSITIF",
        "negatif": "NEGATIVE",
        "neutre": "NEUTRE",
        "all": None
    }
    sentiment_filter = type_map.get(type.lower())

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

# Suppression d'un commentaire
@app.delete("/comments/{comment_id}")
async def delete_comment(
    comment_id: int = Path(..., description="ID du commentaire à supprimer"),
    session_token: Optional[str] = Cookie(None)
):
    if not is_authenticated(session_token):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Non authentifié")

    query = comments.delete().where(comments.c.id == comment_id)
    result = await database.execute(query)

    if result:
        return {"message": "Commentaire supprimé avec succès."}
    return {"message": "Commentaire non trouvé."}