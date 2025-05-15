from datetime import datetime
import warnings
from typing import Optional
from sqlalchemy import delete
from fastapi import HTTPException
from fastapi import (
    APIRouter, Request, Form, Cookie, Depends, HTTPException,
    status, Query, Path
)
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from prometheus_fastapi_instrumentator import Instrumentator
import numpy as np

from requests import Session
from transformers import BertForSequenceClassification, BertTokenizer
from sqlalchemy import select

from src.api import database
from src.api.db import Base, Comment
from src.api.schemas import comment
from src.api.services.auth import AuthService
from src.api.database import engine
from src.features.feature_engineering import FeatureEngineer
from starlette.status import HTTP_303_SEE_OTHER

# Initialisation base de données
Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)

# Ignore warnings
warnings.filterwarnings('ignore')

# Initialisation FastAPI
router = APIRouter()
templates = Jinja2Templates(directory="src/api/templates")

# Chargement du modèle BERT
bert_model = BertForSequenceClassification.from_pretrained("models/bert_model")
bert_tokenizer = BertTokenizer.from_pretrained("models/bert_model")

# Initialisation du feature engineer
feature_engineer = FeatureEngineer("config/config.yaml")


@router.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    text: str = Form(...),
    db: database.Session = Depends(database.get_db)
):
    # Préparation des données pour BERT
    bert_inputs = feature_engineer.transform_bert([text])
    outputs = bert_model(**bert_inputs)
    bert_pred = np.argmax(outputs.logits.detach().numpy(), axis=1)[0] + 1
    sentiment = "POSITIF" if bert_pred == 2 else "NEGATIVE" if bert_pred == 1 else "NEUTRE"

    # Enregistrement en base
    new_comment = Comment(
        content=text,
        sentiment=sentiment,
        author="anonymous",
        created_at=datetime.utcnow()
    )
    db.add(new_comment)
    db.commit()
    db.refresh(new_comment)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": sentiment,
        "input_text": text
    })

@router.get("/filtered_dash", response_class=HTMLResponse)
async def dashboard_filtered(
    request: Request,
    type: str = Query("all"),
    session_token: Optional[str] = Cookie(None),
    db: Session = Depends(database.get_db)
):
    if not AuthService.is_authenticated(session_token):
        return RedirectResponse(url="/login", status_code=HTTP_303_SEE_OTHER)

    type_map = {
        "positif": "POSITIF",
        "negatif": "NEGATIVE",
        "all": None
    }
    sentiment_filter = type_map.get(type.lower())

    if sentiment_filter:
        query = select(Comment).where(Comment.sentiment == sentiment_filter)
    else:
        query = select(Comment)

    results = db.execute(query).scalars().all()

    return templates.TemplateResponse("filtered_dash.html", {
        "request": request,
        "comments": results,
        "filter_type": type
    })



@router.delete("/comments/{comment_id}")
async def delete_comment(
    comment_id: int,
    db: Session = Depends(database.get_db)
):
    query = delete(Comment).where(Comment.id == comment_id)
    result = db.execute(query)
    db.commit()

    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Commentaire non trouvé")

    return {"message": "Commentaire supprimé avec succès"}
