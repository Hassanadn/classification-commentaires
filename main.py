from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import BertForSequenceClassification, BertTokenizer
import pickle
import numpy as np
from src.features.feature_engineering import FeatureEngineer

app = FastAPI()

# Load models
with open("models/random_forest_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

# Initialisation du moteur de templates
templates = Jinja2Templates(directory="src/api")

# Chargement du modèle BERT et du tokenizer
bert_model = BertForSequenceClassification.from_pretrained("models/bert_model")
bert_tokenizer = BertTokenizer.from_pretrained("models/bert_model")

# Initialisation du FeatureEngineer
feature_engineer = FeatureEngineer("config/config.yaml")

# Route d'accueil (page HTML)
@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Route de prédiction
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, text: str = Form(...)):
    # Préparation des données avec le FeatureEngineer
    bert_inputs = feature_engineer.transform_bert([text])
    
    # Prédiction avec BERT
    outputs = bert_model(**bert_inputs)
    bert_pred = np.argmax(outputs.logits.detach().numpy(), axis=1)[0] + 1

    # Interprétation du résultat
    sentiment = "POSITIF" if bert_pred == 2 else "NEGATIVE" if bert_pred == 1 else "NEUTRE"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": sentiment,
        "input_text": text
    })
