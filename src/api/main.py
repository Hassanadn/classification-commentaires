

# from fastapi import FastAPI
# import pickle
# from transformers import BertForSequenceClassification, BertTokenizer
# from src.features.feature_engineering import FeatureEngineer
# import numpy as np
# from fastapi.staticfiles import StaticFiles

# app = FastAPI()

# # Charger le modèle Random Forest
# with open("models/random_forest_v1.pkl", "rb") as f:
#     rf_model = pickle.load(f)

# # Charger le modèle BERT
# bert_model = BertForSequenceClassification.from_pretrained("models/bert_v1")
# bert_tokenizer = BertTokenizer.from_pretrained("models/bert_v1")

# # Initialiser FeatureEngineer (supposé défini dans votre projet)
# feature_engineer = FeatureEngineer("config/config.yaml")

# # app.mount("/", StaticFiles(directory="src/api", html=True), name="static")
# @app.post("/predict")
# async def predict(text: str):
#     # Prétraitement pour Random Forest
#     # rf_features = feature_engineer.transform_tfidf([text])
#     # rf_pred = rf_model.predict(rf_features)[0]
    
#     # Prétraitement pour BERT
#     bert_inputs = feature_engineer.transform_bert([text])
#     outputs = bert_model(**bert_inputs)
#     bert_pred = np.argmax(outputs.logits.detach().numpy(), axis=1)[0] + 1  # Ajuster labels (0->1, 1->2)
    
#     return {"random_forest_prediction": int(bert_pred), "bert_prediction": int(bert_pred)}  # Remplacez None par bert_pred si le modèle BERT est chargé

# # uvicorn src.api.main:app --host 127.0.0.1 --port 8002

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
with open("models/random_forest_v1.pkl", "rb") as f:
    rf_model = pickle.load(f)

bert_model = BertForSequenceClassification.from_pretrained("models/bert_v1")
bert_tokenizer = BertTokenizer.from_pretrained("models/bert_v1")

feature_engineer = FeatureEngineer("config/config.yaml")

# Set up templates and static files
app.mount("/static", StaticFiles(directory="src/api"), name="static")
templates = Jinja2Templates(directory="src/api")

@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, text: str = Form(...)):
    # Prepare input for BERT
    bert_inputs = feature_engineer.transform_bert([text])
    outputs = bert_model(**bert_inputs)
    bert_pred = np.argmax(outputs.logits.detach().numpy(), axis=1)[0] + 1
    
    sentiment = "POSITIF" if bert_pred == 2 else "NÉGATIF"
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": sentiment,
        "input_text": text
    })


