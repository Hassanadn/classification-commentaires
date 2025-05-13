#mport os 
# import sys 
# from fastapi import FastAPI, Form, Request
# from fastapi.responses import HTMLResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# from transformers import BertForSequenceClassification, BertTokenizer
# import numpy as np

# from src.features.feature_engineering import FeatureEngineer

# # from  import FeatureEngineer
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# app = FastAPI()

# # Load models
# # with open("models/random_forest_model.pkl", "rb") as f:
# #     rf_model = pickle.load(f)

# # Initialisation du moteur de templates
# templates = Jinja2Templates(directory="src/api")

# # Chargement du modèle BERT et du tokenizer
# bert_model = BertForSequenceClassification.from_pretrained("models/bert_model")
# bert_tokenizer = BertTokenizer.from_pretrained("models/bert_model")

# # Initialisation du FeatureEngineer
# feature_engineer = FeatureEngineer("config/config.yaml")

# # Route d'accueil (page HTML)
# @app.get("/", response_class=HTMLResponse)
# async def serve_home(request: Request):
#     return templates.TemplateResponse("admin.html", {"request": request})

# # Route de prédiction
# @app.post("/predict", response_class=HTMLResponse)
# async def predict(request: Request, text: str = Form(...)):
#     # Préparation des données avec le FeatureEngineer
#     bert_inputs = feature_engineer.transform_bert([text])
    
#     # Prédiction avec BERT
#     outputs = bert_model(**bert_inputs)
#     bert_pred = np.argmax(outputs.logits.detach().numpy(), axis=1)[0] + 1

#     # Interprétation du résultat
#     sentiment = "POSITIF" if bert_pred == 2 else "NEGATIVE" if bert_pred == 1 else "NEUTRE"

#     return templates.TemplateResponse("index.html", {
#         "request": request,
#         "prediction": sentiment,
#         "input_text": text
#     })
#  #Ajoutez ici la nouvelle route pour l'interface d'administration
# @app.get("/admin", response_class=HTMLResponse)
# async def serve_admin(request: Request):
#     return templates.TemplateResponse("admin.html", {"request": request})




import os
import sys
import warnings
import numpy as np


from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse

from fastapi import (
    FastAPI, Request, Form, Response, Cookie,
    HTTPException, status, Query, Path
)
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from transformers import BertForSequenceClassification, BertTokenizer
from prometheus_fastapi_instrumentator import Instrumentator

# Ignorer les warnings
warnings.filterwarnings('ignore')


# Ajouter le chemin parent pour importer les modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Importation du FeatureEngineer personnalisé
from src.features.feature_engineering import FeatureEngineer

# Initialisation de l'application FastAPI
app = FastAPI()

# Exposition des métriques Prometheus
# Instrumentator().instrument(app).expose(app)
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


# Initialisation du moteur de templates
templates = Jinja2Templates(directory="src/api")

# Chargement du modèle BERT et du tokenizer

# App FastAPI
app = FastAPI()

# Exposer les métriques Prometheus
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

# Configuration des templates et fichiers statiques
templates = Jinja2Templates(directory="src/api/templates")
app.mount("/utils", StaticFiles(directory="src/utils"), name="utils")

# Authentification
security = HTTPBasic()

# Chargement du modèle BERT

bert_model = BertForSequenceClassification.from_pretrained("models/bert_model")
bert_tokenizer = BertTokenizer.from_pretrained("models/bert_model")

# Initialisation du FeatureEngineer
feature_engineer = FeatureEngineer("config/config.yaml")


app.mount("/utils", StaticFiles(directory="src/utils"), name="utils")

# Route d'accueil (page HTML)

# Création des tables
metadata.create_all(engine)

# Connexion à la base de données
@app.on_event("startup")
async def startup():
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

# Vérification de session
def is_authenticated(session_token: Optional[str]) -> bool:
    return session_token == "authenticated"

# Page d'accueil

@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, text: str = Form(...)):
    # Préparation des données avec le FeatureEngineer
    bert_inputs = feature_engineer.transform_bert([text])
    
    # Prédiction avec BERT
    outputs = bert_model(**bert_inputs)
    bert_pred = np.argmax(outputs.logits.detach().numpy(), axis=1)[0] + 1


    # Interprétation du résultat
    sentiment = "POSITIF" if bert_pred == 2 else "NEGATIVE" if bert_pred == 1 else "NEUTRE"

    query = comments.insert().values(text=text, sentiment=sentiment)
    await database.execute(query)


    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": sentiment,
        "input_text": text
    })

# Page de connexion admin
@app.get("/loging", response_class=HTMLResponse)
async def serve_admin_login(request: Request):
    return templates.TemplateResponse("loging.html", {"request": request})

# Connexion admin
@app.post("/admin_login")
async def admin_login(
    request: Request,
    response: Response,
    username: str = Form(...),
    password: str = Form(...)
):
    if username == "mlops" and password == "mlops":
        response = RedirectResponse(url="/admin_stats", status_code=HTTP_303_SEE_OTHER)
        response.set_cookie(key="session_token", value="authenticated", httponly=True)
        return response
    else:
        return templates.TemplateResponse("loging.html", {
            "request": request,
            "success_message": "Nom d'utilisateur ou mot de passe incorrect."
        })

# Statistiques
@app.get("/admin_stats", response_class=HTMLResponse)
async def serve_admin_stats(request: Request, session_token: Optional[str] = Cookie(None)):
    if not is_authenticated(session_token):
        return RedirectResponse(url="/login", status_code=HTTP_303_SEE_OTHER)

    query = select(comments.c.sentiment)
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
@app.get("/dashboard_filtered", response_class=HTMLResponse)
async def dashboard_filtered(
    request: Request,
    type: str = Query(...),
    session_token: Optional[str] = Cookie(None)
):
    if not is_authenticated(session_token):
        return RedirectResponse(url="/login", status_code=HTTP_303_SEE_OTHER)

    type_map = {
        "positif": "POSITIF",
        "negatif": "NEGATIVE",
        "neutre": "NEUTRE",
        "all": None
    }
    sentiment_filter = type_map.get(type.lower())

    query = select(comments)
    if sentiment_filter:
        query = query.where(comments.c.sentiment == sentiment_filter)

    results = await database.fetch_all(query)

    return templates.TemplateResponse("filtered_dash.html", {
        "request": request,
        "comments": results,
        "filter_type": type
    })

# Suppression d’un commentaire
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
    else:
        return {"message": "Commentaire non trouvé."}

