import sys
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.db import  init_db
from src.api.routes import home, auth_routes, dashboard, comments
from api.default_user import create_default_user  # nouveau fichier
# Ajouter le chemin racine au sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))



app = FastAPI()

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À adapter en prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



app = FastAPI()

@app.on_event("startup")
async def startup():
    init_db()
    create_default_user()

# Routes
app.include_router(home.router)
app.include_router(auth_routes.router)
app.include_router(dashboard.router)
app.include_router(comments.router)
