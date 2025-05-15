import sys
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator
from src.api.db import init_db
from src.api.routes import home, auth_routes, dashboard, comments
from src.api.default_user import create_default_user

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

app = FastAPI()

from fastapi.staticfiles import StaticFiles

app.mount("/static-utils", StaticFiles(directory="src/utils"), name="static-utils")

# Middleware CORS (conservé comme dans votre code)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Événement de démarrage (structure identique mais avec gestion d'erreur)
@app.on_event("startup")
async def startup():
    try:
        init_db()
        create_default_user()
        print("Initialisation DB et utilisateur admin réussie")
    except Exception as e:
        print(f"Erreur startup: {str(e)}")
        raise

# Inclusion des routes (identique à votre structure)
app.include_router(home.router)
app.include_router(auth_routes.router)
app.include_router(dashboard.router)
app.include_router(comments.router)

# Configuration Prometheus (optimisée mais discrète)
instrumentator = Instrumentator(
    should_group_status_codes=False,
    excluded_handlers=["/admin*", "/static*"]  # Exclusion des routes admin
)
instrumentator.instrument(app)
instrumentator.expose(app, endpoint="/metrics", include_in_schema=False)

# Middleware simplifié pour tracing
@app.middleware("http")
async def log_requests(request: Request, call_next):
    if "/admin" in request.url.path:
        print(f"Accès admin détecté: {request.url.path}")
    return await call_next(request)