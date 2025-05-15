from fastapi import APIRouter, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request

router = APIRouter()
templates = Jinja2Templates(directory="src/api/templates")

@router.get("/dashboard", response_class=HTMLResponse)
async def read_dashboard(request: Request):
    # Vous pouvez charger ici des données dynamiques
    return templates.TemplateResponse("dashboard.html", {"request": request})
