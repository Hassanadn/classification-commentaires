from fastapi import APIRouter, Depends, HTTPException, Request, Form, Response
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from starlette.status import HTTP_401_UNAUTHORIZED

from api.database import get_db
from api.services.auth import AuthService 


router = APIRouter()
templates = Jinja2Templates(directory="src/api/templates")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

@router.get("/login")
async def login_get(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@router.post("/authentication")
async def authentication_post(
    request: Request,
    response: Response,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    user = AuthService.authenticate_user(username, password, db)
    if not user:
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Identifiants invalides")
    
    access_token = AuthService.create_access_token({"sub": user.username})

    response = templates.TemplateResponse("dashboard.html", {"request": request, "token": access_token})
    response.set_cookie(key="access_token", value=f"Bearer {access_token}", httponly=True, max_age=1800, secure=False)
    return response

@router.post("/token")
async def login_token(
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    user = AuthService.authenticate_user(username, password, db)
    if not user:
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Identifiants invalides")
    access_token = AuthService.create_access_token({"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/dashboard")
async def dashboard(token: str = Depends(oauth2_scheme)):
    payload = AuthService.get_user_info(token)
    if not payload:
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Token invalide ou expiré")
    username = payload.get("sub")
    return {"message": f"Bienvenue sur le dashboard, {username} !"}

@router.post("/logout")
async def logout(response: Response):
    
    response.delete_cookie(key="access_token")
    return templates.TemplateResponse("login.html", {"request": None, "message": "Vous avez été déconnecté."})
