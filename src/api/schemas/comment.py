from pydantic import BaseModel
from datetime import datetime

# Modèle de base pour les commentaires, partagé entre les autres
class CommentBase(BaseModel):
    content: str

# Modèle pour créer un commentaire
class CommentCreate(CommentBase):
    author: str

# Modèle pour la réponse (retourner un commentaire existant)
class CommentOut(CommentBase):
    id: int
    author: str
    created_at: datetime

    class Config:
        from_attributes = True  # Pydantic v2: remplace orm_mode
