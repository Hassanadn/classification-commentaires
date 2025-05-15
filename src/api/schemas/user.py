from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class UserBase(BaseModel):
    username: str
    full_name: Optional[str] = None

    class Config:
        orm_mode = True

class UserCreate(UserBase):
    password: str

class UserOut(UserBase):
    id: int
    created_at: datetime

    class Config:
        orm_mode = True

class UserInDB(UserBase):
    hashed_password: str

    class Config:
        orm_mode = True
