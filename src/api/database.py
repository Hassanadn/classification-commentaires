# from databases import Database
# from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, DateTime, Text, insert, select
# from sqlalchemy.orm import declarative_base
# from typing import Optional
# from datetime import datetime

# from api.schemas.user import UserInDB
# from api.services.auth import AuthService

# DATABASE_URL = "sqlite:///./comments.db"

# database = Database(DATABASE_URL)
# engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
# metadata = MetaData()
# Base = declarative_base()

# # Modèles de table
# class User(Base):
#     __tablename__ = "users"
#     id = Column(Integer, primary_key=True, index=True)
#     username = Column(String(100), unique=True, index=True, nullable=False)
#     hashed_password = Column(String(255), nullable=False)
#     full_name = Column(String(255))
#     created_at = Column(DateTime, default=datetime.utcnow)

# class Comment(Base):
#     __tablename__ = "comments"
#     id = Column(Integer, primary_key=True, index=True)
#     content = Column(Text, nullable=False)
#     author = Column(String(100), nullable=False)
#     created_at = Column(DateTime, default=datetime.utcnow)



# # Fonction pour récupérer un utilisateur depuis la base de données
# async def get_user_by_username(username: str) -> Optional[UserInDB]:
#     query = select(User).where(User.username == username)
#     async with database.connection() as connection:
#         result = await connection.fetch_one(query)
#         if result:
#             return UserInDB(
#                 username=result["username"],
#                 hashed_password=result["hashed_password"]
#             )
#     return None

# async def create_default_user():
#     username = "admin"
#     password = "admin123"
#     full_name = "Admin User"

#     query = select(User).where(User.username == username)
#     existing_user = await database.fetch_one(query)
#     if existing_user:
#         print("✅ Utilisateur par défaut déjà existant.")
#         return

#     hashed_password = AuthService.get_password_hash(password)
#     insert_query = insert(User).values(
#         username=username,
#         hashed_password=hashed_password,
#         full_name=full_name,
#         created_at=datetime.utcnow()
#     )
#     await database.execute(insert_query)
#     print("✅ Utilisateur par défaut créé avec succès.")
from sqlalchemy.orm import Session
from api.db import engine, User
# src/api/database.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session


DATABASE_URL = "sqlite:///./comments.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
def get_user_by_username(username: str):
    with Session(engine) as session:
        return session.query(User).filter(User.username == username).first()
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()