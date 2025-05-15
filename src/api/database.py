from sqlalchemy.orm import Session
from src.api.db import engine, User
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