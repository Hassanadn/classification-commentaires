from sqlalchemy import create_engine, MetaData, Column, Integer, String, DateTime, Text
from sqlalchemy.orm import declarative_base
from databases import Database
from datetime import datetime

DATABASE_URL = "sqlite:///./comments.db"

database = Database(DATABASE_URL)
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Base = declarative_base()
metadata = MetaData()

# Modèles ORM
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)

class Comment(Base):
    __tablename__ = "comments"
    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)  # ici content
    author = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    sentiment = Column(String(20), nullable=False)


def init_db():
    Base.metadata.create_all(bind=engine)
