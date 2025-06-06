# src/models.py

from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, DateTime, Text
from sqlalchemy.orm import declarative_base
from datetime import datetime

DATABASE_URL = "sqlite:///./comments.db"

Base = declarative_base()
metadata = MetaData()

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
    content = Column(Text, nullable=False) 
    author = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    sentiment = Column(String(20), nullable=False)
