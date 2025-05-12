from sqlalchemy import Table, Column, Integer, String, DateTime
from sqlalchemy.sql import func
from .database import metadata

comments = Table(
    "comments",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("text", String, nullable=False),
    Column("sentiment", String, nullable=False),
    Column("created_at", DateTime(timezone=True), server_default=func.now()),
)
