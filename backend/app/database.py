"""
Database connection setup for MARISA CDSS.

Uses PostgreSQL in production (via Docker Compose), but falls back to a
local SQLite file if DATABASE_URL isn't set - this lets you develop and
test the API locally without needing Postgres running yet.
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./marisa_local.db"  # fallback for local dev without Docker
)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency - yields a DB session per request, closes it after."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
