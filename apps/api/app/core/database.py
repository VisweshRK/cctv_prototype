"""Database connection and session management for Presence B2C.
Uses synchronous SQLAlchemy session with SQLite for high performance and zero deadlocks.
"""

from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from apps.api.app.core.config import settings

# SQLite connection URL
DB_URL = "sqlite:///./presence_b2c.db"

engine = create_engine(
    DB_URL,
    connect_args={"check_same_thread": False},
    echo=False
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)

Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
