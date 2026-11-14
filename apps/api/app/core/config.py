"""Application Configuration for Presence B2C API.
Supports PostgreSQL and SQLite fallbacks, JWT secrets, and CORS.
"""

import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "Presence B2C CCTV AI Platform"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"

    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "presence-b2c-super-secret-key-change-in-production-2026")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours

    # Database
    # Default to sqlite+aiosqlite for instant local verification; or postgresql+asyncpg
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "sqlite+aiosqlite:///e:/cctv/presence_b2c.db"
    )

    # CORS
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "*"
    ]

    # Camera Streaming & Storage
    STREAMING_PORT: int = 8000
    SNAPSHOT_DIR: str = "e:/cctv/storage/snapshots"

    model_config = SettingsConfigDict(case_sensitive=True)


settings = Settings()
