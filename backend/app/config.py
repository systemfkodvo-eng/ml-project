"""
Configuration settings for the Health Risk Prediction Platform.
"""
from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # PostgreSQL (Neon)
    database_url: str = "postgresql://neondb_owner:npg_W0uhkRFS4GgU@ep-dawn-credit-a4gy43xa-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require"

    # JWT Authentication
    secret_key: str = "your-super-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # File Storage
    upload_dir: str = "./uploads"
    max_file_size: int = 50 * 1024 * 1024  # 50 MB
    allowed_extensions: list = ["pdf", "png", "jpg", "jpeg", "csv", "json"]

    # OCR Configuration
    tesseract_path: str = "C:/Program Files/Tesseract-OCR/tesseract.exe"

    # Environment
    environment: str = "development"
    debug: bool = True

    # API Settings
    api_v1_prefix: str = "/api/v1"
    project_name: str = "Health Risk Prediction Platform"

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Create uploads directory if it doesn't exist
settings = get_settings()
os.makedirs(settings.upload_dir, exist_ok=True)

