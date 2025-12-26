"""
API Routers for the Health Risk Prediction Platform.
"""
from app.routers.auth import router as auth_router
from app.routers.patients import router as patients_router
from app.routers.documents import router as documents_router
from app.routers.extraction import router as extraction_router
from app.routers.prediction import router as prediction_router
from app.routers.admin import router as admin_router

__all__ = [
    "auth_router",
    "patients_router",
    "documents_router",
    "extraction_router",
    "prediction_router",
    "admin_router"
]

