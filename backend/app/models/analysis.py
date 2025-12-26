"""
Analysis models for ML predictions and results.
"""
# Import des classes de base Pydantic pour la validation des donnée
from pydantic import BaseModel, Field
# Types utiles
from typing import Optional, List, Dict, Any
from datetime import datetime
# Pour créer des énumérations (valeurs fixes)
from enum import Enum


class RiskLevel(str, Enum):
    """Risk level categories."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ModelType(str, Enum):
    """Available ML models."""
    BREAST_CANCER = "breast_cancer"
    # Add more models here as needed


class FeatureImportance(BaseModel):
    """Feature importance for explainability."""
    feature_name: str
    importance_score: float
    contribution: str  # "positive" or "negative"


class AnalysisBase(BaseModel):
    """Base analysis model."""
    patient_id: str
    extraction_id: str
    model_type: ModelType


class AnalysisCreate(AnalysisBase):
    """Model for creating an analysis."""
    input_data: Dict[str, Any]


class Analysis(AnalysisBase):
    """Analysis model with database fields."""
    id: Optional[str] = Field(None, alias="_id")
    practitioner_id: str
    input_data: Dict[str, Any]
    
    # Prediction results
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Risk probability 0-1")
    risk_level: RiskLevel
    risk_percentage: float  # risk_score * 100
    
    # Explainability
    feature_importance: Optional[List[FeatureImportance]] = None
    model_version: str = "1.0.0"
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    practitioner_notes: Optional[str] = None
    
    class Config:
        populate_by_name = True


class AnalysisResponse(BaseModel):
    """Analysis model for API responses."""
    id: str
    patient_id: str
    extraction_id: str
    model_type: ModelType
    
    risk_score: float
    risk_level: RiskLevel
    risk_percentage: float
    feature_importance: Optional[List[FeatureImportance]] = None
    model_version: str
    
    created_at: datetime
    practitioner_notes: Optional[str] = None
    
    # Warning message (required by spec)
    disclaimer: str = "Cet outil est une aide à la décision. Il ne remplace pas le diagnostic médical."
    
    class Config:
        populate_by_name = True


class PredictionRequest(BaseModel):
    """Request model for running a prediction."""
    extraction_id: str
    model_type: ModelType = ModelType.BREAST_CANCER

