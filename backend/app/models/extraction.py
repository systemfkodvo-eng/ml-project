"""
Extraction models for OCR and NER results.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class ValidationStatus(str, Enum):
    """Extraction validation status."""
    PENDING = "pending"
    VALIDATED = "validated"
    CORRECTED = "corrected"
    REJECTED = "rejected"


class ExtractedField(BaseModel):
    """A single extracted field from document."""
    field_name: str
    extracted_value: Any
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")
    validated_value: Optional[Any] = None  # Value after human validation
    is_validated: bool = False
    entity_type: Optional[str] = None  # NER entity type (e.g., "DIAGNOSIS", "MEASUREMENT")
    source_text: Optional[str] = None  # Original text snippet where value was found


class ExtractionBase(BaseModel):
    """Base extraction model."""
    document_id: str
    patient_id: str


class ExtractionCreate(ExtractionBase):
    """Model for creating an extraction record."""
    extracted_fields: List[ExtractedField]
    raw_text: Optional[str] = None


class Extraction(ExtractionBase):
    """Extraction model with database fields."""
    id: Optional[str] = Field(None, alias="_id")
    practitioner_id: str
    extracted_fields: List[ExtractedField]
    raw_text: Optional[str] = None
    status: ValidationStatus = ValidationStatus.PENDING
    extraction_method: str = "ocr_ner"  # or "csv_parse", "json_parse"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    validated_at: Optional[datetime] = None
    validated_by: Optional[str] = None  # practitioner_id who validated
    
    class Config:
        populate_by_name = True


class ExtractionResponse(BaseModel):
    """Extraction model for API responses."""
    id: str
    document_id: str
    patient_id: str
    extracted_fields: List[ExtractedField]
    status: ValidationStatus
    extraction_method: str
    created_at: datetime
    validated_at: Optional[datetime] = None
    
    class Config:
        populate_by_name = True


class ExtractionValidationRequest(BaseModel):
    """Request model for validating extraction results."""
    validated_fields: List[ExtractedField]

