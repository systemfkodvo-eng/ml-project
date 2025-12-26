"""
Patient models for patient data management.
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime, date
from enum import Enum


class Gender(str, Enum):
    """Patient gender."""
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class PatientBase(BaseModel):
    """Base patient model."""
    # Pseudonymized identifier (not real name for privacy)
    patient_code: str = Field(..., min_length=3, max_length=50, description="Unique patient identifier code")
    date_of_birth: Optional[date] = None
    gender: Optional[Gender] = None
    medical_history: Optional[List[str]] = Field(default_factory=list)
    notes: Optional[str] = None


class PatientCreate(PatientBase):
    """Model for creating a new patient."""
    pass


class PatientUpdate(BaseModel):
    """Model for updating patient data."""
    patient_code: Optional[str] = None
    date_of_birth: Optional[date] = None
    gender: Optional[Gender] = None
    medical_history: Optional[List[str]] = None
    notes: Optional[str] = None


class Patient(PatientBase):
    """Patient model with database fields."""
    id: Optional[str] = Field(None, alias="_id")
    practitioner_id: str  # Reference to the practitioner who created this patient
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        populate_by_name = True


class PatientResponse(PatientBase):
    """Patient model for API responses."""
    id: str
    practitioner_id: str
    created_at: datetime
    # updated_at can be null for newly created patients (no update yet)
    # Make it optional to avoid 500 errors when serializing responses
    updated_at: Optional[datetime] = None
    document_count: Optional[int] = 0
    analysis_count: Optional[int] = 0
    
    class Config:
        populate_by_name = True

