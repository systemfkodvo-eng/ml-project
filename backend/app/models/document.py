"""
Document models for file upload and management.
"""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum


class DocumentType(str, Enum):
    """Types of documents that can be uploaded."""
    PDF = "pdf"
    IMAGE = "image"  # jpg, png
    CSV = "csv"
    JSON = "json"


class DocumentStatus(str, Enum):
    """Document processing status."""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    EXTRACTED = "extracted"
    VALIDATED = "validated"
    ERROR = "error"


class DocumentBase(BaseModel):
    """Base document model."""
    filename: str
    document_type: DocumentType
    description: Optional[str] = None


class DocumentCreate(DocumentBase):
    """Model for creating a document record."""
    patient_id: str
    file_path: str
    file_size: int
    mime_type: str


class Document(DocumentBase):
    """Document model with database fields."""
    id: Optional[str] = Field(None, alias="_id")
    patient_id: str
    practitioner_id: str
    file_path: str
    file_size: int
    mime_type: str
    status: DocumentStatus = DocumentStatus.UPLOADED
    raw_text: Optional[str] = None  # OCR extracted text
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        populate_by_name = True


class DocumentResponse(DocumentBase):
    """Document model for API responses."""
    id: str
    patient_id: str
    status: DocumentStatus
    file_size: int
    created_at: datetime
    has_extraction: bool = False
    
    class Config:
        populate_by_name = True

