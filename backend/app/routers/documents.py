"""
Document upload and management routes.
"""
import os
import uuid
from typing import List
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db, Document as DocumentDB, Patient as PatientDB
from app.models.document import DocumentResponse, DocumentType, DocumentStatus
from app.models.user import User
from app.services.auth_service import get_current_active_user

router = APIRouter(prefix="/documents", tags=["Documents"])
settings = get_settings()


def get_document_type(filename: str, mime_type: str) -> str:
    """Determine document type from filename and mime type."""
    ext = filename.lower().split('.')[-1]

    if ext == 'pdf' or 'pdf' in mime_type:
        return 'pdf'
    elif ext in ['jpg', 'jpeg', 'png'] or 'image' in mime_type:
        return 'image'
    elif ext == 'csv' or 'csv' in mime_type:
        return 'csv'
    elif ext == 'json' or 'json' in mime_type:
        return 'json'
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: pdf, jpg, png, csv, json"
        )


@router.post("/upload/{patient_id}", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    patient_id: str,
    file: UploadFile = File(...),
    description: str = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Upload a document for a patient."""
    # Verify patient exists and belongs to user
    patient = db.query(PatientDB).filter(
        PatientDB.id == patient_id,
        PatientDB.user_id == current_user.id
    ).first()

    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Check file size
    content = await file.read()
    if len(content) > settings.max_file_size:
        raise HTTPException(status_code=400, detail=f"File too large. Max: {settings.max_file_size // (1024*1024)}MB")

    # Determine document type
    doc_type = get_document_type(file.filename, file.content_type or "")

    # Generate unique filename and save
    file_ext = file.filename.split('.')[-1]
    unique_filename = f"{uuid.uuid4()}.{file_ext}"
    file_path = Path(settings.upload_dir) / patient_id / unique_filename
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'wb') as f:
        f.write(content)

    # Create document record
    db_doc = DocumentDB(
        patient_id=patient_id,
        filename=unique_filename,
        original_filename=file.filename,
        file_path=str(file_path),
        document_type=doc_type,
        mime_type=file.content_type or "",
        file_size=len(content),
        status="uploaded"
    )

    db.add(db_doc)
    db.commit()
    db.refresh(db_doc)

    return DocumentResponse(
        id=db_doc.id,
        filename=db_doc.original_filename,
        document_type=DocumentType(db_doc.document_type),
        description=description,
        patient_id=patient_id,
        status=DocumentStatus.UPLOADED,
        file_size=db_doc.file_size,
        created_at=db_doc.created_at,
        has_extraction=False
    )


@router.get("/patient/{patient_id}", response_model=List[DocumentResponse])
def list_patient_documents(
    patient_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """List all documents for a patient."""
    patient = db.query(PatientDB).filter(
        PatientDB.id == patient_id,
        PatientDB.user_id == current_user.id
    ).first()

    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    documents = db.query(DocumentDB).filter(
        DocumentDB.patient_id == patient_id
    ).order_by(DocumentDB.created_at.desc()).all()

    return [
        DocumentResponse(
            id=doc.id,
            filename=doc.original_filename,
            document_type=DocumentType(doc.document_type),
            description=None,
            patient_id=doc.patient_id,
            status=DocumentStatus(doc.status),
            file_size=doc.file_size,
            created_at=doc.created_at,
            has_extraction=doc.status in ["extracted", "validated"]
        )
        for doc in documents
    ]


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get a specific document by ID."""
    doc = db.query(DocumentDB).join(PatientDB).filter(
        DocumentDB.id == document_id,
        PatientDB.user_id == current_user.id
    ).first()

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    return DocumentResponse(
        id=doc.id,
        filename=doc.original_filename,
        document_type=DocumentType(doc.document_type),
        description=None,
        patient_id=doc.patient_id,
        status=DocumentStatus(doc.status),
        file_size=doc.file_size,
        created_at=doc.created_at,
        has_extraction=doc.status in ["extracted", "validated"]
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Delete a document."""
    doc = db.query(DocumentDB).join(PatientDB).filter(
        DocumentDB.id == document_id,
        PatientDB.user_id == current_user.id
    ).first()

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Delete file
    if os.path.exists(doc.file_path):
        os.remove(doc.file_path)

    db.delete(doc)
    db.commit()

