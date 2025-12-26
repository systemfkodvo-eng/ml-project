"""
Data extraction routes - OCR and NER processing.
"""
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db, Document as DocumentDB, Extraction as ExtractionDB, Patient as PatientDB
from app.models.document import DocumentType
from app.models.extraction import (
    ExtractionResponse, ExtractedField,
    ValidationStatus, ExtractionValidationRequest
)
from app.models.user import User
from app.services.auth_service import get_current_active_user
from app.services.extraction_service import ExtractionService

router = APIRouter(prefix="/extraction", tags=["Extraction"])
extraction_service = ExtractionService()


@router.post("/process/{document_id}", response_model=ExtractionResponse)
async def process_document(
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Process a document to extract medical data."""
    # Get document
    doc = db.query(DocumentDB).join(PatientDB).filter(
        DocumentDB.id == document_id,
        PatientDB.user_id == current_user.id
    ).first()

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Check if already processed
    existing = db.query(ExtractionDB).filter(ExtractionDB.document_id == document_id).first()
    if existing:
        return ExtractionResponse(
            id=existing.id,
            document_id=document_id,
            patient_id=existing.patient_id,
            extracted_fields=[ExtractedField(**f) for f in existing.extracted_fields] if existing.extracted_fields else [],
            status=ValidationStatus(existing.status),
            extraction_method="auto",
            created_at=existing.created_at,
            validated_at=existing.validated_at
        )

    doc.status = "processing"
    db.commit()

    try:
        doc_type = DocumentType(doc.document_type)
        extracted_fields, raw_text = await extraction_service.extract_from_document(doc.file_path, doc_type)

        if doc_type in [DocumentType.PDF, DocumentType.IMAGE]:
            method = "ocr_ner"
        elif doc_type == DocumentType.CSV:
            method = "csv_parse"
        else:
            method = "json_parse"

        # Save extraction
        db_extraction = ExtractionDB(
            document_id=document_id,
            patient_id=doc.patient_id,
            raw_text=raw_text[:10000] if raw_text else None,
            extracted_fields=[f.model_dump() for f in extracted_fields],
            status="pending"
        )

        db.add(db_extraction)
        doc.status = "extracted"
        db.commit()
        db.refresh(db_extraction)

        return ExtractionResponse(
            id=db_extraction.id,
            document_id=document_id,
            patient_id=doc.patient_id,
            extracted_fields=extracted_fields,
            status=ValidationStatus.PENDING,
            extraction_method=method,
            created_at=db_extraction.created_at
        )

    except Exception as e:
        doc.status = "error"
        db.commit()
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@router.get("/{extraction_id}", response_model=ExtractionResponse)
def get_extraction(
    extraction_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get extraction details by ID."""
    extraction = db.query(ExtractionDB).join(PatientDB).filter(
        ExtractionDB.id == extraction_id,
        PatientDB.user_id == current_user.id
    ).first()

    if not extraction:
        raise HTTPException(status_code=404, detail="Extraction not found")

    return ExtractionResponse(
        id=extraction.id,
        document_id=extraction.document_id,
        patient_id=extraction.patient_id,
        extracted_fields=[ExtractedField(**f) for f in extraction.extracted_fields] if extraction.extracted_fields else [],
        status=ValidationStatus(extraction.status),
        extraction_method="auto",
        created_at=extraction.created_at,
        validated_at=extraction.validated_at
    )


@router.post("/{extraction_id}/validate", response_model=ExtractionResponse)
def validate_extraction(
    extraction_id: str,
    validation: ExtractionValidationRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Validate and confirm extracted data. MANDATORY before prediction."""
    extraction = db.query(ExtractionDB).join(PatientDB).filter(
        ExtractionDB.id == extraction_id,
        PatientDB.user_id == current_user.id
    ).first()

    if not extraction:
        raise HTTPException(status_code=404, detail="Extraction not found")

    # Update fields with validated values
    validated_fields = []
    for field in validation.validated_fields:
        field_dict = field.model_dump()
        field_dict["is_validated"] = True
        if field_dict.get("validated_value") is None:
            field_dict["validated_value"] = field_dict["extracted_value"]
        validated_fields.append(field_dict)

    has_corrections = any(
        f.get("validated_value") != f.get("extracted_value")
        for f in validated_fields
    )

    new_status = "corrected" if has_corrections else "validated"

    extraction.extracted_fields = validated_fields
    extraction.status = new_status
    extraction.validated_at = datetime.utcnow()
    extraction.validated_by = current_user.id

    # Update document status
    doc = db.query(DocumentDB).filter(DocumentDB.id == extraction.document_id).first()
    if doc:
        doc.status = "validated"

    db.commit()
    db.refresh(extraction)

    return ExtractionResponse(
        id=extraction_id,
        document_id=extraction.document_id,
        patient_id=extraction.patient_id,
        extracted_fields=[ExtractedField(**f) for f in validated_fields],
        status=ValidationStatus(new_status),
        extraction_method="auto",
        created_at=extraction.created_at,
        validated_at=extraction.validated_at
    )

