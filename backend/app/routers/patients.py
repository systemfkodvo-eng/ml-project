"""
Patient management routes.
"""
import logging
from datetime import datetime
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from app.database import get_db, Patient as PatientDB, Document as DocumentDB, Analysis as AnalysisDB
from app.models.patient import PatientCreate, PatientUpdate, PatientResponse
from app.models.user import User
from app.services.auth_service import get_current_active_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/patients", tags=["Patients"])


@router.post("/", response_model=PatientResponse, status_code=status.HTTP_201_CREATED)
def create_patient(
    patient_data: PatientCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Create a new patient record."""
    # Check if patient_code already exists for this user
    existing = db.query(PatientDB).filter(
        PatientDB.patient_code == patient_data.patient_code,
        PatientDB.user_id == current_user.id
    ).first()

    if existing:
        raise HTTPException(status_code=400, detail="Patient with this code already exists")

    db_patient = PatientDB(
        patient_code=patient_data.patient_code,
        user_id=current_user.id,
        gender=patient_data.gender,
        date_of_birth=patient_data.date_of_birth,
        notes=patient_data.notes
    )

    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)

    return PatientResponse(
        id=db_patient.id,
        patient_code=db_patient.patient_code,
        gender=db_patient.gender,
        date_of_birth=db_patient.date_of_birth,
        notes=db_patient.notes,
        practitioner_id=db_patient.user_id,
        created_at=db_patient.created_at,
        updated_at=db_patient.updated_at,
        document_count=0,
        analysis_count=0
    )


@router.get("/", response_model=List[PatientResponse])
def list_patients(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    search: str = Query(None, description="Search by patient code"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """List all patients for the current practitioner."""
    logger.info(f"Listing patients for user: {current_user.id}")
    query = db.query(PatientDB).filter(PatientDB.user_id == current_user.id)

    if search:
        query = query.filter(PatientDB.patient_code.ilike(f"%{search}%"))

    patients = query.order_by(PatientDB.created_at.desc()).offset(skip).limit(limit).all()

    result = []
    for p in patients:
        doc_count = db.query(DocumentDB).filter(DocumentDB.patient_id == p.id).count()
        analysis_count = db.query(AnalysisDB).filter(AnalysisDB.patient_id == p.id).count()

        result.append(PatientResponse(
            id=p.id,
            patient_code=p.patient_code,
            gender=p.gender,
            date_of_birth=p.date_of_birth,
            notes=p.notes,
            practitioner_id=p.user_id,
            created_at=p.created_at,
            updated_at=p.updated_at,
            document_count=doc_count,
            analysis_count=analysis_count
        ))

    return result


@router.get("/{patient_id}", response_model=PatientResponse)
def get_patient(
    patient_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get a specific patient by ID."""
    patient = db.query(PatientDB).filter(
        PatientDB.id == patient_id,
        PatientDB.user_id == current_user.id
    ).first()

    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    doc_count = db.query(DocumentDB).filter(DocumentDB.patient_id == patient.id).count()
    analysis_count = db.query(AnalysisDB).filter(AnalysisDB.patient_id == patient.id).count()

    return PatientResponse(
        id=patient.id,
        patient_code=patient.patient_code,
        gender=patient.gender,
        date_of_birth=patient.date_of_birth,
        notes=patient.notes,
        practitioner_id=patient.user_id,
        created_at=patient.created_at,
        updated_at=patient.updated_at,
        document_count=doc_count,
        analysis_count=analysis_count
    )


@router.put("/{patient_id}", response_model=PatientResponse)
def update_patient(
    patient_id: str,
    patient_data: PatientUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Update a patient record."""
    patient = db.query(PatientDB).filter(
        PatientDB.id == patient_id,
        PatientDB.user_id == current_user.id
    ).first()

    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    update_data = patient_data.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(patient, key, value)

    db.commit()
    db.refresh(patient)

    return PatientResponse(
        id=patient.id,
        patient_code=patient.patient_code,
        gender=patient.gender,
        date_of_birth=patient.date_of_birth,
        notes=patient.notes,
        practitioner_id=patient.user_id,
        created_at=patient.created_at,
        updated_at=patient.updated_at
    )


@router.delete("/{patient_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_patient(
    patient_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Delete a patient and all associated data."""
    patient = db.query(PatientDB).filter(
        PatientDB.id == patient_id,
        PatientDB.user_id == current_user.id
    ).first()

    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    # Delete associated documents and analyses
    db.query(DocumentDB).filter(DocumentDB.patient_id == patient_id).delete()
    db.query(AnalysisDB).filter(AnalysisDB.patient_id == patient_id).delete()

    db.delete(patient)
    db.commit()

