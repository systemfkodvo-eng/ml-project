"""
Prediction routes - ML model inference.
"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db, Extraction as ExtractionDB, Analysis as AnalysisDB, Patient as PatientDB
from app.models.analysis import (
    AnalysisResponse, RiskLevel, ModelType, PredictionRequest
)
from app.models.user import User
from app.services.auth_service import get_current_active_user
from app.services.prediction_service import PredictionService

router = APIRouter(prefix="/prediction", tags=["Prediction"])
prediction_service = PredictionService()


@router.post("/run", response_model=AnalysisResponse)
def run_prediction(
    request: PredictionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Run a prediction on validated extraction data."""
    extraction = db.query(ExtractionDB).join(PatientDB).filter(
        ExtractionDB.id == request.extraction_id,
        PatientDB.user_id == current_user.id
    ).first()

    if not extraction:
        raise HTTPException(status_code=404, detail="Extraction not found")

    if extraction.status not in ["validated", "corrected"]:
        raise HTTPException(
            status_code=400,
            detail="Extraction must be validated before prediction. Human validation is mandatory."
        )

    # Prepare input data from validated fields
    input_data = {}
    for field in extraction.extracted_fields or []:
        value = field.get("validated_value") or field.get("extracted_value")
        input_data[field["field_name"]] = value

    try:
        risk_score, risk_level, feature_importance = prediction_service.predict(
            input_data, request.model_type
        )

        # Save analysis
        db_analysis = AnalysisDB(
            patient_id=extraction.patient_id,
            extraction_id=request.extraction_id,
            user_id=current_user.id,
            model_type=request.model_type.value,
            model_version="1.0.0",
            input_features=input_data,
            risk_score=risk_score,
            risk_level=risk_level.value,
            risk_percentage=round(risk_score * 100, 2),
            feature_importance=[f.model_dump() for f in feature_importance]
        )

        db.add(db_analysis)
        db.commit()
        db.refresh(db_analysis)

        return AnalysisResponse(
            id=db_analysis.id,
            patient_id=extraction.patient_id,
            extraction_id=request.extraction_id,
            model_type=request.model_type,
            risk_score=risk_score,
            risk_level=risk_level,
            risk_percentage=db_analysis.risk_percentage,
            feature_importance=feature_importance,
            model_version="1.0.0",
            created_at=db_analysis.created_at,
            disclaimer="Cet outil est une aide à la décision. Il ne remplace pas le diagnostic médical."
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/analyses", response_model=List[AnalysisResponse])
def list_analyses(
    patient_id: str = Query(None, description="Filter by patient ID"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """List all analyses for the current practitioner."""
    query = db.query(AnalysisDB).filter(AnalysisDB.user_id == current_user.id)

    if patient_id:
        query = query.filter(AnalysisDB.patient_id == patient_id)

    analyses = query.order_by(AnalysisDB.created_at.desc()).offset(skip).limit(limit).all()

    return [
        AnalysisResponse(
            id=a.id,
            patient_id=a.patient_id,
            extraction_id=a.extraction_id,
            model_type=ModelType(a.model_type),
            risk_score=a.risk_score,
            risk_level=RiskLevel(a.risk_level),
            risk_percentage=a.risk_percentage,
            feature_importance=a.feature_importance,
            model_version=a.model_version,
            created_at=a.created_at,
            disclaimer="Cet outil est une aide à la décision. Il ne remplace pas le diagnostic médical."
        )
        for a in analyses
    ]


@router.get("/analyses/{analysis_id}", response_model=AnalysisResponse)
def get_analysis(
    analysis_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get a specific analysis result by ID."""
    analysis = db.query(AnalysisDB).filter(
        AnalysisDB.id == analysis_id,
        AnalysisDB.user_id == current_user.id
    ).first()

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return AnalysisResponse(
        id=analysis.id,
        patient_id=analysis.patient_id,
        extraction_id=analysis.extraction_id,
        model_type=ModelType(analysis.model_type),
        risk_score=analysis.risk_score,
        risk_level=RiskLevel(analysis.risk_level),
        risk_percentage=analysis.risk_percentage,
        feature_importance=analysis.feature_importance,
        model_version=analysis.model_version,
        created_at=analysis.created_at,
        disclaimer="Cet outil est une aide à la décision. Il ne remplace pas le diagnostic médical."
    )

