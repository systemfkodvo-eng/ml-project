"""
PostgreSQL database connection and models using SQLAlchemy.
"""
from sqlalchemy import create_engine, Column, String, Float, DateTime, Text, Boolean, JSON, ForeignKey, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
from app.config import get_settings
import uuid
import logging

logger = logging.getLogger(__name__)

settings = get_settings()

# Create engine with SSL for Neon (using psycopg3)
engine = create_engine(
    settings.database_url.replace("postgresql://", "postgresql+psycopg://"),
    echo=settings.debug,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def generate_uuid():
    return str(uuid.uuid4())


# SQLAlchemy Models
class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    role = Column(String(50), default="practitioner")
    specialty = Column(String(100), nullable=True)
    institution = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    patients = relationship("Patient", back_populates="user")


class Patient(Base):
    __tablename__ = "patients"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    patient_code = Column(String(100), nullable=False, index=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    gender = Column(String(20), nullable=True)
    date_of_birth = Column(DateTime, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    user = relationship("User", back_populates="patients")
    documents = relationship("Document", back_populates="patient")
    analyses = relationship("Analysis", back_populates="patient")


class Document(Base):
    __tablename__ = "documents"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    patient_id = Column(String(36), ForeignKey("patients.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    document_type = Column(String(50), nullable=False)  # pdf, image, csv, json
    mime_type = Column(String(100), nullable=True)
    file_size = Column(Integer, nullable=True)
    status = Column(String(50), default="uploaded")  # uploaded, extracted, validated
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    patient = relationship("Patient", back_populates="documents")
    extraction = relationship("Extraction", back_populates="document", uselist=False)


class Extraction(Base):
    __tablename__ = "extractions"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    document_id = Column(String(36), ForeignKey("documents.id"), nullable=False)
    patient_id = Column(String(36), ForeignKey("patients.id"), nullable=False)
    raw_text = Column(Text, nullable=True)
    extracted_fields = Column(JSON, nullable=True)  # List of {field_name, extracted_value, validated_value, confidence}
    status = Column(String(50), default="pending")  # pending, extracted, validated, corrected
    validated_by = Column(String(36), nullable=True)
    validated_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    document = relationship("Document", back_populates="extraction")


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    patient_id = Column(String(36), ForeignKey("patients.id"), nullable=False)
    extraction_id = Column(String(36), ForeignKey("extractions.id"), nullable=False)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    model_type = Column(String(100), default="breast_cancer")
    model_version = Column(String(50), default="1.0.0")
    input_features = Column(JSON, nullable=True)
    risk_score = Column(Float, nullable=False)
    risk_level = Column(String(50), nullable=False)  # low, moderate, high, very_high
    risk_percentage = Column(Float, nullable=False)
    feature_importance = Column(JSON, nullable=True)
    disclaimer = Column(Text, default="Cet outil est une aide à la décision. Il ne remplace pas le diagnostic médical.")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    patient = relationship("Patient", back_populates="analyses")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    action = Column(String(100), nullable=False)  # login, logout, create_patient, upload_document, run_prediction, etc.
    details = Column(Text, nullable=True)
    ip_address = Column(String(50), nullable=True)
    user_agent = Column(String(500), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User")


def init_db():
    """Create all tables."""
    Base.metadata.create_all(bind=engine)
    logger.info("✅ Database tables created successfully!")


def get_db():
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

