"""
NER Service for extracting medical entities from text.
Uses pattern matching and optionally spaCy for Named Entity Recognition.
"""
import re
import logging
from typing import List, Dict, Any, Tuple

from app.models.extraction import ExtractedField

logger = logging.getLogger(__name__)

# Try to import spaCy (optional dependency)
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not available - using pattern-based extraction only")


class NERService:
    """Service for Named Entity Recognition on medical texts."""
    
    # Medical patterns for common fields (French + English)
    MEDICAL_PATTERNS = {
        "diagnosis_status": [
            (r"(?:statut|status|diagnostic)[:\s]*(?:est\s+)?(malin|malignant|bénin|benign)", 0.9),
            (r"(malin|malignant|bénin|benign)", 0.7),
        ],
        "tumor_size": [
            (r"(?:taille|size|diamètre|diameter)[:\s]*(\d+(?:[.,]\d+)?)\s*(?:mm|cm)", 0.9),
            (r"(\d+(?:[.,]\d+)?)\s*(?:mm|cm)", 0.6),
        ],
        "radius_mean": [
            (r"(?:radius[_\s]?mean|rayon[_\s]?moyen)[:\s]*(\d+(?:[.,]\d+)?)", 0.95),
        ],
        "texture_mean": [
            (r"(?:texture[_\s]?mean|texture[_\s]?moyenne)[:\s]*(\d+(?:[.,]\d+)?)", 0.95),
        ],
        "perimeter_mean": [
            (r"(?:perimeter[_\s]?mean|périmètre[_\s]?moyen)[:\s]*(\d+(?:[.,]\d+)?)", 0.95),
        ],
        "area_mean": [
            (r"(?:area[_\s]?mean|aire[_\s]?moyenne)[:\s]*(\d+(?:[.,]\d+)?)", 0.95),
        ],
        "smoothness_mean": [
            (r"(?:smoothness[_\s]?mean|lissage[_\s]?moyen)[:\s]*(\d+(?:[.,]\d+)?)", 0.95),
        ],
        "compactness_mean": [
            (r"(?:compactness[_\s]?mean|compacité[_\s]?moyenne)[:\s]*(\d+(?:[.,]\d+)?)", 0.95),
        ],
        "concavity_mean": [
            (r"(?:concavity[_\s]?mean|concavité[_\s]?moyenne)[:\s]*(\d+(?:[.,]\d+)?)", 0.95),
        ],
        "concave_points_mean": [
            (r"(?:concave[_\s]?points[_\s]?mean)[:\s]*(\d+(?:[.,]\d+)?)", 0.95),
        ],
        "symmetry_mean": [
            (r"(?:symmetry[_\s]?mean|symétrie[_\s]?moyenne)[:\s]*(\d+(?:[.,]\d+)?)", 0.95),
        ],
        "fractal_dimension_mean": [
            (r"(?:fractal[_\s]?dimension[_\s]?mean)[:\s]*(\d+(?:[.,]\d+)?)", 0.95),
        ],
        "patient_age": [
            (r"(?:âge|age)[:\s]*(\d+)\s*(?:ans|years)?", 0.9),
        ],
        "blood_marker": [
            (r"(?:taux|level|marqueur|marker)[:\s]*(\d+(?:[.,]\d+)?)\s*(?:g/[lL]|mg/[lL]|UI/[lL])?", 0.8),
        ],
    }
    
    # Breast cancer specific features (Wisconsin dataset)
    BREAST_CANCER_FEATURES = [
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
        "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean",
        "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se",
        "smoothness_se", "compactness_se", "concavity_se", "concave_points_se",
        "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst",
        "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst",
        "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
    ]
    
    def __init__(self):
        """Initialize NER service."""
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy model loaded successfully")
            except OSError:
                logger.warning("spaCy model not found - using pattern matching only")
    
    def extract_entities(self, text: str) -> List[ExtractedField]:
        """
        Extract medical entities from text using pattern matching.
        
        Args:
            text: Raw text from OCR or document
            
        Returns:
            List of extracted fields with confidence scores
        """
        extracted_fields = []
        text_lower = text.lower()
        
        for field_name, patterns in self.MEDICAL_PATTERNS.items():
            for pattern, base_confidence in patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    value = match.group(1) if match.groups() else match.group(0)
                    # Clean numeric values
                    if any(c.isdigit() for c in value):
                        value = value.replace(',', '.')
                        try:
                            value = float(value)
                        except ValueError:
                            pass
                    
                    extracted_fields.append(ExtractedField(
                        field_name=field_name,
                        extracted_value=value,
                        confidence=base_confidence,
                        entity_type="MEDICAL_VALUE",
                        source_text=match.group(0)[:100]
                    ))
                    break  # Take first match for each field
        
        return extracted_fields
    
    def extract_breast_cancer_features(self, text: str) -> List[ExtractedField]:
        """Extract breast cancer specific features from text."""
        fields = []
        lines = text.split('\n')
        
        for line in lines:
            for feature in self.BREAST_CANCER_FEATURES:
                pattern = rf"{feature}[:\s,]*(-?\d+(?:\.\d+)?)"
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    fields.append(ExtractedField(
                        field_name=feature,
                        extracted_value=float(match.group(1)),
                        confidence=0.95,
                        entity_type="BREAST_CANCER_FEATURE",
                        source_text=line[:100]
                    ))
        
        return fields

