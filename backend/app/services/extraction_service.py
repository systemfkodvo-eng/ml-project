"""
Extraction Service - Combines OCR and NER for document data extraction.
"""
import json
import csv
import io
import logging
from typing import List, Dict, Any
from pathlib import Path

from app.services.ocr_service import OCRService
from app.services.ner_service import NERService
from app.models.extraction import ExtractedField
from app.models.document import DocumentType

logger = logging.getLogger(__name__)


class ExtractionService:
    """Service for extracting structured data from various document types."""
    
    def __init__(self):
        """Initialize extraction service with OCR and NER."""
        self.ocr_service = OCRService()
        self.ner_service = NERService()
    
    async def extract_from_document(
        self, 
        file_path: str, 
        document_type: DocumentType
    ) -> tuple[List[ExtractedField], str]:
        """
        Extract data from a document based on its type.
        
        Args:
            file_path: Path to the document file
            document_type: Type of the document
            
        Returns:
            Tuple of (extracted fields, raw text)
        """
        if document_type == DocumentType.CSV:
            return await self._extract_from_csv(file_path)
        elif document_type == DocumentType.JSON:
            return await self._extract_from_json(file_path)
        elif document_type in [DocumentType.PDF, DocumentType.IMAGE]:
            return await self._extract_from_ocr(file_path, document_type)
        else:
            raise ValueError(f"Unsupported document type: {document_type}")
    
    async def _extract_from_csv(self, file_path: str) -> tuple[List[ExtractedField], str]:
        """Extract data from a CSV file."""
        extracted_fields = []
        raw_text = ""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
            
            # Parse CSV
            f_io = io.StringIO(raw_text)
            reader = csv.DictReader(f_io)
            
            # Take first row as data (for single patient CSV)
            for row in reader:
                for key, value in row.items():
                    if value and value.strip():
                        # Try to convert to float if numeric
                        try:
                            parsed_value = float(value)
                        except ValueError:
                            parsed_value = value.strip()
                        
                        extracted_fields.append(ExtractedField(
                            field_name=key.strip().lower().replace(' ', '_'),
                            extracted_value=parsed_value,
                            confidence=1.0,  # CSV parsing is deterministic
                            entity_type="CSV_FIELD",
                            source_text=f"{key}: {value}"
                        ))
                break  # Only first row
            
            logger.info(f"Extracted {len(extracted_fields)} fields from CSV")
            return extracted_fields, raw_text
            
        except Exception as e:
            logger.error(f"Error extracting from CSV: {e}")
            raise
    
    async def _extract_from_json(self, file_path: str) -> tuple[List[ExtractedField], str]:
        """Extract data from a JSON file."""
        extracted_fields = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
            
            data = json.loads(raw_text)
            
            # Flatten JSON structure
            def flatten(obj, prefix=''):
                fields = []
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        new_key = f"{prefix}_{k}" if prefix else k
                        fields.extend(flatten(v, new_key))
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        fields.extend(flatten(item, f"{prefix}_{i}"))
                else:
                    fields.append(ExtractedField(
                        field_name=prefix.lower().replace(' ', '_'),
                        extracted_value=obj,
                        confidence=1.0,
                        entity_type="JSON_FIELD",
                        source_text=f"{prefix}: {obj}"[:100]
                    ))
                return fields
            
            extracted_fields = flatten(data)
            logger.info(f"Extracted {len(extracted_fields)} fields from JSON")
            return extracted_fields, raw_text
            
        except Exception as e:
            logger.error(f"Error extracting from JSON: {e}")
            raise
    
    async def _extract_from_ocr(
        self, 
        file_path: str, 
        document_type: DocumentType
    ) -> tuple[List[ExtractedField], str]:
        """Extract data from image/PDF using OCR + NER."""
        try:
            # Step 1: OCR - Extract raw text
            file_type = 'pdf' if document_type == DocumentType.PDF else 'image'
            raw_text = await self.ocr_service.extract_text(file_path, file_type)
            
            if not raw_text:
                logger.warning("No text extracted from document")
                return [], ""
            
            # Step 2: NER - Extract entities
            extracted_fields = self.ner_service.extract_entities(raw_text)
            
            # Also try breast cancer specific features
            bc_fields = self.ner_service.extract_breast_cancer_features(raw_text)
            extracted_fields.extend(bc_fields)
            
            logger.info(f"Extracted {len(extracted_fields)} fields via OCR+NER")
            return extracted_fields, raw_text
            
        except Exception as e:
            logger.error(f"Error in OCR extraction: {e}")
            raise

