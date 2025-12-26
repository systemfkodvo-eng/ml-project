"""
OCR Service for extracting text from images and PDFs.
Uses Tesseract OCR for text recognition.
"""
import os
import logging
from typing import Optional
from pathlib import Path

try:
    import pytesseract
    from PIL import Image
    from pdf2image import convert_from_path
    import PyPDF2
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Configure Tesseract path for Windows
if OCR_AVAILABLE and os.path.exists(settings.tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = settings.tesseract_path


class OCRService:
    """Service for Optical Character Recognition."""
    
    @staticmethod
    def is_available() -> bool:
        """Check if OCR service is available."""
        return OCR_AVAILABLE
    
    @staticmethod
    async def extract_text_from_image(image_path: str) -> str:
        """
        Extract text from an image file using Tesseract OCR.
        
        Args:
            image_path: Path to the image file (jpg, png, etc.)
            
        Returns:
            Extracted text from the image
        """
        if not OCR_AVAILABLE:
            logger.warning("OCR not available - pytesseract not installed")
            return ""
        
        try:
            image = Image.open(image_path)
            # Use French and English for medical documents
            text = pytesseract.image_to_string(image, lang='fra+eng')
            logger.info(f"Successfully extracted {len(text)} characters from image")
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
            raise
    
    @staticmethod
    async def extract_text_from_pdf(pdf_path: str) -> str:
        """
        Extract text from a PDF file.
        First tries direct text extraction, falls back to OCR if needed.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text from the PDF
        """
        text = ""
        
        # First, try direct text extraction (for text-based PDFs)
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            # If we got substantial text, return it
            if len(text.strip()) > 100:
                logger.info(f"Extracted {len(text)} characters directly from PDF")
                return text.strip()
        except Exception as e:
            logger.warning(f"Direct PDF text extraction failed: {e}")
        
        # Fall back to OCR (for scanned PDFs)
        if not OCR_AVAILABLE:
            logger.warning("OCR not available for scanned PDF")
            return text.strip()
        
        try:
            # Convert PDF pages to images
            images = convert_from_path(pdf_path)
            
            for i, image in enumerate(images):
                page_text = pytesseract.image_to_string(image, lang='fra+eng')
                text += f"\n--- Page {i+1} ---\n{page_text}"
            
            logger.info(f"Extracted {len(text)} characters from PDF via OCR")
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF via OCR: {e}")
            raise
    
    @staticmethod
    async def extract_text(file_path: str, file_type: str) -> str:
        """
        Extract text from a file based on its type.
        
        Args:
            file_path: Path to the file
            file_type: Type of file ('pdf', 'image')
            
        Returns:
            Extracted text
        """
        if file_type == 'pdf':
            return await OCRService.extract_text_from_pdf(file_path)
        elif file_type in ['image', 'jpg', 'jpeg', 'png']:
            return await OCRService.extract_text_from_image(file_path)
        else:
            raise ValueError(f"Unsupported file type for OCR: {file_type}")

