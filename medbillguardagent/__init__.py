"""
MedBillGuardAgent - AI micro-service to detect over-charges in Indian hospital bills.

This package provides:
- FastAPI web service for bill analysis
- OCR pipeline for document processing  
- Rate validation against CGHS/ESI/NPPA databases
- LLM-powered intelligent parsing and analysis
"""

__version__ = "1.0.0"
__author__ = "VivaranAI Team"
__email__ = "dev@vivaranai.com"

# Package-level imports for convenience
# from .main import app  # TODO: Uncomment when main.py is created
from .schemas import MedBillGuardRequest, MedBillGuardResponse, RedFlag

__all__ = [
    # "app",  # TODO: Uncomment when main.py is created
    "MedBillGuardRequest", 
    "MedBillGuardResponse",
    "RedFlag",
] 