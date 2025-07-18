"""
Medical vertical module using pack-driven architecture.

This module provides medical-specific agents and tools that use external
rule packs instead of hard-coded logic.
"""

from .medical_rate_validator import MedicalRateValidator

__all__ = [
    "MedicalRateValidator"
]
