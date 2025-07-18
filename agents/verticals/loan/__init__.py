"""
Loan vertical module using pack-driven architecture.

This module provides loan-specific agents and tools that use external
rule packs instead of hard-coded logic.
"""

from .loan_risk_agent import LoanRiskAgent

__all__ = [
    "LoanRiskAgent"
]
