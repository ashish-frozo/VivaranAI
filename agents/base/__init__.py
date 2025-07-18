"""
Base abstractions for multi-vertical pack-driven architecture.

This module provides the foundational classes and interfaces for the
pack-driven system that supports multiple verticals (medical, loan, etc.).
"""

from .validators import BaseRateValidator, ValidationDelta

# Import base_tool if it exists
try:
    from .base_tool import BaseTool
    __all__ = ["BaseTool", "BaseRateValidator", "ValidationDelta"]
except ImportError:
    __all__ = ["BaseRateValidator", "ValidationDelta"]
