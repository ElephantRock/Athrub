"""Athrub research package."""

from .contracts import BenchmarkRecord, DecisionRequest, DecisionResult
from .reference import FlatReferenceBackend, ScalarDecisionHead, TextDecisionCodec
from .shared_context import SharedContextBackend

__all__ = [
    "BenchmarkRecord",
    "DecisionRequest",
    "DecisionResult",
    "FlatReferenceBackend",
    "ScalarDecisionHead",
    "SharedContextBackend",
    "TextDecisionCodec",
]
__version__ = "0.1.0.dev0"
