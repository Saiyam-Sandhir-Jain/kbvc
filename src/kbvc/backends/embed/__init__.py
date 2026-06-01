# kbvc/backends/embed/__init__.py
"""
EmbedBackend ABC — interface all embedding backends must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class EmbedBackend(ABC):
    """Abstract base class for all embedding backends."""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Embed a single string. Returns the vector as a list of floats."""
        ...

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple strings. Returns a list of vectors (same order as input)."""
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """The embedding dimension for this model."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The canonical model identifier."""
        ...

    @classmethod
    def from_config(cls, config: dict) -> "EmbedBackend":
        raise NotImplementedError(
            "Each backend subclass must implement from_config()"
        )
