# kbvc/adapters/base.py
"""
SourceAdapter ABC — the extension point for non-file KO sources.

v1: only TextFileAdapter is built-in.
v2+: community adapters for audio, images, web pages, APIs, etc.
     Register via pyproject.toml entry_points under "kbvc.adapters".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from kbvc.core.chunker import Chunk
    from kbvc.core.ko import KnowledgeObject


class SourceAdapter(ABC):
    """
    Processes a KO source into a list of text chunks for embedding.

    The adapter pattern decouples KBVC's embedding pipeline from the
    specifics of how different source types are read and chunked.
    """

    @abstractmethod
    def can_process(self, source_type: str, path: str) -> bool:
        """Return True if this adapter can handle the given source."""
        ...

    @abstractmethod
    def extract_chunks(self, ko: "KnowledgeObject") -> List["Chunk"]:
        """Read the source and return a list of text chunks for embedding."""
        ...
