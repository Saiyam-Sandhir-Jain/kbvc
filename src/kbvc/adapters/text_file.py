# kbvc/adapters/text_file.py
"""
TextFileAdapter — v1 built-in adapter for Markdown and plain text files.

This is the only adapter needed for Saiyam's existing .md knowledge base.
Wraps the chunker.py functions so they can be called through the adapter
registry pattern (enabling v2+ adapters to follow the same interface).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List

from kbvc.adapters.base import SourceAdapter

if TYPE_CHECKING:
    from kbvc.core.chunker import Chunk
    from kbvc.core.ko import KnowledgeObject


class TextFileAdapter(SourceAdapter):
    """Processes Markdown (.md) and plain text (.txt) files."""

    SUPPORTED_EXTENSIONS = {".md", ".txt", ".markdown"}

    def can_process(self, source_type: str, path: str) -> bool:
        if source_type == "file":
            suffix = Path(path).suffix.lower()
            return suffix in self.SUPPORTED_EXTENSIONS
        return False

    def extract_chunks(self, ko: "KnowledgeObject") -> List["Chunk"]:
        from kbvc.core.chunker import parse_frontmatter, split_into_chunks

        src = Path(ko.path)
        if not src.exists():
            raise FileNotFoundError(f"Source file not found: {ko.path}")

        content = src.read_text(encoding="utf-8")
        frontmatter, body = parse_frontmatter(content)
        return split_into_chunks(body, frontmatter)
