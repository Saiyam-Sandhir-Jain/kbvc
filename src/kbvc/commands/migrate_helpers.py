# kbvc/commands/migrate_helpers.py
"""
Shared helpers for kbvc migrate and kbvc backend commands.
Keeps the CLI thin by centralising backend loading logic.
"""

from __future__ import annotations

from typing import Dict


def load_backend(name: str, cfg: Dict[str, str]):
    """
    Load and return a VectorDBBackend instance by provider name.

    Uses the URL / key from cfg but overrides the backend type with `name`,
    allowing migration between backends that share the same server config.
    """
    n = name.lower().strip()
    if n in ("qdrant",):
        from kbvc.backends.vectordb.qdrant import QdrantBackend
        return QdrantBackend.from_config(cfg)
    if n in ("pgvector", "supabase", "postgres", "neon"):
        from kbvc.backends.vectordb.pgvector import PgvectorBackend
        return PgvectorBackend.from_config(cfg)
    if n in ("chroma", "chromadb"):
        from kbvc.backends.vectordb.chroma import ChromaBackend
        return ChromaBackend.from_config(cfg)
    if n in ("pinecone",):
        from kbvc.backends.vectordb.pinecone import PineconeBackend
        return PineconeBackend.from_config(cfg)
    raise ValueError(
        f"Unknown backend: '{name}'.\n"
        "Supported: qdrant, pgvector, chroma, pinecone"
    )


def detect_backend(cfg: Dict[str, str]) -> str:
    """Return the configured vector backend name, or 'unknown'."""
    return cfg.get("vectordb.backend", "unknown")
