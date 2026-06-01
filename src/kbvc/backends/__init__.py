# kbvc/backends/__init__.py
"""
Backend factory — instantiate the configured embed and vector DB backends.
All commands that need to embed or query vectors call get_embed_backend()
and get_vectordb_backend() with the config dict from KbvcRepo.config().
"""

from __future__ import annotations

from kbvc.backends.embed import EmbedBackend
from kbvc.backends.vectordb import VectorDBBackend


def get_embed_backend(config: dict) -> EmbedBackend:
    """
    Instantiate the configured embed backend.
    Raises ValueError with a helpful message if not configured.
    """
    backend_name = config.get("embed.backend", "").strip()
    if not backend_name:
        raise ValueError(
            "No embedding backend configured.\n"
            "Run: kbvc config set embed.backend openai  "
            "(or gemini / ollama / huggingface)"
        )
    if backend_name == "openai":
        from kbvc.backends.embed.openai import OpenAIEmbedBackend
        return OpenAIEmbedBackend.from_config(config)
    if backend_name == "gemini":
        from kbvc.backends.embed.gemini import GeminiEmbedBackend
        return GeminiEmbedBackend.from_config(config)
    if backend_name == "ollama":
        from kbvc.backends.embed.ollama import OllamaEmbedBackend
        return OllamaEmbedBackend.from_config(config)
    if backend_name == "huggingface":
        from kbvc.backends.embed.huggingface import HuggingFaceEmbedBackend
        return HuggingFaceEmbedBackend.from_config(config)
    raise ValueError(f"Unknown embed backend: '{backend_name}'")


def get_vectordb_backend(config: dict) -> VectorDBBackend:
    """
    Instantiate the configured vector DB backend.
    Raises ValueError with a helpful message if not configured.
    """
    backend_name = config.get("vectordb.backend", "").strip()
    if not backend_name:
        raise ValueError(
            "No vector DB backend configured.\n"
            "Run: kbvc config set vectordb.backend qdrant  "
            "(or pgvector / pinecone / chroma)"
        )
    if backend_name == "qdrant":
        from kbvc.backends.vectordb.qdrant import QdrantBackend
        return QdrantBackend.from_config(config)
    if backend_name == "pgvector":
        from kbvc.backends.vectordb.pgvector import PgvectorBackend
        return PgvectorBackend.from_config(config)
    if backend_name == "pinecone":
        from kbvc.backends.vectordb.pinecone import PineconeBackend
        return PineconeBackend.from_config(config)
    if backend_name == "chroma":
        from kbvc.backends.vectordb.chroma import ChromaBackend
        return ChromaBackend.from_config(config)
    raise ValueError(f"Unknown vectordb backend: '{backend_name}'")
