# kbvc/backends/vectordb/__init__.py
"""
VectorDBBackend ABC — interface all vector DB backends must implement.

VSAL (Vector Storage Abstraction Layer):
  KBVC code never touches Qdrant Points, PgVector rows, or Pinecone records
  directly.  It works only with ChunkRecord — the universal storage unit.
  Each backend adapter maps ChunkRecord to its native representation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# ChunkRecord — universal KBVC storage unit (VSAL)
# ---------------------------------------------------------------------------

@dataclass
class ChunkRecord:
    """
    Universal storage unit for KBVC.

    Every vector DB backend maps this to its native representation:
      Qdrant    → Point  (id + vector + payload)
      PgVector  → Row    (kbvc_chunks table)
      Pinecone  → Vector (namespace + metadata)
      Chroma    → Document
      LanceDB   → Row in kbvc_chunks.lance

    KBVC core only creates and consumes ChunkRecords — never raw DB objects.
    """
    vector_id: str        # e.g. "main__manifestai__chunk_0"
    branch: str
    ko_id: str
    ko_version: int
    chunk_index: int
    chunk_hash: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""  # ISO-8601 UTC; set by commit command

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ChunkRecord":
        return cls(**d)

    # Convenience: produce the {id, vector, metadata} shape the existing
    # upsert_batch API expects (backward-compatible shim).
    def to_upsert_item(self) -> Dict[str, Any]:
        return {
            "id": self.vector_id,
            "vector": self.embedding,
            "metadata": {
                "branch": self.branch,
                "ko_id": self.ko_id,
                "ko_version": self.ko_version,
                "chunk_index": self.chunk_index,
                "chunk_hash": self.chunk_hash,
                "created_at": self.created_at,
                **self.metadata,
            },
        }


# ---------------------------------------------------------------------------
# VectorDBBackend ABC
# ---------------------------------------------------------------------------

class VectorDBBackend(ABC):
    """Abstract base class for all vector database backends."""

    # ── existing write/read interface (unchanged) ───────────────────────────

    @abstractmethod
    def upsert(
        self,
        collection: str,
        id: str,
        vector: List[float],
        metadata: Dict[str, Any],
    ) -> None:
        """Upsert a single vector with metadata."""
        ...

    @abstractmethod
    def upsert_batch(
        self,
        collection: str,
        items: List[Dict],
    ) -> None:
        """
        Upsert multiple vectors.
        Each item: {"id": str, "vector": List[float], "metadata": dict}
        """
        ...

    @abstractmethod
    def delete(self, collection: str, id: str) -> None:
        """Delete a single vector by its string ID."""
        ...

    @abstractmethod
    def delete_by_prefix(self, collection: str, id_prefix: str) -> None:
        """
        Delete all vectors whose string ID starts with id_prefix.
        Used to clean up an entire KO's vectors when a KO is removed.
        """
        ...

    @abstractmethod
    def query(
        self,
        collection: str,
        vector: List[float],
        top_k: int,
        filter: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Nearest-neighbour search.
        Returns list of {id: str, score: float, metadata: dict}.
        """
        ...

    @abstractmethod
    def patch_metadata(
        self,
        collection: str,
        id_prefix: str,
        patch: Dict[str, Any],
    ) -> None:
        """
        Update metadata fields on all vectors matching id_prefix.
        Does NOT touch the embedding vector — used for volatility changes,
        version number sync, commit_id back-fill, etc.
        """
        ...

    @abstractmethod
    def exists_batch(
        self,
        collection: str,
        ids: List[str],
    ) -> Dict[str, bool]:
        """
        Check which vector IDs exist in the collection.
        Returns {id: bool} mapping.
        Used by kbvc checkout (v2 optimization) to determine which chunks
        are still live vs. need re-embedding.
        Implement as a batch fetch by ID — NOT an ANN search.
        """
        ...

    # ── VSAL: schema management & migration (new) ───────────────────────────

    def initialize_schema(self, collection: str, dimensions: int) -> None:
        """
        Idempotently create the backend schema for KBVC.

        Called by `kbvc backend init`.  Default implementation is a no-op
        (backends that auto-create collections on first upsert don't need this,
        but explicit init is preferred for production deployments).
        """

    def export_chunks(self, collection: str) -> List[ChunkRecord]:
        """
        Export all ChunkRecords from this backend.

        Used by `kbvc migrate --from <backend>`.  Default raises NotImplementedError
        so backends that don't support full export are caught at migration time.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement export_chunks(). "
            "Migration from this backend is not yet supported."
        )

    def import_chunks(self, collection: str, chunks: List[ChunkRecord]) -> None:
        """
        Import a list of ChunkRecords into this backend.

        Used by `kbvc migrate --to <backend>`.  Default implementation calls
        upsert_batch() using the backward-compatible shim — works for any backend
        that has upsert_batch without changes.
        """
        if not chunks:
            return
        self.initialize_schema(collection, len(chunks[0].embedding))
        self.upsert_batch(collection, [c.to_upsert_item() for c in chunks])

    @classmethod
    def from_config(cls, config: dict) -> "VectorDBBackend":
        raise NotImplementedError(
            "Each backend subclass must implement from_config()"
        )
