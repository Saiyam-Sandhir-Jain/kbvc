# kbvc/backends/vectordb/chroma.py
"""
ChromaDB LOCAL backend — embedded or persistent local instance.

This backend uses chromadb.PersistentClient (or EphemeralClient for testing)
and runs entirely in-process.  No server, no API key, no network.

For a remote/server Chroma instance (self-hosted HTTP or Chroma Cloud),
use the `chromadb_remote` backend:
    kbvc config set vectordb.backend chromadb_remote

Configuration:
    kbvc config set vectordb.backend chroma
    kbvc config set vectordb.url ./chroma_db    # local directory path

Install:
    pip install kbvc[chroma]
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from kbvc.backends.vectordb import VectorDBBackend, ChunkRecord


class ChromaBackend(VectorDBBackend):
    """Local embedded ChromaDB (PersistentClient)."""

    def __init__(self, path: str = "./chroma_db") -> None:
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb not installed. Run: pip install kbvc[chroma]"
            )
        self._client = chromadb.PersistentClient(path=path)
        self._path = path

    # ── helpers ────────────────────────────────────────────────────────────────

    def _get_collection(self, collection: str):
        return self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _build_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Chroma metadata values must be str/int/float/bool — no None."""
        safe: Dict[str, Any] = {}
        for k, v in metadata.items():
            if v is None:
                safe[k] = ""
            elif isinstance(v, (str, int, float, bool)):
                safe[k] = v
            else:
                safe[k] = str(v)
        return safe

    # ── VectorDBBackend interface ──────────────────────────────────────────────

    def upsert(
        self,
        collection: str,
        id: str,
        vector: List[float],
        metadata: Dict[str, Any],
    ) -> None:
        col = self._get_collection(collection)
        safe_meta = self._build_metadata(metadata)
        col.upsert(
            ids=[id],
            embeddings=[vector],
            metadatas=[safe_meta],
            documents=[""],
        )

    def upsert_batch(self, collection: str, items: List[Dict]) -> None:
        if not items:
            return
        col = self._get_collection(collection)
        col.upsert(
            ids=[item["id"] for item in items],
            embeddings=[item["vector"] for item in items],
            metadatas=[self._build_metadata(item.get("metadata", {})) for item in items],
            documents=["" for _ in items],
        )

    def delete(self, collection: str, id: str) -> None:
        col = self._get_collection(collection)
        col.delete(ids=[id])

    def delete_by_prefix(self, collection: str, id_prefix: str) -> None:
        col = self._get_collection(collection)
        result = col.get(where_document=None)
        ids_to_delete = [i for i in result["ids"] if i.startswith(id_prefix)]
        if ids_to_delete:
            col.delete(ids=ids_to_delete)

    def query(
        self,
        collection: str,
        vector: List[float],
        top_k: int,
        filter: Optional[Dict] = None,
    ) -> List[Dict]:
        col = self._get_collection(collection)
        where = filter if filter else None
        results = col.query(
            query_embeddings=[vector],
            n_results=min(top_k, col.count() or 1),
            where=where,
            include=["metadatas", "distances"],
        )
        output = []
        for i, doc_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i]
            # Chroma cosine distance: 0=identical, 2=opposite. Convert to similarity.
            score = 1.0 - (distance / 2.0)
            meta = results["metadatas"][0][i] if results.get("metadatas") else {}
            output.append({"id": doc_id, "score": score, "metadata": meta})
        return output

    def patch_metadata(
        self,
        collection: str,
        id_prefix: str,
        patch: Dict[str, Any],
    ) -> None:
        col = self._get_collection(collection)
        result = col.get(include=["metadatas"])
        ids = [i for i in result["ids"] if i.startswith(id_prefix)]
        if not ids:
            return
        updated_metas = []
        for i, doc_id in enumerate(result["ids"]):
            if doc_id in ids:
                meta = dict(result["metadatas"][i] or {})
                meta.update(patch)
                updated_metas.append(self._build_metadata(meta))
        if ids:
            col.update(ids=ids, metadatas=updated_metas)

    def exists_batch(self, collection: str, ids: List[str]) -> Dict[str, bool]:
        col = self._get_collection(collection)
        result = col.get(ids=ids, include=[])
        found = set(result["ids"])
        return {id_: (id_ in found) for id_ in ids}

    def initialize_schema(self, collection: str, dimensions: int) -> None:
        """Idempotently create or verify the Chroma collection."""
        self._get_collection(collection)

    def export_chunks(self, collection: str) -> List[ChunkRecord]:
        """Export all documents from a Chroma collection as ChunkRecords."""
        col = self._get_collection(collection)
        result = col.get(include=["embeddings", "metadatas"])
        records = []
        for i, doc_id in enumerate(result["ids"]):
            meta = (result["metadatas"] or [])[i] or {}
            # Safe embedding extraction — handle NumPy arrays
            embedding = []
            if result.get("embeddings") is not None:
                try:
                    embedding = list(result["embeddings"][i])
                except (IndexError, TypeError):
                    pass
            records.append(ChunkRecord(
                vector_id=doc_id,
                branch=meta.get("branch", ""),
                ko_id=meta.get("ko_id", ""),
                ko_version=int(meta.get("ko_version", 0)),
                chunk_index=int(meta.get("chunk_index", 0)),
                chunk_hash=meta.get("chunk_hash", ""),
                embedding=embedding,
                metadata={k: v for k, v in meta.items()
                          if k not in ("branch", "ko_id", "ko_version",
                                       "chunk_index", "chunk_hash", "created_at")},
                created_at=meta.get("created_at", ""),
            ))
        return records

    @classmethod
    def from_config(cls, config: dict) -> "ChromaBackend":
        path = config.get("vectordb.url", "./chroma_db")
        return cls(path=path)
