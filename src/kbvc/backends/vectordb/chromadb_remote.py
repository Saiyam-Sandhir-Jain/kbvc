# kbvc/backends/vectordb/chromadb_remote.py
"""
ChromaDB REMOTE backend — HTTP server or Chroma Cloud.

This is a SEPARATE backend from `chroma` (which is local/embedded).
Use this when you have:

  1. A self-hosted Chroma HTTP server (docker run chromadb/chroma):
         kbvc config set vectordb.backend chromadb_remote
         kbvc config set vectordb.url http://localhost:8000
         kbvc config set vectordb.collection kbvc
         # No API key needed for open HTTP server

  2. Chroma Cloud (managed cloud service at trychroma.com):
         kbvc config set vectordb.backend chromadb_remote
         kbvc config set vectordb.mode cloud
         kbvc config set vectordb.key ck-...
         kbvc config set vectordb.tenant my-tenant
         kbvc config set vectordb.database my-db
         kbvc config set vectordb.collection kbvc

The distinction from the local `chroma` backend:
  - chroma           → chromadb.PersistentClient(path=...)  — no network
  - chromadb_remote  → chromadb.HttpClient(...) or chromadb.CloudClient(...)

Install:
    pip install kbvc[chroma]    # same package, different client class
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from kbvc.backends.vectordb import VectorDBBackend, ChunkRecord


class ChromaRemoteBackend(VectorDBBackend):
    """
    Remote ChromaDB backend supporting both self-hosted HTTP and Chroma Cloud.

    Chroma Cloud is the managed cloud service at https://trychroma.com — it
    requires a tenant, database, and API key.  Self-hosted just needs a URL.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        ssl: bool = False,
        api_key: Optional[str] = None,
        tenant: Optional[str] = None,
        database: Optional[str] = None,
        mode: str = "http",  # "http" | "cloud"
    ) -> None:
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb not installed. Run: pip install kbvc[chroma]"
            )
        if mode == "cloud":
            if not api_key:
                raise ValueError(
                    "Chroma Cloud requires an API key.\n"
                    "Run: kbvc config set vectordb.key ck-..."
                )
            self._client = chromadb.CloudClient(
                tenant=tenant or "default_tenant",
                database=database or "default_database",
                api_key=api_key,
            )
        else:
            # Self-hosted HTTP server
            headers: Dict[str, str] = {}
            if api_key:
                # Some self-hosted Chroma deployments use token auth
                headers["Authorization"] = f"Bearer {api_key}"
            self._client = chromadb.HttpClient(
                host=host,
                port=port,
                ssl=ssl,
                headers=headers or None,
                tenant=tenant or "default_tenant",
                database=database or "default_database",
            )
        self._mode = mode

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
        col.upsert(
            ids=[id],
            embeddings=[vector],
            metadatas=[self._build_metadata(metadata)],
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
        self._get_collection(collection).delete(ids=[id])

    def delete_by_prefix(self, collection: str, id_prefix: str) -> None:
        col = self._get_collection(collection)
        result = col.get(include=[])
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
        count = col.count()
        if count == 0:
            return []
        results = col.query(
            query_embeddings=[vector],
            n_results=min(top_k, count),
            where=filter if filter else None,
            include=["metadatas", "distances"],
        )
        output = []
        for i, doc_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i]
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
        id_set = set(ids)
        updated_ids = []
        updated_metas = []
        for i, doc_id in enumerate(result["ids"]):
            if doc_id in id_set:
                meta = dict(result["metadatas"][i] or {})
                meta.update(patch)
                updated_ids.append(doc_id)
                updated_metas.append(self._build_metadata(meta))
        if updated_ids:
            col.update(ids=updated_ids, metadatas=updated_metas)

    def exists_batch(self, collection: str, ids: List[str]) -> Dict[str, bool]:
        col = self._get_collection(collection)
        result = col.get(ids=ids, include=[])
        found = set(result["ids"])
        return {id_: (id_ in found) for id_ in ids}

    def initialize_schema(self, collection: str, dimensions: int) -> None:
        """Idempotently create or verify the remote Chroma collection."""
        self._get_collection(collection)

    def export_chunks(self, collection: str) -> List[ChunkRecord]:
        """Export all documents as ChunkRecords.

        Note: For large collections this fetches everything in one call.
        Chroma does not support cursor-based pagination yet.
        """
        col = self._get_collection(collection)
        result = col.get(include=["embeddings", "metadatas"])
        records = []
        for i, doc_id in enumerate(result["ids"]):
            meta = (result["metadatas"] or [])[i] or {}
            embedding = (result["embeddings"] or [])[i] if result.get("embeddings") else []
            records.append(ChunkRecord(
                vector_id=doc_id,
                branch=meta.get("branch", ""),
                ko_id=meta.get("ko_id", ""),
                ko_version=int(meta.get("ko_version", 0)),
                chunk_index=int(meta.get("chunk_index", 0)),
                chunk_hash=meta.get("chunk_hash", ""),
                embedding=list(embedding),
                metadata={k: v for k, v in meta.items()
                          if k not in ("branch", "ko_id", "ko_version",
                                       "chunk_index", "chunk_hash", "created_at")},
                created_at=meta.get("created_at", ""),
            ))
        return records

    @classmethod
    def from_config(cls, config: dict) -> "ChromaRemoteBackend":
        mode = config.get("vectordb.mode", "http")
        api_key = config.get("vectordb.key", "") or None

        if mode == "cloud":
            return cls(
                mode="cloud",
                api_key=api_key,
                tenant=config.get("vectordb.tenant"),
                database=config.get("vectordb.database"),
            )

        # HTTP mode — parse host + port from vectordb.url
        url = config.get("vectordb.url", "http://localhost:8000")
        parsed = urlparse(url if "://" in url else f"http://{url}")
        host = parsed.hostname or "localhost"
        port = parsed.port or 8000
        ssl = parsed.scheme == "https"
        return cls(
            host=host,
            port=port,
            ssl=ssl,
            api_key=api_key,
            tenant=config.get("vectordb.tenant"),
            database=config.get("vectordb.database"),
            mode="http",
        )
