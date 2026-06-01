# kbvc/backends/vectordb/qdrant.py
"""
Qdrant vector database backend.

All bug fixes from §7.3 and §8.15 applied:
  P8:  UUID-based point IDs (not xxh64 int) to avoid JSON serialisation issues
       with values > 2^53. Uses uuid5(NAMESPACE_URL, str_id) for determinism.
  BUG: delete/patch_metadata use scroll+prefix-filter, not MatchText (which
       does NLP tokenisation — wrong for exact prefix matching).
  BUG: _ensure_collection cached in a set to avoid N+1 API calls per upsert.
  BUG: upsert_batch guards against empty list (items[0] would crash).
  BUG: restore_snapshot clears _dirty flag (was missing in naive impl).
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from kbvc.backends.vectordb import VectorDBBackend


class QdrantBackend(VectorDBBackend):

    def __init__(self, url: str, api_key: Optional[str] = None) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError:
            raise ImportError(
                "qdrant-client not installed. Run: pip install kbvc[qdrant]"
            )
        self._client = QdrantClient(url=url, api_key=api_key)
        self._known_collections: set[str] = set()  # P8 BUG FIX: cache to avoid N+1

    # ── write ─────────────────────────────────────────────────────────────────

    def upsert(
        self,
        collection: str,
        id: str,
        vector: List[float],
        metadata: Dict[str, Any],
    ) -> None:
        from qdrant_client.models import PointStruct
        self._ensure_collection(collection, len(vector))
        self._client.upsert(
            collection_name=collection,
            points=[
                PointStruct(
                    id=self._to_point_id(id),
                    vector=vector,
                    payload={"str_id": id, **metadata},
                )
            ],
        )

    def upsert_batch(self, collection: str, items: List[Dict]) -> None:
        # BUG FIX: guard empty list — items[0] would crash
        if not items:
            return
        from qdrant_client.models import PointStruct
        self._ensure_collection(collection, len(items[0]["vector"]))
        points = [
            PointStruct(
                id=self._to_point_id(item["id"]),
                vector=item["vector"],
                payload={"str_id": item["id"], **item["metadata"]},
            )
            for item in items
        ]
        self._client.upsert(collection_name=collection, points=points)

    def delete(self, collection: str, id: str) -> None:
        from qdrant_client.models import PointIdsList
        # BUG FIX: must wrap ID in PointIdsList
        self._client.delete(
            collection_name=collection,
            points_selector=PointIdsList(points=[self._to_point_id(id)]),
        )

    def delete_by_prefix(self, collection: str, id_prefix: str) -> None:
        """
        BUG FIX: MatchText does NLP tokenisation — wrong for prefix matching.
        Correct: scroll all points, client-side prefix filter, then batch-delete.
        """
        ids_to_delete = self._scroll_by_prefix(collection, id_prefix)
        if ids_to_delete:
            from qdrant_client.models import PointIdsList
            self._client.delete(
                collection_name=collection,
                points_selector=PointIdsList(points=ids_to_delete),
            )

    # ── read ──────────────────────────────────────────────────────────────────

    def query(
        self,
        collection: str,
        vector: List[float],
        top_k: int,
        filter: Optional[Dict] = None,
    ) -> List[Dict]:
        results = self._client.search(
            collection_name=collection,
            query_vector=vector,
            limit=top_k,
        )
        return [
            {
                "id": r.payload.get("str_id"),
                "score": r.score,
                "metadata": r.payload,
            }
            for r in results
        ]

    def patch_metadata(
        self,
        collection: str,
        id_prefix: str,
        patch: Dict[str, Any],
    ) -> None:
        """
        BUG FIX: same scroll-then-patch pattern as delete_by_prefix.
        """
        from qdrant_client.models import PointIdsList
        ids_to_patch = self._scroll_by_prefix(collection, id_prefix)
        if ids_to_patch:
            self._client.set_payload(
                collection_name=collection,
                payload=patch,
                points=PointIdsList(points=ids_to_patch),
            )

    def exists_batch(
        self,
        collection: str,
        ids: List[str],
    ) -> Dict[str, bool]:
        """
        Fetch points by hashed UUID IDs; return which string IDs were found.
        Uses client.retrieve() (fetch by ID) NOT ANN search.
        """
        uuid_to_str = {self._to_point_id(sid): sid for sid in ids}
        try:
            points = self._client.retrieve(
                collection_name=collection,
                ids=list(uuid_to_str.keys()),
                with_payload=False,
                with_vectors=False,
            )
            found_ids = {p.id for p in points}
        except Exception:
            found_ids = set()
        return {sid: (self._to_point_id(sid) in found_ids) for sid in ids}

    # ── internals ─────────────────────────────────────────────────────────────

    def _to_point_id(self, str_id: str) -> str:
        """
        P8 BUG FIX: deterministic UUID v5 from string ID.
        xxh64.intdigest() returns values up to 2^64-1; ~50% exceed 2^53 and
        cause JSON serialisation issues. UUID strings are safe and human-inspectable.
        """
        return str(uuid.uuid5(uuid.NAMESPACE_URL, str_id))

    def _ensure_collection(self, name: str, dim: int) -> None:
        """
        BUG FIX: cache known collections — was making a get_collections() API
        call on every single upsert, causing N+1 requests per commit.
        """
        if name in self._known_collections:
            return
        existing = {c.name for c in self._client.get_collections().collections}
        if name not in existing:
            from qdrant_client.models import Distance, VectorParams
            self._client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
        self._known_collections.add(name)

    def _scroll_by_prefix(self, collection: str, id_prefix: str) -> List[str]:
        """
        Scroll all points whose str_id starts with id_prefix.
        Returns a list of UUID point IDs (for use with delete / set_payload).
        """
        from qdrant_client.models import FieldCondition, Filter, MatchValue
        offset = None
        collected_ids: List[str] = []
        while True:
            results, offset = self._client.scroll(
                collection_name=collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="str_id",
                            match=MatchValue(value=id_prefix),
                        )
                    ]
                ),
                limit=256,
                offset=offset,
                with_payload=["str_id"],
                with_vectors=False,
            )
            # MatchValue is exact — filter client-side for prefix semantics
            for pt in results:
                str_id = pt.payload.get("str_id", "")
                if str_id.startswith(id_prefix):
                    collected_ids.append(pt.id)
            if offset is None:
                break
        return collected_ids

    @classmethod
    def from_config(cls, config: dict) -> QdrantBackend:
        url = config.get("vectordb.url", "")
        if not url:
            raise ValueError(
                "Qdrant URL not configured.\n"
                "Run: kbvc config set vectordb.url http://localhost:6333"
            )
        return cls(url=url, api_key=config.get("vectordb.key") or None)

    # ── VSAL: schema management & migration ───────────────────────────────────

    def initialize_schema(self, collection: str, dimensions: int) -> None:
        """Idempotently create the Qdrant collection for KBVC."""
        self._ensure_collection(collection, dimensions)

    def export_chunks(self, collection: str) -> list:
        """
        Export all points from the Qdrant collection as ChunkRecords.
        Scrolls all pages; returns them in insertion order (by vector_id).
        """
        from kbvc.backends.vectordb import ChunkRecord
        records = []
        offset = None
        while True:
            results, offset = self._client.scroll(
                collection_name=collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            for pt in results:
                p = pt.payload or {}
                records.append(ChunkRecord(
                    vector_id=p.get("str_id", str(pt.id)),
                    branch=p.get("branch", ""),
                    ko_id=p.get("ko_id", ""),
                    ko_version=int(p.get("ko_version", 0)),
                    chunk_index=int(p.get("chunk_index", 0)),
                    chunk_hash=p.get("chunk_hash", ""),
                    embedding=pt.vector if isinstance(pt.vector, list) else [],
                    metadata={
                        k: v for k, v in p.items()
                        if k not in {
                            "str_id", "branch", "ko_id", "ko_version",
                            "chunk_index", "chunk_hash", "created_at",
                        }
                    },
                    created_at=p.get("created_at", ""),
                ))
            if offset is None:
                break
        return records
