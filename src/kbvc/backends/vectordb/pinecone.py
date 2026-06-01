# kbvc/backends/vectordb/pinecone.py
"""
Pinecone vector database backend.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from kbvc.backends.vectordb import VectorDBBackend


class PineconeBackend(VectorDBBackend):

    def __init__(self, api_key: str, index_name: str = "kbvc") -> None:
        try:
            from pinecone import Pinecone
        except ImportError:
            raise ImportError(
                "pinecone-client not installed. Run: pip install kbvc[pinecone]"
            )
        self._pc = Pinecone(api_key=api_key)
        self._index = self._pc.Index(index_name)
        self._index_name = index_name

    def upsert(self, collection: str, id: str, vector: List[float], metadata: Dict[str, Any]) -> None:
        # Pinecone uses namespaces for collections
        self._index.upsert(
            vectors=[{"id": id, "values": vector, "metadata": metadata}],
            namespace=collection,
        )

    def upsert_batch(self, collection: str, items: List[Dict]) -> None:
        if not items:
            return
        vectors = [
            {"id": item["id"], "values": item["vector"], "metadata": item["metadata"]}
            for item in items
        ]
        # Pinecone recommends batches of ≤100
        BATCH = 100
        for i in range(0, len(vectors), BATCH):
            self._index.upsert(vectors=vectors[i:i+BATCH], namespace=collection)

    def delete(self, collection: str, id: str) -> None:
        self._index.delete(ids=[id], namespace=collection)

    def delete_by_prefix(self, collection: str, id_prefix: str) -> None:
        # Pinecone supports prefix deletion on serverless indexes
        self._index.delete(prefix=id_prefix, namespace=collection)

    def query(self, collection: str, vector: List[float], top_k: int,
              filter: Optional[Dict] = None) -> List[Dict]:
        results = self._index.query(
            vector=vector,
            top_k=top_k,
            namespace=collection,
            include_metadata=True,
            filter=filter,
        )
        return [
            {"id": m.id, "score": m.score, "metadata": m.metadata or {}}
            for m in results.matches
        ]

    def patch_metadata(self, collection: str, id_prefix: str, patch: Dict[str, Any]) -> None:
        # Pinecone requires ID list for metadata update — query by prefix first
        results = self._index.query(
            vector=[0.0] * 1,  # dummy — use list mode
            top_k=10000,
            namespace=collection,
            filter={"str_id": {"$regex": f"^{id_prefix}"}},
            include_metadata=False,
        )
        ids = [m.id for m in results.matches if m.id.startswith(id_prefix)]
        for id_ in ids:
            self._index.update(id=id_, set_metadata=patch, namespace=collection)

    def exists_batch(self, collection: str, ids: List[str]) -> Dict[str, bool]:
        results = self._index.fetch(ids=ids, namespace=collection)
        found = set(results.vectors.keys())
        return {id_: (id_ in found) for id_ in ids}

    @classmethod
    def from_config(cls, config: dict) -> PineconeBackend:
        api_key = config.get("vectordb.key", "")
        if not api_key:
            raise ValueError(
                "Pinecone API key not configured.\n"
                "Run: kbvc config set vectordb.key pc-..."
            )
        index = config.get("vectordb.collection", "kbvc")
        return cls(api_key=api_key, index_name=index)
