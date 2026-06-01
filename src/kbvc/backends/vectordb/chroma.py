# kbvc/backends/vectordb/chroma.py
"""
ChromaDB backend — local-first vector database (good for development).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from kbvc.backends.vectordb import VectorDBBackend


class ChromaBackend(VectorDBBackend):

    def __init__(self, path: str = "./chroma_db", collection_prefix: str = "kbvc") -> None:
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb not installed. Run: pip install kbvc[chroma]"
            )
        self._client = chromadb.PersistentClient(path=path)
        self._collections: Dict[str, Any] = {}

    def _get_collection(self, name: str) -> Any:
        if name not in self._collections:
            self._collections[name] = self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[name]

    def upsert(self, collection: str, id: str, vector: List[float], metadata: Dict[str, Any]) -> None:
        col = self._get_collection(collection)
        col.upsert(ids=[id], embeddings=[vector], metadatas=[metadata])

    def upsert_batch(self, collection: str, items: List[Dict]) -> None:
        if not items:
            return
        col = self._get_collection(collection)
        col.upsert(
            ids=[item["id"] for item in items],
            embeddings=[item["vector"] for item in items],
            metadatas=[item["metadata"] for item in items],
        )

    def delete(self, collection: str, id: str) -> None:
        self._get_collection(collection).delete(ids=[id])

    def delete_by_prefix(self, collection: str, id_prefix: str) -> None:
        col = self._get_collection(collection)
        # Chroma supports where clause on metadata
        results = col.get(where={"str_id": {"$startswith": id_prefix}})  # type: ignore
        if results and results.get("ids"):
            col.delete(ids=results["ids"])

    def query(self, collection: str, vector: List[float], top_k: int,
              filter: Optional[Dict] = None) -> List[Dict]:
        col = self._get_collection(collection)
        results = col.query(query_embeddings=[vector], n_results=top_k)
        output = []
        for i, id_ in enumerate(results["ids"][0]):
            output.append({
                "id": id_,
                "score": 1 - results["distances"][0][i],
                "metadata": results["metadatas"][0][i],
            })
        return output

    def patch_metadata(self, collection: str, id_prefix: str, patch: Dict[str, Any]) -> None:
        col = self._get_collection(collection)
        results = col.get(where={"str_id": {"$startswith": id_prefix}})  # type: ignore
        if results and results.get("ids"):
            updated_metas = []
            for meta in results["metadatas"]:
                updated = {**meta, **patch}
                updated_metas.append(updated)
            col.update(ids=results["ids"], metadatas=updated_metas)

    def exists_batch(self, collection: str, ids: List[str]) -> Dict[str, bool]:
        col = self._get_collection(collection)
        results = col.get(ids=ids)
        found = set(results.get("ids", []))
        return {id_: (id_ in found) for id_ in ids}

    @classmethod
    def from_config(cls, config: dict) -> ChromaBackend:
        path = config.get("vectordb.url", "./chroma_db")
        return cls(path=path)
