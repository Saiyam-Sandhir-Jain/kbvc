# kbvc/backends/vectordb/lancedb.py
"""
LanceDB vector backend — embedded, serverless, zero external-process required.

LanceDB stores everything in a local directory (or S3/GCS path), making it
ideal for local development, laptops, and single-machine deployments where
you don't want to run a separate Qdrant/Postgres server.

Configuration:
    kbvc config set vectordb.backend lancedb
    kbvc config set vectordb.url ./kbvc_lance        # local path (default)
    # Or cloud:
    kbvc config set vectordb.url s3://my-bucket/kbvc

Install:
    pip install kbvc[lancedb]
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from kbvc.backends.vectordb import VectorDBBackend, ChunkRecord

_SCHEMA_CACHE: Dict[str, Any] = {}


class LanceDBBackend(VectorDBBackend):
    """LanceDB-backed VSAL adapter.

    All KBVC chunk data is stored in a single LanceDB table per collection
    (e.g. ``kbvc``).  Each row corresponds to one ChunkRecord.

    LanceDB uses columnar Arrow storage, so bulk exports and filters are
    very fast even on millions of rows.
    """

    def __init__(self, uri: str = "./kbvc_lance") -> None:
        try:
            import lancedb  # noqa: F401
        except ImportError:
            raise ImportError(
                "lancedb not installed. Run: pip install kbvc[lancedb]"
            )
        import lancedb as _lancedb
        self._db = _lancedb.connect(uri)
        self._uri = uri

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_table(self, collection: str):
        """Return the LanceDB table, creating it if necessary."""
        try:
            return self._db.open_table(collection)
        except Exception:
            return None  # table does not exist yet

    def _ensure_table(self, collection: str, vector_dim: int):
        """Create the table if it doesn't exist, return it."""
        tbl = self._get_table(collection)
        if tbl is not None:
            return tbl
        import pyarrow as pa

        schema = pa.schema([
            pa.field("vector_id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), vector_dim)),
            pa.field("branch", pa.string()),
            pa.field("ko_id", pa.string()),
            pa.field("ko_version", pa.int32()),
            pa.field("chunk_index", pa.int32()),
            pa.field("chunk_hash", pa.string()),
            pa.field("commit_id", pa.string()),
            pa.field("created_at", pa.string()),
        ])
        tbl = self._db.create_table(collection, schema=schema, exist_ok=True)
        return tbl

    # ------------------------------------------------------------------
    # VectorDBBackend interface
    # ------------------------------------------------------------------

    def upsert(
        self,
        collection: str,
        id: str,
        vector: List[float],
        metadata: Dict[str, Any],
    ) -> None:
        self.upsert_batch(collection, [{"id": id, "vector": vector, "metadata": metadata}])

    def upsert_batch(self, collection: str, items: List[Dict]) -> None:
        if not items:
            return
        vector_dim = len(items[0]["vector"])
        tbl = self._ensure_table(collection, vector_dim)

        rows = []
        for item in items:
            meta = item.get("metadata", {})
            rows.append({
                "vector_id": item["id"],
                "vector": [float(v) for v in item["vector"]],
                "branch": meta.get("branch", ""),
                "ko_id": meta.get("ko_id", ""),
                "ko_version": int(meta.get("ko_version", 0)),
                "chunk_index": int(meta.get("chunk_index", 0)),
                "chunk_hash": meta.get("chunk_hash", ""),
                "commit_id": meta.get("commit_id", ""),
                "created_at": meta.get("created_at", ""),
            })
        # Merge (upsert): delete existing IDs then insert fresh rows
        ids = [r["vector_id"] for r in rows]
        id_list = ", ".join(f"'{i}'" for i in ids)
        try:
            tbl.delete(f"vector_id IN ({id_list})")
        except Exception:
            pass
        tbl.add(rows)

    def delete(self, collection: str, id: str) -> None:
        tbl = self._get_table(collection)
        if tbl is None:
            return
        tbl.delete(f"vector_id = '{id}'")

    def delete_by_prefix(self, collection: str, id_prefix: str) -> None:
        tbl = self._get_table(collection)
        if tbl is None:
            return
        # LanceDB SQL: string starts-with via LIKE
        safe_prefix = id_prefix.replace("'", "''")
        tbl.delete(f"vector_id LIKE '{safe_prefix}%'")

    def query(
        self,
        collection: str,
        vector: List[float],
        top_k: int,
        filter: Optional[Dict] = None,
    ) -> List[Dict]:
        tbl = self._get_table(collection)
        if tbl is None:
            return []
        q = tbl.search(vector).limit(top_k).metric("cosine")
        if filter:
            # Build a simple SQL WHERE clause from the filter dict
            clauses = []
            for k, v in filter.items():
                safe_v = str(v).replace("'", "''")
                clauses.append(f"{k} = '{safe_v}'")
            if clauses:
                q = q.where(" AND ".join(clauses))
        results = q.to_list()
        output = []
        for row in results:
            meta = {
                "branch": row.get("branch", ""),
                "ko_id": row.get("ko_id", ""),
                "ko_version": row.get("ko_version", 0),
                "chunk_index": row.get("chunk_index", 0),
                "chunk_hash": row.get("chunk_hash", ""),
                "commit_id": row.get("commit_id", ""),
                "created_at": row.get("created_at", ""),
            }
            output.append({
                "id": row["vector_id"],
                "score": 1.0 - float(row.get("_distance", 0.0)),
                "metadata": meta,
            })
        return output

    def patch_metadata(
        self,
        collection: str,
        id_prefix: str,
        patch: Dict[str, Any],
    ) -> None:
        tbl = self._get_table(collection)
        if tbl is None:
            return
        safe_prefix = id_prefix.replace("'", "''")
        existing = (
            tbl.search()
               .where(f"vector_id LIKE '{safe_prefix}%'")
               .to_list()
        )
        if not existing:
            return
        # Re-upsert with patched metadata
        updated = []
        for row in existing:
            updated_row = dict(row)
            for k, v in patch.items():
                if k in updated_row:
                    updated_row[k] = v
            updated.append(updated_row)
        ids = [r["vector_id"] for r in updated]
        id_list = ", ".join(f"'{i}'" for i in ids)
        tbl.delete(f"vector_id IN ({id_list})")
        tbl.add(updated)

    def exists_batch(self, collection: str, ids: List[str]) -> Dict[str, bool]:
        tbl = self._get_table(collection)
        if tbl is None:
            return {id_: False for id_ in ids}
        id_list = ", ".join(f"'{i}'" for i in ids)
        results = (
            tbl.search()
               .where(f"vector_id IN ({id_list})")
               .select(["vector_id"])
               .to_list()
        )
        found = {row["vector_id"] for row in results}
        return {id_: (id_ in found) for id_ in ids}

    def initialize_schema(self, collection: str, dimensions: int) -> None:
        """Idempotently create the LanceDB table for KBVC chunks."""
        self._ensure_table(collection, dimensions)

    def export_chunks(self, collection: str) -> List[ChunkRecord]:
        """Export all rows from the LanceDB table as ChunkRecords."""
        tbl = self._get_table(collection)
        if tbl is None:
            return []
        rows = tbl.to_pandas().to_dict(orient="records")
        records = []
        for row in rows:
            records.append(ChunkRecord(
                vector_id=str(row["vector_id"]),
                branch=str(row.get("branch", "")),
                ko_id=str(row.get("ko_id", "")),
                ko_version=int(row.get("ko_version", 0)),
                chunk_index=int(row.get("chunk_index", 0)),
                chunk_hash=str(row.get("chunk_hash", "")),
                embedding=list(row.get("vector", [])),
                metadata={
                    "commit_id": str(row.get("commit_id", "")),
                },
                created_at=str(row.get("created_at", "")),
            ))
        return records

    @classmethod
    def from_config(cls, config: dict) -> "LanceDBBackend":
        uri = config.get("vectordb.url", "./kbvc_lance")
        return cls(uri=uri)
