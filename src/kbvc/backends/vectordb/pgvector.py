# kbvc/backends/vectordb/pgvector.py
"""
pgvector backend — for PostgreSQL with the pgvector extension.
Supports Supabase, Neon, RDS PostgreSQL, and self-hosted postgres.
This is Saiyam's existing vector store; KBVC is a drop-in replacement for ingest.py.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from kbvc.backends.vectordb import VectorDBBackend


class PgvectorBackend(VectorDBBackend):

    def __init__(self, dsn: str) -> None:
        try:
            import psycopg2
            import pgvector.psycopg2  # noqa: F401 — registers vector type
        except ImportError:
            raise ImportError(
                "psycopg2-binary and pgvector not installed. "
                "Run: pip install kbvc[pgvector]"
            )
        self._conn = psycopg2.connect(dsn)
        from pgvector.psycopg2 import register_vector as rv
        rv(self._conn)
        self._conn.autocommit = True
        self._known_tables: set[str] = set()

    # ── write ─────────────────────────────────────────────────────────────────

    def upsert(
        self,
        collection: str,
        id: str,
        vector: List[float],
        metadata: Dict[str, Any],
    ) -> None:
        self._ensure_table(collection, len(vector))
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {collection} (str_id, embedding, metadata)
                VALUES (%s, %s, %s)
                ON CONFLICT (str_id) DO UPDATE
                  SET embedding = EXCLUDED.embedding,
                      metadata  = EXCLUDED.metadata
                """,
                (id, vector, json.dumps(metadata)),
            )

    def upsert_batch(self, collection: str, items: List[Dict]) -> None:
        if not items:
            return
        self._ensure_table(collection, len(items[0]["vector"]))
        with self._conn.cursor() as cur:
            for item in items:
                cur.execute(
                    f"""
                    INSERT INTO {collection} (str_id, embedding, metadata)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (str_id) DO UPDATE
                      SET embedding = EXCLUDED.embedding,
                          metadata  = EXCLUDED.metadata
                    """,
                    (item["id"], item["vector"], json.dumps(item["metadata"])),
                )

    def delete(self, collection: str, id: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {collection} WHERE str_id = %s", (id,)
            )

    def delete_by_prefix(self, collection: str, id_prefix: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {collection} WHERE str_id LIKE %s",
                (id_prefix + "%",),
            )

    # ── read ──────────────────────────────────────────────────────────────────

    def query(
        self,
        collection: str,
        vector: List[float],
        top_k: int,
        filter: Optional[Dict] = None,
    ) -> List[Dict]:
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT str_id, metadata,
                       1 - (embedding <=> %s::vector) AS score
                FROM {collection}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (vector, vector, top_k),
            )
            rows = cur.fetchall()
        return [
            {
                "id": row[0],
                "score": float(row[2]),
                "metadata": json.loads(row[1]) if isinstance(row[1], str) else row[1],
            }
            for row in rows
        ]

    def patch_metadata(
        self,
        collection: str,
        id_prefix: str,
        patch: Dict[str, Any],
    ) -> None:
        """
        Update metadata fields on all vectors with str_id starting with id_prefix.
        Uses jsonb merge so only the patched keys change.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE {collection}
                SET metadata = metadata || %s::jsonb
                WHERE str_id LIKE %s
                """,
                (json.dumps(patch), id_prefix + "%"),
            )

    def exists_batch(
        self,
        collection: str,
        ids: List[str],
    ) -> Dict[str, bool]:
        if not ids:
            return {}
        with self._conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(ids))
            cur.execute(
                f"SELECT str_id FROM {collection} WHERE str_id IN ({placeholders})",
                ids,
            )
            found = {row[0] for row in cur.fetchall()}
        return {sid: (sid in found) for sid in ids}

    # ── internals ─────────────────────────────────────────────────────────────

    def _ensure_table(self, name: str, dim: int) -> None:
        if name in self._known_tables:
            return
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {name} (
                    id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    str_id    TEXT UNIQUE NOT NULL,
                    embedding vector({dim}),
                    metadata  JSONB DEFAULT '{{}}'
                )
                """
            )
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {name}_embedding_idx
                ON {name} USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
                """
            )
        self._known_tables.add(name)

    @classmethod
    def from_config(cls, config: dict) -> PgvectorBackend:
        dsn = config.get("vectordb.url", "")
        if not dsn:
            raise ValueError(
                "PostgreSQL DSN not configured.\n"
                "Run: kbvc config set vectordb.url postgresql://user:pass@host/db"
            )
        return cls(dsn=dsn)

    # ── VSAL: schema management & migration ───────────────────────────────────

    def initialize_schema(self, collection: str, dimensions: int) -> None:
        """
        Idempotently create the kbvc_chunks table for KBVC.

        The VSAL canonical schema uses explicit KBVC columns rather than
        packing everything into a JSONB blob — better query performance and
        allows future `kbvc migrate schema` upgrades.
        """
        if collection in self._known_tables:
            return
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {collection} (
                    vector_id    TEXT PRIMARY KEY,
                    branch       TEXT NOT NULL DEFAULT '',
                    ko_id        TEXT NOT NULL DEFAULT '',
                    ko_version   INTEGER NOT NULL DEFAULT 0,
                    chunk_index  INTEGER NOT NULL DEFAULT 0,
                    chunk_hash   TEXT NOT NULL DEFAULT '',
                    embedding    vector({dimensions}),
                    metadata     JSONB DEFAULT '{{}}',
                    created_at   TEXT DEFAULT ''
                )
                """
            )
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {collection}_embedding_idx
                ON {collection} USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
                """
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {collection}_ko_idx ON {collection} (ko_id, branch)"
            )
        self._known_tables.add(collection)

    def export_chunks(self, collection: str) -> list:
        """Export all rows from pgvector as ChunkRecords."""
        import json as _json
        from kbvc.backends.vectordb import ChunkRecord
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT vector_id, branch, ko_id, ko_version,
                       chunk_index, chunk_hash, embedding, metadata, created_at
                FROM {collection}
                ORDER BY ko_id, chunk_index
                """
            )
            rows = cur.fetchall()
        records = []
        for row in rows:
            meta = row[7]
            if isinstance(meta, str):
                meta = _json.loads(meta)
            records.append(ChunkRecord(
                vector_id=row[0],
                branch=row[1],
                ko_id=row[2],
                ko_version=int(row[3]),
                chunk_index=int(row[4]),
                chunk_hash=row[5],
                embedding=list(row[6]) if row[6] is not None else [],
                metadata=meta or {},
                created_at=row[8] or "",
            ))
        return records
