# kbvc/core/lineage.py
"""
ChainTracer — traces a vector ID back through the commit DAG to its origin.

Pure read operation: reads only from .kbvc/ — no vector DB queries required.
This is the audit-trail feature: who embedded what, when, with which model.

Usage:
    tracer = ChainTracer(repo.kbvc_dir)
    record = tracer.trace("main__manifestai__chunk_2")
    print(record.embed_model, record.commit_message)
"""

from __future__ import annotations

import json
from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# LineageRecord
# ---------------------------------------------------------------------------

@dataclass
class LineageRecord:
    """Full audit trail for a single vector."""
    vector_id: str
    branch: str
    ko_id: str
    chunk_index: int
    section: str
    ko_version: int
    commit_id: str
    commit_message: str
    commit_timestamp: str
    embed_model: str
    embed_dims: int
    vectordb_backend: str


# ---------------------------------------------------------------------------
# ChainTracer
# ---------------------------------------------------------------------------

class ChainTracer:
    """
    Traces a vector ID back through the commit DAG to its origin.

    Vector ID format: <branch>__<ko_id>__chunk_<N>
    Example:          main__manifestai__chunk_2

    Chain:
        vector_id → branch / ko_id / chunk_index
            → ko_versions/<ko_id>/vN.json  (find which version changed this chunk)
            → commits/<commit_id>.json      (full commit metadata)
            → retrieval/r-vN.json           (embedding model + dims)
            → .kbvc/config                  (vector DB backend name)
    """

    def __init__(self, kbvc_dir: Path) -> None:
        self.kbvc_dir = kbvc_dir

    def trace(self, vector_id: str) -> LineageRecord:
        """
        Trace a vector ID to its full lineage record.

        Raises:
            ValueError       If the vector_id format is unrecognised.
            FileNotFoundError If required metadata files are absent.
        """
        branch, ko_id, chunk_index = self._parse_vector_id(vector_id)

        # Find which KO version last touched this chunk
        target_version_data, section = self._find_version_for_chunk(
            ko_id, chunk_index
        )

        commit_id: str = target_version_data["commit_id"]
        ko_ver: int = target_version_data["version"]

        # Load commit object
        commit_path = self.kbvc_dir / "commits" / f"{commit_id}.json"
        if not commit_path.exists():
            raise FileNotFoundError(
                f"Commit '{commit_id[:7]}' not found in .kbvc/commits/"
            )
        commit = json.loads(commit_path.read_text(encoding="utf-8"))

        # Load retrieval snapshot for embed model info
        r_snap_name: str = commit.get("retrieval_snapshot", "r-v1")
        retrieval = self._load_retrieval_snapshot(r_snap_name)

        # Read vectordb backend name from config
        vectordb_backend = self._read_vectordb_backend()

        return LineageRecord(
            vector_id=vector_id,
            branch=branch,
            ko_id=ko_id,
            chunk_index=chunk_index,
            section=section,
            ko_version=ko_ver,
            commit_id=commit_id,
            commit_message=commit.get("message", ""),
            commit_timestamp=commit.get("timestamp", ""),
            embed_model=retrieval.get("embedding_model", "unknown"),
            embed_dims=int(retrieval.get("dims", 0)),
            vectordb_backend=vectordb_backend,
        )

    # ── internals ─────────────────────────────────────────────────────────────

    def _parse_vector_id(self, vector_id: str) -> tuple[str, str, int]:
        """
        Parse <branch>__<ko_id>__chunk_<N> → (branch, ko_id, chunk_index).
        Double underscores are the delimiter (single underscores appear in names).
        """
        parts = vector_id.split("__")
        if len(parts) != 3 or not parts[2].startswith("chunk_"):
            raise ValueError(
                f"Unrecognised vector ID format: '{vector_id}'\n"
                "Expected: <branch>__<ko_id>__chunk_<N>"
            )
        branch = parts[0]
        ko_id = parts[1]
        try:
            chunk_index = int(parts[2].replace("chunk_", ""))
        except ValueError:
            raise ValueError(
                f"Invalid chunk index in vector ID: '{parts[2]}'"
            )
        return branch, ko_id, chunk_index

    def _find_version_for_chunk(
        self,
        ko_id: str,
        chunk_index: int,
    ) -> tuple[dict, str]:
        """
        Scan ko_versions/<ko_id>/v*.json to find which version last changed
        this chunk index. Falls back to v1 if not found in changed_chunks of any version.

        Returns (version_data_dict, section_name).
        """
        ko_versions_dir = self.kbvc_dir / "ko_versions" / ko_id
        if not ko_versions_dir.exists():
            raise FileNotFoundError(
                f"No version snapshots found for KO '{ko_id}'"
            )

        version_files = sorted(
            ko_versions_dir.glob("v*.json"),
            key=lambda p: int(p.stem[1:]),
        )
        if not version_files:
            raise FileNotFoundError(
                f"No version snapshots found for KO '{ko_id}'"
            )

        target_data: Optional[dict] = None
        section = "unknown"

        for vpath in version_files:
            v = json.loads(vpath.read_text(encoding="utf-8"))
            if chunk_index in v.get("changed_chunks", []):
                target_data = v
                # Try to get section name from chunks list (may be empty in v1)
                for chunk in v.get("chunks", []):
                    if chunk.get("index") == chunk_index:
                        section = chunk.get("section", "unknown")
                        break

        if target_data is None:
            # Chunk was present since the initial import — use v1
            target_data = json.loads(version_files[0].read_text(encoding="utf-8"))

        return target_data, section

    def _load_retrieval_snapshot(self, snapshot_name: str) -> dict:
        path = self.kbvc_dir / "retrieval" / f"{snapshot_name}.json"
        if not path.exists():
            return {}  # graceful: missing retrieval snapshot → return empty dict
        return json.loads(path.read_text(encoding="utf-8"))

    def _read_vectordb_backend(self) -> str:
        """Read the vectordb backend name from .kbvc/config."""
        cfg_path = self.kbvc_dir / "config"
        if not cfg_path.exists():
            return "unknown"
        parser = ConfigParser()
        parser.read(cfg_path)
        try:
            return parser.get("vectordb", "backend") or "unknown"
        except Exception:
            return "unknown"
