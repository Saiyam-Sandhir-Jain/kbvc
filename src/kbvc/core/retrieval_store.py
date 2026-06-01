# kbvc/core/retrieval_store.py
"""
RetrievalConfigStore — snapshots the active retrieval configuration per commit.

Stored at: .kbvc/retrieval/r-vN.json

Every commit records "retrieval_snapshot": "r-vN" so that kbvc checkout
can restore the full AI knowledge state: KOs + graph + prompt + retrieval config.

This is what makes a KBVC commit a *complete AI knowledge state*,
not just a file-change record.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# RetrievalSnapshot
# ---------------------------------------------------------------------------

@dataclass
class RetrievalSnapshot:
    """
    Immutable record of the retrieval configuration at commit time.

    Fields:
        version           Monotonically increasing version number.
        commit_id         Full SHA-256 commit hash (filled in after commit is created).
        created_at        ISO-8601 UTC timestamp.
        profile           "vector" | "graphrag" | "hybrid".
        hop_depth         Graph traversal hops (graphrag profile).
        embedding_model   Model identifier (e.g. "text-embedding-3-small").
        dims              Embedding dimensions.
        top_k             Number of results returned.
        weight_semantic   Blend weight for vector similarity (graphrag).
        weight_graph      Blend weight for graph proximity (graphrag).
    """
    version: int
    commit_id: str
    created_at: str
    profile: str
    hop_depth: int
    embedding_model: str
    dims: int
    top_k: int
    weight_semantic: float
    weight_graph: float


# ---------------------------------------------------------------------------
# RetrievalConfigStore
# ---------------------------------------------------------------------------

class RetrievalConfigStore:
    """
    Saves immutable retrieval config snapshots at .kbvc/retrieval/r-vN.json.
    One snapshot per commit.
    """

    def __init__(self, retrieval_dir: Path) -> None:
        self.retrieval_dir = retrieval_dir

    # ── write ─────────────────────────────────────────────────────────────────

    def snapshot(
        self,
        commit_id: str,
        config: dict,
        embed_backend: Any,  # EmbedBackend — avoid circular import
    ) -> str:
        """
        Create a new immutable snapshot from the current config + embed backend.

        Args:
            commit_id:     Full commit hash or "pending" (back-filled after commit creation).
            config:        Flat dot-key config dict from KbvcRepo.config().
            embed_backend: Active EmbedBackend instance (provides model_name + dimensions).

        Returns:
            Snapshot name, e.g. "r-v2".
        """
        version = self._next_version()
        snap = RetrievalSnapshot(
            version=version,
            commit_id=commit_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            profile=config.get("retrieval.profile", "vector"),
            hop_depth=int(config.get("retrieval.hop_depth", 2)),
            embedding_model=getattr(embed_backend, "model_name", "unknown"),
            dims=embed_backend.dimensions,
            top_k=int(config.get("retrieval.top_k", 5)),
            weight_semantic=float(config.get("retrieval.weight_semantic", 0.7)),
            weight_graph=float(config.get("retrieval.weight_graph", 0.3)),
        )
        name = f"r-v{version}"
        (self.retrieval_dir / f"{name}.json").write_text(
            json.dumps(asdict(snap), indent=2), encoding="utf-8"
        )
        return name

    # ── read ─────────────────────────────────────────────────────────────────

    def load(self, snapshot_name: str) -> RetrievalSnapshot:
        path = self.retrieval_dir / f"{snapshot_name}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Retrieval snapshot '{snapshot_name}' not found at {path}"
            )
        return RetrievalSnapshot(**json.loads(path.read_text(encoding="utf-8")))

    def latest(self) -> RetrievalSnapshot | None:
        """Return the highest-numbered snapshot, or None."""
        files = sorted(
            self.retrieval_dir.glob("r-v*.json"),
            key=lambda p: int(p.stem.replace("r-v", "")),
        )
        if not files:
            return None
        return RetrievalSnapshot(**json.loads(files[-1].read_text(encoding="utf-8")))

    # ── internals ─────────────────────────────────────────────────────────────

    def _next_version(self) -> int:
        return len(list(self.retrieval_dir.glob("r-v*.json"))) + 1
