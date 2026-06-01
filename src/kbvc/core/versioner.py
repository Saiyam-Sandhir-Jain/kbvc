# kbvc/core/versioner.py
"""
KOVersioner — immutable per-KO version snapshots.

Every kbvc commit that touches a KO writes a snapshot at:
    .kbvc/ko_versions/<ko_id>/vN.json

These snapshots are the basis for:
  - kbvc history <file>   (per-KO audit trail with reasons)
  - kbvc checkout         (restore a specific KO version)
  - kbvc trace            (vector lineage)
  - kbvc diff             (chunk-level diff between commits)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# KOVersionSnapshot
# ---------------------------------------------------------------------------

@dataclass
class KOVersionSnapshot:
    """
    Immutable record of a KO's state at a specific commit.

    Fields:
        version          KO semantic version number (monotonically increasing).
        commit_id        Full SHA-256 hash of the commit that created this version.
        reason           Human-readable change reason (from kbvc annotate; "" if absent).
        frontmatter      Full frontmatter dict at commit time.
        chunks           List of {index, section, text, hash} dicts.
                         NOTE: In v1 commit path this is left empty for performance —
                         kbvc trace reads the source file directly. Set it when you
                         need the full chunk text (e.g. kbvc diff).
        vector_ids       Vector IDs upserted to the DB for this version.
        changed_chunks   Indices that were re-embedded in this version (cost audit).
        deleted_chunks   Indices that were removed vs the prior version.
    """
    version: int
    commit_id: str
    reason: str
    frontmatter: dict
    chunks: List[dict]         # [{index, section, text, hash}]
    vector_ids: List[str]
    changed_chunks: List[int]
    deleted_chunks: List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# KOVersioner
# ---------------------------------------------------------------------------

class KOVersioner:
    """
    Manages immutable per-KO version snapshots stored under
    .kbvc/ko_versions/<ko_id>/vN.json.
    """

    def __init__(self, ko_versions_dir: Path) -> None:
        self.base_dir = ko_versions_dir

    def _dir(self, ko_id: str) -> Path:
        d = self.base_dir / ko_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ── write ─────────────────────────────────────────────────────────────────

    def save_version(self, ko_id: str, snapshot: KOVersionSnapshot) -> Path:
        """Persist a snapshot. Raises if the file already exists (immutability guard)."""
        path = self._dir(ko_id) / f"v{snapshot.version}.json"
        if path.exists():
            raise FileExistsError(
                f"KO version snapshot already exists: {path}. "
                "Version snapshots are immutable."
            )
        data = {
            "version": snapshot.version,
            "commit_id": snapshot.commit_id,
            "reason": snapshot.reason,
            "frontmatter": snapshot.frontmatter,
            "chunks": snapshot.chunks,
            "vector_ids": snapshot.vector_ids,
            "changed_chunks": snapshot.changed_chunks,
            "deleted_chunks": snapshot.deleted_chunks,
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return path

    # ── read ─────────────────────────────────────────────────────────────────

    def load_version(self, ko_id: str, version: int) -> KOVersionSnapshot:
        path = self._dir(ko_id) / f"v{version}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"KO version snapshot not found: {path}"
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        return KOVersionSnapshot(**data)

    def latest_version(self, ko_id: str) -> Optional[KOVersionSnapshot]:
        """Return the highest-numbered version snapshot, or None if none exist."""
        d = self.base_dir / ko_id
        if not d.exists():
            return None
        version_files = sorted(
            d.glob("v*.json"),
            key=lambda p: int(p.stem[1:]),
        )
        if not version_files:
            return None
        data = json.loads(version_files[-1].read_text(encoding="utf-8"))
        return KOVersionSnapshot(**data)

    def all_versions(self, ko_id: str) -> List[KOVersionSnapshot]:
        """Return all snapshots for a KO, ordered oldest → newest."""
        d = self.base_dir / ko_id
        if not d.exists():
            return []
        return [
            KOVersionSnapshot(**json.loads(p.read_text(encoding="utf-8")))
            for p in sorted(d.glob("v*.json"), key=lambda p: int(p.stem[1:]))
        ]

    def exists(self, ko_id: str, version: int) -> bool:
        return (self.base_dir / ko_id / f"v{version}.json").exists()
