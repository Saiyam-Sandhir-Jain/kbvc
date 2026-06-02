# kbvc/core/commit.py
"""
CommitObject — global snapshot of everything that changed together.

A KBVC commit spans ALL KOs + graph + prompt + retrieval config changed
in one operation. This is NOT per-KO; it is the root of the commit DAG.

Commit ID: full SHA-256 of the semantic content (parent + branch + message
+ sorted KO changes + graph/prompt/retrieval snapshot names).
Timestamp is stored but NOT hashed (mirrors git: same content → same hash).

Display: 7-char prefix for CLI output only; full 64-char hash stored on disk.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# KOChange — per-KO diff summary embedded in a CommitObject
# ---------------------------------------------------------------------------

@dataclass
class KOChange:
    from_version: int
    to_version: int
    chunks_reembedded: List[int]
    reason: str = ""   # from kbvc annotate; empty string if not provided
    chunks: List[dict] = None  # [{index, section}] — for version snapshot; not in commit hash

    def __post_init__(self):
        if self.chunks is None:
            self.chunks = []


# ---------------------------------------------------------------------------
# CommitObject
# ---------------------------------------------------------------------------

@dataclass
class CommitObject:
    """
    Global KBVC commit object.

    Fields:
        commit_id          Full 64-char SHA-256 hex string (never truncate for storage).
        parent             Full commit_id of the parent commit, or None for root.
        branch             Branch name at time of commit.
        timestamp          ISO-8601 UTC timestamp (stored, not hashed).
        message            User-supplied commit message.
        changed_kos        Dict of ko_id → KOChange for all KOs modified in this commit.
        graph_snapshot     Name of the graph snapshot file (e.g. "graph-v4").
        prompt_snapshot    Name of the prompt snapshot file (e.g. "p-v1").
        retrieval_snapshot Name of the retrieval config snapshot (e.g. "r-v1").
    """

    commit_id: str
    parent: Optional[str]
    branch: str
    timestamp: str
    message: str
    changed_kos: Dict[str, KOChange]
    graph_snapshot: str
    prompt_snapshot: str
    retrieval_snapshot: str

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, commits_dir: Path) -> None:
        """Write commit object to .kbvc/commits/<full_hash>.json."""
        data = asdict(self)
        path = commits_dir / f"{self.commit_id}.json"
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, commits_dir: Path, commit_id: str) -> CommitObject:
        """
        Load by full or abbreviated hash (min 4 chars).
        Raises FileNotFoundError if no match.
        Raises ValueError if the prefix is ambiguous.
        """
        # Try exact match first (fast path for full hashes)
        exact = commits_dir / f"{commit_id}.json"
        if exact.exists():
            return cls._from_path(exact)

        # Prefix search
        matches = [
            p for p in commits_dir.glob("*.json")
            if p.stem.startswith(commit_id)
        ]
        if len(matches) == 1:
            return cls._from_path(matches[0])
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous commit prefix '{commit_id}': "
                + ", ".join(p.stem[:7] for p in matches)
            )
        raise FileNotFoundError(f"No commit matching '{commit_id}'")

    @classmethod
    def _from_path(cls, path: Path) -> CommitObject:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["changed_kos"] = {
            k: KOChange(**v) for k, v in data["changed_kos"].items()
        }
        return cls(**data)

    # ── factory ───────────────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        parent: Optional[str],
        branch: str,
        message: str,
        changed_kos: Dict[str, KOChange],
        graph_snapshot: str,
        prompt_snapshot: str,
        retrieval_snapshot: str,
    ) -> CommitObject:
        """
        Create a new CommitObject with a deterministic SHA-256 commit_id.

        Determinism rules (v9 §3.2 / key design decision #18):
        - Timestamp is NOT included in the hash — only stored for display.
        - changed_kos is sorted by ko_id for cross-platform consistency.
        - chunk indices are sorted before hashing.

        This means: same staged content committed twice → same hash.
        (Different message or branch → different hash, as expected.)
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        ko_summary = "".join(
            f"{ko_id}:{v.from_version}:{v.to_version}"
            f":{':'.join(map(str, sorted(v.chunks_reembedded)))}"
            for ko_id, v in sorted(changed_kos.items())
        )

        content = (
            f"{parent or ''}{branch}{message}"
            f"{graph_snapshot}{prompt_snapshot}{retrieval_snapshot}"
            f"{ko_summary}"
        )

        commit_id = hashlib.sha256(content.encode("utf-8")).hexdigest()

        return cls(
            commit_id=commit_id,
            parent=parent,
            branch=branch,
            timestamp=timestamp,
            message=message,
            changed_kos=changed_kos,
            graph_snapshot=graph_snapshot,
            prompt_snapshot=prompt_snapshot,
            retrieval_snapshot=retrieval_snapshot,
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    @property
    def display_id(self) -> str:
        """7-character display abbreviation (CLI output only — never store this)."""
        return self.commit_id[:7]


# ---------------------------------------------------------------------------
# DAG traversal helper
# ---------------------------------------------------------------------------

def walk_dag(commits_dir: Path, head_commit_id: Optional[str]) -> List[CommitObject]:
    """
    Walk the commit DAG from HEAD to root (newest → oldest).
    Safer than sorting by mtime (P3 from §8.15).
    Returns empty list if head_commit_id is None.
    """
    if not head_commit_id:
        return []
    result: List[CommitObject] = []
    current_id: Optional[str] = head_commit_id
    seen: set = set()  # cycle guard (shouldn't happen, but defensive)
    while current_id and current_id not in seen:
        seen.add(current_id)
        try:
            commit = CommitObject.load(commits_dir, current_id)
        except FileNotFoundError:
            break
        result.append(commit)
        current_id = commit.parent
    return result
