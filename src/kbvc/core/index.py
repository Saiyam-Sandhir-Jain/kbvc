# kbvc/core/index.py
"""
StagingIndex — the staging area used by kbvc add and consumed by kbvc commit.

Persisted at .kbvc/index (JSON). Mirrors Git's index file in purpose:
records what will be included in the next commit.

Fields:
    staged_files  Relative source paths queued for embedding.
    ko_reasons    Per-KO free-text change reason set by kbvc annotate.
    graph_dirty   True if kbvc link was called since the last commit.
    prompt_dirty  True if kbvc prompt set was called since the last commit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List


@dataclass
class StagingIndex:
    staged_files: List[str] = field(default_factory=list)
    ko_reasons: Dict[str, str] = field(default_factory=dict)
    graph_dirty: bool = False
    prompt_dirty: bool = False

    # ── persistence ──────────────────────────────────────────────────────────

    @classmethod
    def load(cls, index_path: Path) -> StagingIndex:
        """Load from disk; return an empty index if the file is absent or corrupt."""
        if index_path.exists():
            try:
                data = json.loads(index_path.read_text(encoding="utf-8"))
                return cls(**data)
            except (json.JSONDecodeError, TypeError):
                pass   # corrupt index → return clean state
        return cls()

    def save(self, index_path: Path) -> None:
        index_path.write_text(
            json.dumps(asdict(self), indent=2), encoding="utf-8"
        )

    # ── mutations ────────────────────────────────────────────────────────────

    def stage(self, path: str) -> None:
        """Add a source path to the staging list (idempotent)."""
        if path not in self.staged_files:
            self.staged_files.append(path)

    def unstage(self, path: str) -> None:
        self.staged_files = [f for f in self.staged_files if f != path]

    def set_reason(self, ko_id: str, reason: str) -> None:
        """Set a per-KO change reason (from kbvc annotate)."""
        self.ko_reasons[ko_id] = reason

    def mark_graph_dirty(self) -> None:
        self.graph_dirty = True

    def mark_prompt_dirty(self) -> None:
        self.prompt_dirty = True

    def clear(self) -> None:
        """Reset the index after a successful commit."""
        self.staged_files.clear()
        self.ko_reasons.clear()
        self.graph_dirty = False
        self.prompt_dirty = False

    # ── state ────────────────────────────────────────────────────────────────

    @property
    def is_empty(self) -> bool:
        """True if there is nothing to commit."""
        return (
            not self.staged_files
            and not self.graph_dirty
            and not self.prompt_dirty
        )
