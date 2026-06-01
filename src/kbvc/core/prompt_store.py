# kbvc/core/prompt_store.py
"""
PromptVersionStore — versions retrieval and system prompts alongside commits.

Current prompt:  .kbvc/prompts/current.json  (mutable; updated by kbvc prompt set)
Snapshots:       .kbvc/prompts/p-vN.json     (immutable; one per commit)

Prompts are committed alongside KO embeddings and graph state. kbvc checkout
restores the prompt too. This is the "make KBVC famous" feature — no other
RAG system versions prompts alongside knowledge.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# PromptVersion
# ---------------------------------------------------------------------------

@dataclass
class PromptVersion:
    """
    A single version of the retrieval + system prompt pair.

    Fields:
        version           Monotonically increasing version number.
        commit_id         Full commit hash; "" while pending (pre-commit).
        created_at        ISO-8601 UTC timestamp.
        retrieval_prompt  The prompt used to guide RAG retrieval / answer generation.
        system_prompt     Optional system-level context (e.g. persona).
        notes             Free-text notes about why this prompt changed.
    """
    version: int
    commit_id: str      # "" while pending; filled by snapshot()
    created_at: str
    retrieval_prompt: str
    system_prompt: str
    notes: str


# ---------------------------------------------------------------------------
# PromptVersionStore
# ---------------------------------------------------------------------------

DEFAULT_RETRIEVAL_PROMPT = (
    "Answer using only the provided context. "
    "If the context does not contain the answer, say so explicitly."
)

DEFAULT_SYSTEM_PROMPT = ""


class PromptVersionStore:
    """
    Manages the current prompt and its immutable versioned snapshots.

    Usage:
        store = PromptVersionStore(repo.prompts_dir)
        store.set("Answer strictly from context.", system_prompt="You are Sams.")
        snap_name = store.snapshot(commit_id="pending")   # → "p-v1"
    """

    def __init__(self, prompts_dir: Path) -> None:
        self.prompts_dir = prompts_dir
        self.current_path = prompts_dir / "current.json"
        self._dirty = False

    # ── read ─────────────────────────────────────────────────────────────────

    def get_current(self) -> Optional[PromptVersion]:
        """Return the current (mutable) prompt, or None if never set."""
        if self.current_path.exists():
            return PromptVersion(**json.loads(
                self.current_path.read_text(encoding="utf-8")
            ))
        return None

    def load_snapshot(self, snapshot_name: str) -> PromptVersion:
        path = self.prompts_dir / f"{snapshot_name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Prompt snapshot '{snapshot_name}' not found")
        return PromptVersion(**json.loads(path.read_text(encoding="utf-8")))

    def log(self) -> List[PromptVersion]:
        """Return all prompt snapshots ordered oldest → newest."""
        files = sorted(
            self.prompts_dir.glob("p-v*.json"),
            key=lambda p: int(p.stem.replace("p-v", "")),
        )
        return [
            PromptVersion(**json.loads(f.read_text(encoding="utf-8")))
            for f in files
        ]

    # ── write ─────────────────────────────────────────────────────────────────

    def set(
        self,
        retrieval_prompt: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        notes: str = "",
    ) -> PromptVersion:
        """
        Record a new current prompt (pending commit).
        Version number is current_version + 1; commit_id is "" until snapshot().
        """
        current = self.get_current()
        new_version = (current.version + 1) if current else 1
        pv = PromptVersion(
            version=new_version,
            commit_id="",
            created_at=datetime.now(timezone.utc).isoformat(),
            retrieval_prompt=retrieval_prompt,
            system_prompt=system_prompt,
            notes=notes,
        )
        self.current_path.write_text(
            json.dumps(asdict(pv), indent=2), encoding="utf-8"
        )
        self._dirty = True
        return pv

    def snapshot(self, commit_id: str) -> str:
        """
        Stamp current prompt with commit_id and save as an immutable snapshot.
        If no current prompt exists, creates a default one automatically.

        BUG FIX: version is derived internally from the count of existing snapshots —
        not passed by the caller (avoids double-increment if called twice).

        Returns snapshot name e.g. "p-v1".
        """
        pv = self.get_current()
        if pv is None:
            # Auto-create default prompt (first commit without explicit kbvc prompt set)
            pv = self.set(DEFAULT_RETRIEVAL_PROMPT)

        # Derive version from existing snapshot count (not from current.version)
        # so that the file name always matches the actual count of snapshots.
        version = self._next_snapshot_version()
        pv.version = version
        pv.commit_id = commit_id

        name = f"p-v{version}"
        (self.prompts_dir / f"{name}.json").write_text(
            json.dumps(asdict(pv), indent=2), encoding="utf-8"
        )
        self._dirty = False
        return name

    def restore(self, snapshot_name: str) -> None:
        """
        Restore current.json from a named snapshot (for kbvc prompt checkout).
        The restored prompt's commit_id is cleared so it won't look like a real commit.
        """
        pv = self.load_snapshot(snapshot_name)
        pv.commit_id = ""     # pending — restored state is not committed yet
        self.current_path.write_text(
            json.dumps(asdict(pv), indent=2), encoding="utf-8"
        )
        self._dirty = False   # restored state is clean until user calls set() again

    # ── state ─────────────────────────────────────────────────────────────────

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    # ── internals ─────────────────────────────────────────────────────────────

    def _next_snapshot_version(self) -> int:
        """Count existing p-vN.json snapshots to determine next version number."""
        return len(list(self.prompts_dir.glob("p-v*.json"))) + 1
