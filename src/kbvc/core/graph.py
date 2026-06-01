# kbvc/core/graph.py
"""
RelationGraph — working graph and versioned snapshots.

Working state:   .kbvc/graph/current.json  (mutable; updated by kbvc link)
Snapshots:       .kbvc/graph/graph-vN.json  (immutable; created per commit)
Delta snapshots: v3 scope — spec in §3.4C; keys accepted as no-ops in v1.

Relations are stored separately from KO content. kbvc link NEVER triggers
re-embedding. Two branches have independent graph state.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Relation dataclass
# ---------------------------------------------------------------------------

@dataclass
class Relation:
    """
    An edge in the knowledge graph connecting two KOs (or entities in v2).

    Fields:
        id         Unique relation ID (e.g. "rel-a1b2c3d4").
        from_id    Source ko_id or entity_id.
        to_id      Target ko_id or entity_id.
        type       Relation type from the standard taxonomy (§6 kbvc link).
        note       Human-readable description of the relation.
        level      "ko" | "entity" — entity level is v2; always "ko" in v1.
        created    ISO date when this relation was created.
        branch     Branch on which this relation lives.
        valid_from ISO date: when this relation became true in the world (optional).
        valid_to   ISO date: when this relation ceased to be true (optional).
        confidence Float 0.0–1.0 expressing certainty of the relation.
                   1.0 = human-verified; lower values = auto-discovered.
        source     Origin of the relation: "human" | "auto" | "agent".
                   Defaults to "human" so existing relations are unaffected.
    """
    id: str
    from_id: str
    to_id: str
    type: str
    note: str
    level: str        # "ko" | "entity"
    created: str
    branch: str
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    # Knowledge confidence (VSAL phase — safe defaults preserve backward compat)
    confidence: float = 1.0   # 0.0–1.0; 1.0 = fully trusted
    source: str = "human"     # "human" | "auto" | "agent"


# ---------------------------------------------------------------------------
# Standard relation type taxonomy (informational — not enforced in v1)
# ---------------------------------------------------------------------------

STANDARD_RELATION_TYPES = {
    # Temporal
    "developed_during", "studied_at", "worked_at",
    # Causal / Influence
    "informed_by", "influenced",
    # Hierarchical
    "extends", "part_of",
    # Reference
    "created_at", "cites", "contradicts",
    # Ownership / Membership
    "used_in",
    # Knowledge provenance (VSAL phase)
    "supersedes",     # A supersedes B — B is deprecated by A
    "supported_by",   # Claim A is supported_by evidence B
}


# ---------------------------------------------------------------------------
# RelationGraph
# ---------------------------------------------------------------------------

class RelationGraph:
    """
    Manages the working relation graph and its versioned snapshots.

    Usage:
        graph = RelationGraph(repo.graph_dir, branch="main")
        graph.add("manifestai", "wsd-paper", "informed_by", note="...")
        snapshot_name = graph.snapshot(version=2)   # creates graph-v2.json
        graph.restore_snapshot("graph-v1")           # roll back to graph-v1
    """

    def __init__(self, graph_dir: Path, branch: str) -> None:
        self.graph_dir = graph_dir
        self.branch = branch
        self.current_path = graph_dir / "current.json"
        self.relations: List[Relation] = self._load_current()
        self._dirty = False

    # ── loading ───────────────────────────────────────────────────────────────

    def _load_current(self) -> List[Relation]:
        if self.current_path.exists():
            raw = self.current_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            return [Relation(**r) for r in data]
        return []

    # ── mutations ─────────────────────────────────────────────────────────────

    def add(
        self,
        from_id: str,
        to_id: str,
        rel_type: str,
        note: str = "",
        level: str = "ko",
        valid_from: Optional[str] = None,
        valid_to: Optional[str] = None,
    ) -> Relation:
        """Add a new relation and persist current.json."""
        rel = Relation(
            id=f"rel-{uuid.uuid4().hex[:8]}",
            from_id=from_id,
            to_id=to_id,
            type=rel_type,
            note=note,
            level=level,
            created=date.today().isoformat(),
            branch=self.branch,
            valid_from=valid_from,
            valid_to=valid_to,
        )
        self.relations.append(rel)
        self._dirty = True
        self._save_current()
        return rel

    def remove(self, rel_id: str) -> bool:
        """Remove a relation by ID. Returns True if found and removed."""
        before = len(self.relations)
        self.relations = [r for r in self.relations if r.id != rel_id]
        if len(self.relations) < before:
            self._dirty = True
            self._save_current()
            return True
        return False

    # ── traversal ─────────────────────────────────────────────────────────────

    def neighbors(
        self,
        ko_id: str,
        depth: int = 1,
        rel_type: Optional[str] = None,
    ) -> List[dict]:
        """
        BFS traversal up to `depth` hops from `ko_id`.
        Returns a list of {relation: <dict>, direction: "incoming"|"outgoing"}.
        """
        visited = {ko_id}
        frontier = {ko_id}
        result: List[dict] = []

        for _ in range(depth):
            next_frontier: set = set()
            for node in frontier:
                for r in self.relations:
                    if rel_type and r.type != rel_type:
                        continue
                    other: Optional[str] = None
                    direction: Optional[str] = None
                    if r.from_id == node:
                        other, direction = r.to_id, "outgoing"
                    elif r.to_id == node:
                        other, direction = r.from_id, "incoming"
                    if other and other not in visited:
                        result.append({
                            "relation": asdict(r),
                            "direction": direction,
                        })
                        next_frontier.add(other)
                        visited.add(other)
            frontier = next_frontier

        return result

    def list_relations(self, ko_id: Optional[str] = None) -> List[Relation]:
        """Return all relations, optionally filtered to those involving ko_id."""
        if ko_id is None:
            return list(self.relations)
        return [
            r for r in self.relations
            if r.from_id == ko_id or r.to_id == ko_id
        ]

    # ── snapshots ─────────────────────────────────────────────────────────────

    def snapshot(self, version: int) -> str:
        """
        Write an immutable snapshot of the current relations.
        Snapshot name: "graph-vN".
        Returns the snapshot name.
        """
        name = f"graph-v{version}"
        snapshot_path = self.graph_dir / f"{name}.json"
        snapshot_path.write_text(
            json.dumps([asdict(r) for r in self.relations], indent=2),
            encoding="utf-8",
        )
        self._dirty = False   # snapshot == current state; nothing pending
        return name

    def restore_snapshot(self, snapshot_name: str) -> None:
        """
        Restore working state from a named snapshot.
        BUG FIX: _dirty must be cleared — restore is not a pending change.
        """
        snapshot_path = self.graph_dir / f"{snapshot_name}.json"
        if not snapshot_path.exists():
            raise FileNotFoundError(
                f"Graph snapshot '{snapshot_name}' not found at {snapshot_path}"
            )
        data = json.loads(snapshot_path.read_text(encoding="utf-8"))
        self.relations = [Relation(**r) for r in data]
        self._dirty = False    # ← BUG FIX: was missing in naive impl
        self._save_current()

    def next_version(self) -> int:
        """Compute the next snapshot version number from existing files."""
        existing = list(self.graph_dir.glob("graph-v*.json"))
        # Filter out current.json (pattern "graph-v*.json" won't match it anyway)
        versions = []
        for p in existing:
            try:
                versions.append(int(p.stem.replace("graph-v", "")))
            except ValueError:
                pass
        return max(versions, default=0) + 1

    # ── state ─────────────────────────────────────────────────────────────────

    @property
    def is_dirty(self) -> bool:
        """True if relations have been modified since the last snapshot."""
        return self._dirty

    def mark_dirty(self) -> None:
        """Explicitly mark the graph as needing a snapshot (e.g. after kbvc link)."""
        self._dirty = True

    # ── persistence ───────────────────────────────────────────────────────────

    def _save_current(self) -> None:
        self.current_path.write_text(
            json.dumps([asdict(r) for r in self.relations], indent=2),
            encoding="utf-8",
        )
