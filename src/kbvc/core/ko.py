# kbvc/core/ko.py
"""
KnowledgeObject dataclass and KOStore (the flat-file registry).

v1:  one file → one KO; source_files is always [].
v2:  per-KO files at .kbvc/kos/<ko_id>.json replace ko_store.json.
v3:  multi-source KOs (source_files populated).

All v2/v3 fields are present in the v1 dataclass so no schema
migration is required when those releases ship.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Entity (v2 — populated by kbvc extract; always [] in v1)
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    """Fine-grained entity extracted from a KO chunk (v2).

    In v1 every KO's `entities` list is empty. The field is here so v2
    can populate it without any schema migration.
    """
    id: str            # e.g. "ent-transformers"
    name: str          # e.g. "Transformers"
    type: str          # "model" | "concept" | "person" | "institution"
    ko_id: str         # which KO this was extracted from
    chunk_index: int   # which chunk it appears in


# ---------------------------------------------------------------------------
# KnowledgeObject
# ---------------------------------------------------------------------------

@dataclass
class KnowledgeObject:
    """
    Versioned unit of knowledge derived from one or more sources.

    Stored per-entry inside ko_store.json (v1) or .kbvc/kos/<id>.json (v2+).
    """
    id: str
    source_type: str              # "file" | "image" | "audio" | "video" | "api" | "web" | "memory"
    path: str                     # file path or URL
    type: str                     # from frontmatter: "project" | "education" | "patent" | ...
    tags: List[str]
    volatility: str               # "frozen" | "slow" | "live"
    version: int                  # semantic version — incremented on each kbvc commit
    last_updated: str             # ISO date of last kbvc commit touching this KO
    chunk_hashes: List[str]       # xxhash hex of each chunk — drives diff-only re-embed
    vector_ids: List[str]         # e.g. ["main__manifestai__chunk_0", ...]
    branch: str

    # Optional temporal validity (§3.3)
    valid_from: Optional[str] = None   # ISO date: when this KO's content became true
    valid_to: Optional[str] = None     # ISO date: when this KO's content ceased to be true

    # v3: multi-file KO support (always [] in v1 — do not implement logic in v1)
    source_files: List[str] = field(default_factory=list)

    # cache-invalidation dependency graph (§3.3 / §3.13)
    depends_on: List[str] = field(default_factory=list)

    # v2: entity extraction (always [] in v1)
    entities: List[Entity] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> KnowledgeObject:
        d = dict(d)
        # Deserialise nested Entity objects
        d["entities"] = [Entity(**e) for e in d.get("entities", [])]
        return cls(**d)


# ---------------------------------------------------------------------------
# KOStore — flat-file registry
# ---------------------------------------------------------------------------

class KOStore:
    """
    Loads and saves ko_store.json.

    v1: single flat JSON file at .kbvc/ko_store.json (list of KO dicts).
    v2: migrates to per-KO files at .kbvc/kos/<ko_id>.json (same schema).

    All mutations must go through add() / update() / remove() —
    never write to the file directly.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._kos: Dict[str, KnowledgeObject] = {}
        if path.exists():
            raw = path.read_text(encoding="utf-8")
            for entry in json.loads(raw):
                ko = KnowledgeObject.from_dict(entry)
                self._kos[ko.id] = ko

    # ── read ─────────────────────────────────────────────────────────────────

    def get(self, ko_id: str) -> Optional[KnowledgeObject]:
        return self._kos.get(ko_id)

    def all(self) -> List[KnowledgeObject]:
        return list(self._kos.values())

    def __len__(self) -> int:
        return len(self._kos)

    # ── write ─────────────────────────────────────────────────────────────────

    def add(self, ko: KnowledgeObject) -> None:
        if ko.id in self._kos:
            raise KeyError(
                f"KO '{ko.id}' already exists — use update() to modify it"
            )
        self._kos[ko.id] = ko
        self._save()

    def update(self, ko: KnowledgeObject) -> None:
        if ko.id not in self._kos:
            raise KeyError(
                f"KO '{ko.id}' not found — use add() for new KOs"
            )
        self._kos[ko.id] = ko
        self._save()

    def upsert(self, ko: KnowledgeObject) -> None:
        """Add or replace — convenience method for idempotent writes."""
        self._kos[ko.id] = ko
        self._save()

    def remove(self, ko_id: str) -> None:
        self._kos.pop(ko_id, None)
        self._save()

    # ── persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        self.path.write_text(
            json.dumps(
                [ko.to_dict() for ko in self._kos.values()],
                indent=2,
            ),
            encoding="utf-8",
        )
