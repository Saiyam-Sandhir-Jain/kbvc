# kbvc/commands/stale.py
"""
kbvc stale — detect and report stale Knowledge Objects.

A KO is stale when one or more of its declared dependencies (depends_on)
have been updated (committed at a later date) without the KO itself being
recommitted.

Output tiers:
  STALE    — dependency was committed after this KO's last update
  FROZEN   — KO is frozen; may be intentionally out of date
  FRESH    — all dependencies newer-than-or-equal to this KO

The report also includes relations in the graph that reference KOs by ID
where the referenced KO no longer exists (orphan relation detection).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from kbvc.core.ko import KOStore
    from kbvc.core.graph import RelationGraph


# ---------------------------------------------------------------------------
# Report models
# ---------------------------------------------------------------------------

@dataclass
class StalenessEntry:
    ko_id: str
    ko_path: str
    volatility: str
    ko_last_updated: str
    stale_deps: List[dict]  # [{"dep_id": str, "dep_updated": str}]
    status: str             # "STALE" | "FROZEN" | "FRESH"


@dataclass
class OrphanRelation:
    rel_id: str
    from_id: str
    to_id: str
    missing_side: str  # "from" | "to" | "both"


@dataclass
class StalenessReport:
    stale: List[StalenessEntry]
    frozen: List[StalenessEntry]
    fresh: List[StalenessEntry]
    orphan_relations: List[OrphanRelation]


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def compute_staleness(
    ko_store: "KOStore",
    graph: "RelationGraph",
) -> StalenessReport:
    """
    Compute the full freshness report for all tracked KOs.

    Rules:
      - STALE:  any dep has last_updated > this KO's last_updated
      - FROZEN: ko.volatility == "frozen"  (may or may not have stale deps)
      - FRESH:  all deps updated on or before this KO

    Orphan detection:
      Relations in the graph that point to ko_ids not in the KO store.
    """
    all_kos = {ko.id: ko for ko in ko_store.all()}
    stale: List[StalenessEntry] = []
    frozen: List[StalenessEntry] = []
    fresh: List[StalenessEntry] = []

    for ko in ko_store.all():
        stale_deps = []
        for dep_id in (ko.depends_on or []):
            dep = all_kos.get(dep_id)
            if dep is None:
                stale_deps.append({"dep_id": dep_id, "dep_updated": "MISSING"})
                continue
            # ISO date strings compare lexicographically correctly
            if dep.last_updated > ko.last_updated:
                stale_deps.append({
                    "dep_id": dep_id,
                    "dep_updated": dep.last_updated,
                })

        entry = StalenessEntry(
            ko_id=ko.id,
            ko_path=ko.path,
            volatility=ko.volatility,
            ko_last_updated=ko.last_updated,
            stale_deps=stale_deps,
            status="FROZEN" if ko.volatility == "frozen"
                   else ("STALE" if stale_deps else "FRESH"),
        )

        if ko.volatility == "frozen":
            frozen.append(entry)
        elif stale_deps:
            stale.append(entry)
        else:
            fresh.append(entry)

    # Orphan relation detection
    orphans: List[OrphanRelation] = []
    for rel in graph.relations:
        from_missing = rel.from_id not in all_kos
        to_missing = rel.to_id not in all_kos
        if from_missing or to_missing:
            if from_missing and to_missing:
                side = "both"
            elif from_missing:
                side = "from"
            else:
                side = "to"
            orphans.append(OrphanRelation(
                rel_id=rel.id,
                from_id=rel.from_id,
                to_id=rel.to_id,
                missing_side=side,
            ))

    return StalenessReport(
        stale=sorted(stale, key=lambda e: e.ko_last_updated),
        frozen=frozen,
        fresh=fresh,
        orphan_relations=orphans,
    )
