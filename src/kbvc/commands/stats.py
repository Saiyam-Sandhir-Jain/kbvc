# kbvc/commands/stats.py
"""
kbvc stats — knowledge evolution analytics.

Produces a summary of:
  - KO counts by type and volatility
  - Relation counts by type
  - Commit frequency (commits per week/month)
  - Most-changed KO (most commits)
  - Most-connected KO (most relations)
  - Growth over time (KOs added per month)
  - Embedding cost estimate (total chunks embedded across all commits)
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from kbvc.core.ko import KOStore
    from kbvc.core.graph import RelationGraph


# ---------------------------------------------------------------------------
# StatsReport
# ---------------------------------------------------------------------------

@dataclass
class StatsReport:
    # Counts
    total_kos: int
    total_relations: int
    total_commits: int
    total_chunks_embedded: int

    # Breakdowns
    kos_by_type: Dict[str, int]
    kos_by_volatility: Dict[str, int]
    relations_by_type: Dict[str, int]

    # Star KOs
    most_changed_ko: Optional[str]
    most_changed_ko_commits: int
    most_connected_ko: Optional[str]
    most_connected_ko_relations: int

    # Growth
    kos_added_this_month: int
    commits_this_month: int

    # Branches
    branch_count: int
    branch_names: List[str]

    # Embedding cost estimate
    estimated_embed_calls: int


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_stats(
    ko_store: "KOStore",
    graph: "RelationGraph",
    commits: list,          # List[CommitObject] from walk_dag
    kbvc_dir: Path,
) -> StatsReport:
    all_kos = ko_store.all()

    # Basic counts
    total_kos = len(all_kos)
    total_relations = len(graph.relations)
    total_commits = len(commits)

    # KO breakdowns
    kos_by_type: Counter = Counter(ko.type for ko in all_kos)
    kos_by_volatility: Counter = Counter(ko.volatility for ko in all_kos)

    # Relation breakdowns
    relations_by_type: Counter = Counter(r.type for r in graph.relations)

    # Most changed KO (most times appearing in any commit's changed_kos)
    change_freq: Counter = Counter()
    total_chunks_embedded = 0
    for c in commits:
        for ko_id, change in c.changed_kos.items():
            change_freq[ko_id] += 1
            total_chunks_embedded += len(change.chunks_reembedded)

    most_changed_ko = change_freq.most_common(1)[0][0] if change_freq else None
    most_changed_ko_commits = change_freq.most_common(1)[0][1] if change_freq else 0

    # Most connected KO (most relations)
    conn_freq: Counter = Counter()
    for r in graph.relations:
        conn_freq[r.from_id] += 1
        conn_freq[r.to_id] += 1
    most_connected_ko = conn_freq.most_common(1)[0][0] if conn_freq else None
    most_connected_ko_relations = conn_freq.most_common(1)[0][1] if conn_freq else 0

    # Growth this month
    now = datetime.now(timezone.utc)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    kos_added_this_month = sum(
        1 for ko in all_kos
        if ko.last_updated >= month_start.date().isoformat()
    )
    commits_this_month = sum(
        1 for c in commits
        if _parse_iso(c.timestamp) >= month_start
    )

    # Branches
    heads_dir = kbvc_dir / "refs" / "heads"
    branch_names = sorted(p.name for p in heads_dir.iterdir()) if heads_dir.exists() else []

    return StatsReport(
        total_kos=total_kos,
        total_relations=total_relations,
        total_commits=total_commits,
        total_chunks_embedded=total_chunks_embedded,
        kos_by_type=dict(kos_by_type),
        kos_by_volatility=dict(kos_by_volatility),
        relations_by_type=dict(relations_by_type),
        most_changed_ko=most_changed_ko,
        most_changed_ko_commits=most_changed_ko_commits,
        most_connected_ko=most_connected_ko,
        most_connected_ko_relations=most_connected_ko_relations,
        kos_added_this_month=kos_added_this_month,
        commits_this_month=commits_this_month,
        branch_count=len(branch_names),
        branch_names=branch_names,
        estimated_embed_calls=total_chunks_embedded,
    )


def _parse_iso(ts: str) -> datetime:
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)
