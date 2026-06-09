# kbvc/utils/ignore.py
"""
.kbvcignore — gitignore-style file exclusion for kbvc add / kbvc status.

Reads patterns from:
  <repo_root>/.kbvcignore   (committed; shared across the team)
  <repo_root>/.kbvc/ignore  (local-only; listed in .gitignore by default)

Pattern syntax is a subset of gitignore:
  - Lines starting with ``#`` are comments and ignored
  - Blank lines are ignored
  - A trailing ``/`` matches directories only
  - A leading ``**/`` matches in any subdirectory
  - ``*`` matches anything except a path separator
  - ``**`` matches anything including path separators
  - All other patterns are matched against the relative path from repo root
    OR against each path component (so "node_modules" ignores the dir
    wherever it appears, not just at the root)

Built-in defaults (always applied, regardless of .kbvcignore):
  .venv/  .env/  node_modules/  __pycache__/  .git/  .kbvc/
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import List


# Patterns that are always ignored regardless of .kbvcignore
_BUILTIN_IGNORES: List[str] = [
    ".venv",
    "venv",
    ".env",
    "node_modules",
    "__pycache__",
    ".git",
    ".kbvc",
    "*.pyc",
    "*.pyo",
    ".DS_Store",
]


def load_ignore_patterns(repo_root: Path) -> List[str]:
    """
    Load all ignore patterns for this repo.
    Returns the combined list: builtins + .kbvcignore + .kbvc/ignore.
    """
    patterns = list(_BUILTIN_IGNORES)

    for ignore_file in (
        repo_root / ".kbvcignore",
        repo_root / ".kbvc" / "ignore",
    ):
        if ignore_file.exists():
            try:
                for line in ignore_file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        patterns.append(line)
            except OSError:
                pass

    return patterns


def is_ignored(rel_path: str, patterns: List[str]) -> bool:
    """
    Return True if ``rel_path`` (relative to repo root, using forward slashes)
    matches any of the ignore patterns.

    Matching rules:
      - Strip trailing / from pattern (directory-only marker) and match the
        path component, since we're given a file path.
      - If the pattern contains no /, match against each path component.
      - If the pattern contains /, match against the full relative path with
        fnmatch (supporting * and ** globs).
    """
    # Normalise to forward slashes
    rel_path = rel_path.replace("\\", "/")
    parts = rel_path.split("/")

    for pat in patterns:
        # Strip trailing slash (dir-only hint — we still match files inside)
        p = pat.rstrip("/")
        if not p:
            continue

        if "/" not in p:
            # Component match: ignore if any part of the path matches the pattern
            if any(fnmatch.fnmatch(part, p) for part in parts):
                return True
        else:
            # Full-path match with glob
            # Handle leading **/ to mean "anywhere in path"
            if p.startswith("**/"):
                suffix = p[3:]
                if fnmatch.fnmatch(rel_path, suffix) or fnmatch.fnmatch(
                    rel_path, "**/" + suffix
                ):
                    return True
                # Also check against each sub-path
                for i in range(len(parts)):
                    sub = "/".join(parts[i:])
                    if fnmatch.fnmatch(sub, suffix):
                        return True
            else:
                if fnmatch.fnmatch(rel_path, p):
                    return True

    return False
