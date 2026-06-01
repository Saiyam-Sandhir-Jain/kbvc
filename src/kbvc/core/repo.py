# kbvc/core/repo.py
"""
KbvcRepo — the central repository access object.

All KBVC commands call KbvcRepo.require() as their first step to verify
they're inside a KBVC repository (analogous to git's "not a git repository"
check). Every path helper, config reader, and sub-store is accessed via this object.

Repository identity is stored in .kbvc/repo.json (§3.0).
Config lives in .kbvc/config (INI, read as flat dot-key dict).
"""

from __future__ import annotations

import json
import subprocess
import uuid
from configparser import ConfigParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

KBVC_VERSION = "0.1.0"
FORMAT_VERSION = "1"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class NotKBVCRepositoryError(Exception):
    """Raised when any kbvc command is run outside a KBVC repository."""
    pass


# ---------------------------------------------------------------------------
# KbvcRepo
# ---------------------------------------------------------------------------

class KbvcRepo:
    """
    Central access point for a KBVC repository.

    Init:   KbvcRepo.init(Path("."))    — creates a new repository
    Load:   KbvcRepo.require()          — load existing repo or raise
    """

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.kbvc_dir = self.root / ".kbvc"

    # ── factory ───────────────────────────────────────────────────────────────

    @staticmethod
    def require(path: Path = Path(".")) -> KbvcRepo:
        """
        Walk up the directory tree looking for .kbvc/repo.json.
        Returns the KbvcRepo if found; raises NotKBVCRepositoryError otherwise.

        This mirrors Git's behaviour: you can run kbvc commands from subdirectories.
        """
        resolved = path.resolve()
        for candidate in [resolved, *resolved.parents]:
            if (candidate / ".kbvc" / "repo.json").exists():
                return KbvcRepo(candidate)
        raise NotKBVCRepositoryError(
            "Not a KBVC repository (no .kbvc/repo.json found in this directory "
            "or any parent).\n"
            "Run: kbvc init    to initialize a new repository\n"
            "  or: kbvc clone  to clone an existing knowledge base"
        )

    @classmethod
    def init(
        cls,
        path: Path,
        name: Optional[str] = None,
        no_git: bool = False,
    ) -> KbvcRepo:
        """
        Create a new KBVC repository at `path`.

        Steps:
          1. Run git init (unless --no-git or already a git repo)
          2. Create .kbvc/ directory tree
          3. Write repo.json, HEAD, refs, empty stores, config skeleton
          4. Write kbvc.lock placeholder
          5. Update .gitignore
        """
        path = path.resolve()
        path.mkdir(parents=True, exist_ok=True)

        # ── Step 1: git init ──────────────────────────────────────────────────
        if not no_git and not (path / ".git").exists():
            try:
                subprocess.run(
                    ["git", "init", str(path)],
                    check=True,
                    capture_output=True,
                )
            except FileNotFoundError:
                # P1 pitfall: git not on PATH
                raise RuntimeError(
                    "git not found on PATH. Install git or use --no-git flag."
                )

        # ── Step 2: .kbvc/ directory tree ─────────────────────────────────────
        kbvc = path / ".kbvc"
        for subdir in [
            "commits",
            "graph",
            "prompts",
            "retrieval",
            "migrations",
            "ko_versions",
            "objects",       # v3 shared object store — dir created now; no content in v1
            "refs/heads",
        ]:
            (kbvc / subdir).mkdir(parents=True, exist_ok=True)

        # ── Step 3: core metadata files ────────────────────────────────────────

        # repo.json — repository identity marker (§3.0)
        (kbvc / "repo.json").write_text(
            json.dumps(
                {
                    "format_version": FORMAT_VERSION,
                    "kbvc_version": KBVC_VERSION,
                    "repo_id": str(uuid.uuid4()),
                    "name": name or path.name,
                    "created_by": f"kbvc {KBVC_VERSION}",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        # HEAD — points to main branch by default
        (kbvc / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")

        # Empty main branch ref (no commit yet)
        (kbvc / "refs" / "heads" / "main").write_text("", encoding="utf-8")

        # Empty KO store
        (kbvc / "ko_store.json").write_text("[]", encoding="utf-8")

        # Empty relation graph
        (kbvc / "graph" / "current.json").write_text("[]", encoding="utf-8")

        # Empty staging index
        (kbvc / "index").write_text(
            json.dumps(
                {
                    "staged_files": [],
                    "ko_reasons": {},
                    "graph_dirty": False,
                    "prompt_dirty": False,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        # Config skeleton (INI)
        (kbvc / "config").write_text(
            "[core]\n"
            "  format_version = 1\n\n"
            "[vectordb]\n"
            "  backend =\n"
            "  url =\n"
            "  key =\n"
            "  collection = kbvc\n\n"
            "[embed]\n"
            "  backend =\n"
            "  model =\n"
            "  key =\n"
            "  url =\n\n"
            "[retrieval]\n"
            "  profile = vector\n"
            "  hop_depth = 2\n"
            "  top_k = 5\n"
            "  weight_semantic = 0.7\n"
            "  weight_graph = 0.3\n\n"
            "[chunk]\n"
            "  size = 400\n"
            "  overlap = 50\n"
            "  split_on = ##\n\n"
            "[graph]\n"
            "  snapshot_mode = full\n"      # delta logic is v3; accepted as no-op
            "  delta_threshold = 1000\n",
            encoding="utf-8",
        )

        # ── Step 4: kbvc.lock placeholder ──────────────────────────────────────
        lock_content = (
            "# kbvc.lock — auto-generated by KBVC. "
            "Commit this file. Do not edit manually.\n"
            f'kbvc_version: "{KBVC_VERSION}"\n'
            "format_version: 1\n"
            'generated_at: ""\n'
            "embedding:\n"
            '  provider: ""\n'
            '  model: ""\n'
            "  dims: 0\n"
            "vector_store:\n"
            '  provider: ""\n'
            '  collection: kbvc\n'
            "retrieval:\n"
            "  profile: vector\n"
            "  hop_depth: 2\n"
            "  top_k: 5\n"
            "  weight_semantic: 0.7\n"
            "  weight_graph: 0.3\n"
        )
        (path / "kbvc.lock").write_text(lock_content, encoding="utf-8")

        # ── Step 5: .gitignore — keep secrets out of git ───────────────────────
        block = (
            "\n# KBVC — local secrets (never commit API keys)\n"
            ".kbvc/secrets\n"
        )
        gitignore = path / ".gitignore"
        if not gitignore.exists() or block not in gitignore.read_text(encoding="utf-8"):
            with open(gitignore, "a", encoding="utf-8") as f:
                f.write(block)

        return cls(path)

    # ── path helpers ──────────────────────────────────────────────────────────

    @property
    def commits_dir(self) -> Path:
        return self.kbvc_dir / "commits"

    @property
    def graph_dir(self) -> Path:
        return self.kbvc_dir / "graph"

    @property
    def prompts_dir(self) -> Path:
        return self.kbvc_dir / "prompts"

    @property
    def retrieval_dir(self) -> Path:
        return self.kbvc_dir / "retrieval"

    @property
    def ko_versions_dir(self) -> Path:
        return self.kbvc_dir / "ko_versions"

    @property
    def ko_store_path(self) -> Path:
        return self.kbvc_dir / "ko_store.json"

    @property
    def index_path(self) -> Path:
        return self.kbvc_dir / "index"

    @property
    def lock_path(self) -> Path:
        return self.root / "kbvc.lock"

    # ── HEAD / branch helpers ─────────────────────────────────────────────────

    def current_branch(self) -> str:
        """Return the current branch name, or 'detached' in detached HEAD state."""
        head_text = (self.kbvc_dir / "HEAD").read_text(encoding="utf-8").strip()
        if head_text.startswith("ref: refs/heads/"):
            return head_text[len("ref: refs/heads/"):]
        return "detached"

    def head_commit(self) -> Optional[str]:
        """Return the full commit hash at HEAD, or None if no commits yet."""
        branch = self.current_branch()
        if branch == "detached":
            return None
        ref_path = self.kbvc_dir / "refs" / "heads" / branch
        if ref_path.exists():
            h = ref_path.read_text(encoding="utf-8").strip()
            return h if h else None
        return None

    def advance_head(self, commit_id: str) -> None:
        """Update the current branch ref to point to the new commit."""
        branch = self.current_branch()
        if branch == "detached":
            # Write directly to HEAD (detached commit)
            (self.kbvc_dir / "HEAD").write_text(commit_id, encoding="utf-8")
        else:
            (self.kbvc_dir / "refs" / "heads" / branch).write_text(
                commit_id, encoding="utf-8"
            )

    # ── config ────────────────────────────────────────────────────────────────

    def config(self) -> dict:
        """
        Parse .kbvc/config (INI) into a flat dot-key dict.

        [vectordb]
          url = http://...   →  {"vectordb.url": "http://..."}
        [embed]
          model = text-3     →  {"embed.model": "text-3"}
        """
        parser = ConfigParser()
        parser.read(self.kbvc_dir / "config")
        flat: dict = {}
        for section in parser.sections():
            for key, val in parser.items(section):
                flat[f"{section}.{key}"] = val
        return flat

    def repo_info(self) -> dict:
        """Return the contents of repo.json."""
        return json.loads(
            (self.kbvc_dir / "repo.json").read_text(encoding="utf-8")
        )
