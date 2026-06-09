# Changelog

All notable changes to KBVC are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
KBVC uses [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

_Changes merged to `main` but not yet released._

---

## [0.1.4] — 2026-06-07

### Fixed (pass 1 — instruction bugs)

- **CRITICAL** `kbvc query` now passes `filter={"branch": current_branch}` to
  the vector DB. Previously switching branches still returned vectors committed
  on other branches. Branch isolation is now enforced at query time.
- `kbvc diff <commit_a> <commit_b> [file]` now implements a real chunk-level
  diff between two commits. Previously it always printed the help message
  regardless of arguments.
- `kbvc status` no longer lists already-staged files under "Unstaged
  modifications". Files in the staging index are now skipped in that section.
- `kbvc gc --snapshots` now walks **all** branch HEADs (not just the current
  branch) before pruning. Feature-branch snapshots are no longer incorrectly
  marked as unreachable from master/main.
- `kbvc backend info` now shows correct dimensions for `gemini-embedding-2`
  (3072). Previously it fell back to the generic default of 1536.
- `kbvc annotate` now shows a clear warning when the target file is not yet
  staged, making it obvious the reason will only be recorded on the next commit.
- `kbvc promote` crashed with `argument of type 'PosixPath' is not iterable`
  because `StagingIndex` was constructed with a raw `Path` argument instead of
  using `StagingIndex.load()`. Fixed in `commands/promote.py`.
- `kbvc add ghost.md` (non-existent file) now exits non-zero. Previously it
  printed "✗ Not found" to stderr but returned exit code 0, hiding errors in
  scripts and CI.

### Fixed (pass 2 — deep stability audit)

- **CRITICAL** `qdrant.py` `query()` accepted a `filter` argument but
  silently ignored it — never passing it to `self._client.search()`. The branch
  isolation fix from pass 1 was therefore a no-op for all Qdrant users. Filter
  is now converted to a `qdrant_client.models.Filter` and passed as
  `query_filter`.
- **CRITICAL** `qdrant.py` `_scroll_by_prefix()` used `MatchValue(value=prefix)`
  which is an **exact** match, not a prefix match. Since no record has
  `str_id` exactly equal to the prefix string, every scroll returned 0 results —
  making `delete_by_prefix()` and `patch_metadata()` complete no-ops. Fixed by
  removing the broken Qdrant filter and doing all prefix-matching client-side on
  the scrolled payload.
- `chroma.py` `delete_by_prefix()` passed `where_document=None` to `col.get()`,
  which is not a valid ChromaDB API call. Fixed to call `col.get()` with no
  arguments.
- `lancedb.py` `patch_metadata()` and `exists_batch()` called `tbl.search()`
  without a query vector, which crashes in LanceDB (ANN search requires a
  vector). Both methods now use `tbl.to_pandas()` for full-table access.
- `lancedb.py` removed the module-level `_SCHEMA_CACHE` dict that was defined
  but never referenced anywhere in the file.
- `repo.py` had `KBVC_VERSION = "0.1.2"` hardcoded, so every new repository's
  `repo.json` reported an old version. Now reads from `kbvc.__version__` at
  import time.
- `lock.py` had `KBVC_VERSION = "0.1.0"` hardcoded (four versions behind), so
  every `kbvc.lock` update recorded a stale version. Now reads from
  `kbvc.__version__`.
- `config.py` `_VALID_SECTIONS` was missing `"core"` and `"graph"`. Running
  `kbvc config set core.format_version 2` or `kbvc config set graph.snapshot_mode
  full` raised `Unknown config section`. Both sections are now included.
- `sync.py` volatility tiers were mis-mapped: `slow` included `{"slow","live"}`
  and `all` was identical to `slow` (making `all` useless). Fixed to:
  `live` → `{"live"}`, `slow` → `{"slow"}`, `all` → `{"slow","live"}`.
- `stats.py` `_parse_iso()` returned naive `datetime` objects while `month_start`
  was timezone-aware, causing `TypeError: can't compare offset-naive and
  offset-aware datetimes` every time `kbvc stats` was run. Fixed by always
  attaching `timezone.utc` to parsed datetimes.
- `commit.py` `KOChange.chunks` field was typed `List[dict] = None` (incorrect
  for a dataclass — mutable default and wrong `None` default). Fixed to
  `field(default_factory=list)` with the `__post_init__` removed.
- `push.py` had a duplicate `from kbvc.core.chunker import ...` statement inside
  the per-commit loop that shadowed the import already present at function scope.

### Changed (pass 1)

- `[all]` extras now include `google-genai>=0.1` and `sentence-transformers>=2.0`
  so embedding works out of the box after `pip install kbvc[all]`.
- `kbvc promote --type` now accepts `finding`, `fact`, `rule`, `decision`, `note`.
- `_MODEL_DIMS` (backend.py) and `_GEMINI_DIMS` (gemini.py) now include
  `gemini-embedding-2: 3072`.

---

## [0.1.2] — 2026-06-03

Stability patch — all bugs discovered during real-world pre-release testing.

### Fixed

- **`kbvc gc --snapshots` crashes with `AttributeError`** — `repo.head()` does not exist; method is `repo.head_commit()`. Also `CommitObject.all_from()` does not exist; replaced with `walk_dag()` from `kbvc.core.commit`. Both crash paths are now fixed; added `head()` alias to `KbvcRepo` for forward compatibility.
- **`chunk_hash` always empty in vector metadata** — `kbvc commit` and `kbvc push` both omitted `chunk_hash` from the metadata dict written to the vector DB. `kbvc explain` and `kbvc gc` rely on this field being present. Fixed in both `commands/commit.py` and `commands/push.py`.
- **`.gitignore` did not exclude `.kbvc/config`** — `kbvc init` only added `.kbvc/secrets` to `.gitignore`, but API keys are stored in `.kbvc/config`. Anyone who ran `git add .` could accidentally commit their API key. Fixed: `.kbvc/config` is now excluded on `kbvc init`.
- **`kbvc doctor` false-negative for git on empty repo** — `git rev-parse --abbrev-ref HEAD` exits 128 on repos with no commits. Fixed to use `git rev-parse --git-dir` first, then get branch name separately with graceful "no commits yet" handling.

### Changed

- `kbvc init` output now confirms `git: initialised ✓` and shows `kbvc backend init` + `kbvc doctor` in next steps.

---

## [0.1.0] — 2026-06-02

First public release of KBVC — Knowledge Base Version Control.

### Added

#### Core Infrastructure (Phases 1–4)
- `kbvc init` — initialize a KBVC repository with Git integration
- `kbvc add` / `kbvc commit` — Git-style staging and commit loop for knowledge
- `kbvc log` — commit history with full DAG traversal
- `kbvc status` — show staged vs. committed state
- `kbvc checkout` — time travel to any past knowledge state
- `kbvc diff` — compare staged state against HEAD
- `kbvc history` — version history for a single KO
- `kbvc branch` — branch management (create, list, switch)
- `kbvc link` / `kbvc graph` — knowledge graph with typed relations
- `kbvc depends` / `kbvc impact` — dependency traversal
- `kbvc annotate` — per-KO change reason before commit
- `kbvc trace` — full version audit trail for a KO
- `kbvc doctor` — repository health check

#### Intelligence & Analytics (Phase 5)
- `kbvc ingest` — multi-source ingestion (website, GitHub, PDF, Notion, text)
- `kbvc push` / `kbvc remote` — push to remote vector DB
- `kbvc analyze` — heuristic semantic relation suggestion
- `kbvc extract` — entity extraction from KO content
- `kbvc stale` — staleness and orphan relation detection
- `kbvc stats` — knowledge analytics report

#### Infrastructure & Provenance (Phase 6)
- `kbvc backend init` / `kbvc backend info` — schema management
- `kbvc explain` — trace any vector back to its source document and commit
- `kbvc promote` — promote agent memory into a permanent KO
- `kbvc clone` — clone a KBVC repo with lock-file-aware setup guidance

#### Maintenance Operations (Phase 7)
- `kbvc gc` — garbage-collect orphaned vectors
- `kbvc migrate backend` — zero-downtime vector DB migration without re-embedding
- `kbvc migrate embeddings` — swap embedding models with full re-embedding
- `kbvc migrate schema` — lock file schema version bump

#### Sync & Automation (Phase 8)
- `kbvc sync` — volatility-aware auto-commit (live / slow / all)
- `kbvc sync --dry-run` — preview what would be committed

#### Contradiction Detection (Phase 9)
- `kbvc contradict list` — list all `contradicts` relations with context
- `kbvc contradict resolve` — resolve a knowledge conflict

#### AI-Assisted Q&A (Phase 10)
- `kbvc ask` — grounded knowledge Q&A with citations and provenance

#### Embedding Backends
- OpenAI (`text-embedding-3-small`, `text-embedding-3-large`, `ada-002`)
- Gemini (`gemini-embedding-001`)
- Ollama (any locally-served model)
- HuggingFace (`sentence-transformers`)

#### Vector DB Backends
- Qdrant (full VSAL implementation)
- pgvector (full VSAL implementation)
- Chroma (full VSAL implementation including `initialize_schema`, `export_chunks`)
- Pinecone (full VSAL implementation including `initialize_schema`, `export_chunks`)
- **LanceDB** (new — embedded, serverless, no external process required)

#### Other
- `kbvc.lock` — reproducible deployment manifest (analogous to `package-lock.json`)
- SHA-256 commit DAG with full content-addressability
- Prompt versioning per commit (unique to KBVC)
- Retrieval config versioning per commit
- `ChunkRecord` VSAL — universal vector storage abstraction layer
- `rich` added to core dependencies for terminal output

### Fixed
- `commands/sync.py` — `StagingIndex(path)` corrected to `StagingIndex.load(path)`
- `core/chunker.py` — docstring corrected ("xxhash" → "sha256")
- `cli.py` clone command — now reads `kbvc.lock` and provides provider-specific next steps

### Architecture
- Clean separation: `core/` (data models) / `backends/` (pluggable infra) / `commands/` (business logic) / `cli.py` (Click wiring)
- 195 tests passing — all external backends mocked, fully offline
- Python 3.9–3.12 supported

---

[Unreleased]: https://github.com/Saiyam-Sandhir-Jain/kbvc/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Saiyam-Sandhir-Jain/kbvc/releases/tag/v0.1.0
