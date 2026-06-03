# Changelog

All notable changes to KBVC are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
KBVC uses [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

_Changes merged to `main` but not yet released._

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
