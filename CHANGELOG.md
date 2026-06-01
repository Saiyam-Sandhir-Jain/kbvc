# Changelog

All notable changes to KBVC are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
KBVC uses [semantic versioning](https://semver.org/).

---

## [0.1.0] — 2026-05-31

Initial release. All nine phases complete.

### Added

**Phase 1 — Core version control**
- `kbvc init` — repository initialisation with optional git integration
- `kbvc add` / `kbvc status` — staging area for knowledge files
- `kbvc commit` — SHA-256 deterministic commit objects with chunk-level diff-only re-embedding
- `kbvc log` — full DAG commit history
- `kbvc config` — INI-based configuration (embed backend, vectordb backend, retrieval profile)
- Chunker with section-based, paragraph, and whole-document split strategies
- `kbvc.lock` — lock file recording exact embedding model and vector store config

**Phase 2 — Versioning workflow**
- `kbvc status` — staged / unstaged / stale KO summary
- `kbvc checkout` — restore full knowledge state to any commit or single KO
- `kbvc diff` — chunk-level diff against last commit
- `kbvc prompt` — prompt versioning (set / get / log / checkout)

**Phase 3 — Knowledge graph & retrieval**
- `kbvc link` / `kbvc unlink` — typed relations between KOs
- `kbvc graph` — neighbourhood visualisation
- `kbvc query` — nearest-neighbour vector search
- GraphRAG retrieval profile combining vector similarity with graph traversal
- Standard relation taxonomy: informed_by, cites, extends, part_of, contradicts, used_in, and more

**Phase 4 — Collaboration & audit**
- `kbvc branch` — parallel knowledge experiments (create / switch / list / delete)
- `kbvc depends` / `kbvc impact` — explicit dependency tracking and impact analysis
- `kbvc annotate` — human notes on KO versions
- `kbvc history` — per-KO version history
- `kbvc trace` — trace a vector ID to its commit, model, and section
- `kbvc doctor` — repository health check

**Phase 5 — Ingestion & remote sync**
- `kbvc ingest` — pull external content (website / GitHub / PDF / Notion) into staged `.md` files
- `kbvc push` / `kbvc remote` — incremental push to named remotes; only new commits sent
- `kbvc analyze` — auto-discover relation candidates (heuristic + vector similarity modes)
- `kbvc extract` — named entity extraction (models, tools, institutions, concepts)
- `kbvc stale` — freshness detection; flags KOs whose dependencies have been updated
- `kbvc stats` — evolution analytics dashboard

**Phase 6 — VSAL & provenance**
- ChunkRecord universal storage schema (Vector Storage Abstraction Layer)
- `kbvc backend init` / `kbvc backend info` — schema management
- `kbvc migrate backend` — backend-to-backend ChunkRecord migration
- `kbvc explain` — full knowledge provenance chain
- `kbvc promote` — elevate agent observations into versioned KOs
- Relation `confidence` and `source` fields
- New relation types: `supersedes`, `supported_by`

**Phase 7 — Maintenance**
- `kbvc gc` — garbage-collect orphaned vectors; optionally prune unreachable snapshots
- `kbvc migrate embeddings` — zero-downtime embedding model swap with in-place vector overwrite
- `kbvc migrate schema` — kbvc.lock schema version bump

**Phase 8 — Automation**
- `kbvc sync` — volatility-aware auto-commit (live / slow / all filters)

**Phase 9 — Contradiction management**
- `kbvc contradict list` — scan contradicts relations with resolution status
- `kbvc contradict resolve` — confirm resolution via supersedes relation

### Supported backends (v0.1.0)
- Embedding: OpenAI, Google Gemini, Ollama, HuggingFace sentence-transformers
- Vector DB: Qdrant, pgvector, Pinecone, ChromaDB

### Test coverage
195 tests · 0 failures · 0 errors

---

## [Unreleased]

### Planned
- `kbvc pull --rebuild` — re-embed all KOs after cloning
- `kbvc merge` — merge knowledge branches
- Python SDK — programmatic API for embedding in RAG frameworks
- `kbvc ask` — AI-assisted knowledge management
- Web UI — local dashboard (`kbvc ui`)
- GitHub Actions integration
- LanceDB / DuckDB / Parquet data lake backends
