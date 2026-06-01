# KBVC — Knowledge Base Version Control

> **Git-native Knowledge Infrastructure Layer for AI systems.**
> Version: `0.1.0` · Python ≥ 3.9 · No mandatory cloud dependencies

---

## The Core Idea

Modern AI systems are only as good as the knowledge they retrieve. But today, that knowledge has no version control. When you update a document, old vectors silently persist in your database. When you change your prompt, there is no record of what it was before. When a retrieval result is wrong, you cannot trace which commit embedded it, which model created the vector, or why the chunk was written that way.

KBVC fixes this. It is to RAG pipelines what Git is to code — a complete audit trail, a reproducibility guarantee, and a collaborative workflow for the knowledge layer of AI systems.

```
Without KBVC:          With KBVC:
                       
docs/ ─────────────    docs/ ─────────────
  │ (changed silently)   │ kbvc add .
  ↓                      │ kbvc commit -m "update API docs"
vector DB ─────────    .kbvc/commits/a3f7c91…
  (stale vectors)        │ KOs: manifestai@1→2
  (no audit trail)       │ Graph: graph-v3
  (non-reproducible)     │ Prompt: p-v1
                         │ Retrieval: r-v1
                       vector DB (only changed chunks)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      kbvc CLI                           │
│  init  add  commit  push  log  status  checkout  diff   │
│  link  graph  query  analyze  extract  stale  stats     │
│  ingest(website/github/pdf/notion)  doctor  trace       │
│  backend(init/info)  migrate  explain  promote          │
│  gc  sync  contradict(list/resolve)                     │
│  relation(list/show/create)  clone                      │
└────────────────────────┬────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
   ┌──────────┐   ┌──────────┐   ┌──────────────┐
   │  core/   │   │backends/ │   │  commands/   │
   │  ko      │   │  embed   │   │  commit      │
   │  commit  │   │  openai  │   │  ingest      │
   │  graph   │   │  gemini  │   │  push        │
   │  chunker │   │  ollama  │   │  analyze     │
   │  index   │   │  hf      │   │  stale       │
   │  versioner│  │  vectordb│   │  stats       │
   │  lineage  │  │  ─────── │   │  migrate     │
   │  prompts  │  │  VSAL    │   │  backend     │
   │  retrieval│  │  ChunkRec│   │  explain     │
   │  relation_│  │  qdrant  │   │  promote     │
   │  registry │  │  pgvector│   │  gc          │
   └──────────┘   │  pinecone│   │  sync        │
                  │  chroma  │   │  contradict  │
                  └──────────┘   └──────────────┘
```

### Vector Storage Abstraction Layer (VSAL)

KBVC never writes raw backend objects (Qdrant Points, PgVector rows, Pinecone records). All storage goes through a single universal unit:

```
KBVCChunk (ChunkRecord)
├── vector_id      ← e.g. "main__manifestai__chunk_0"
├── branch
├── ko_id
├── ko_version
├── chunk_index
├── chunk_hash
├── embedding
├── metadata
└── created_at
```

Each backend adapter maps this to its native representation. This means the storage backend is an implementation detail — the KBVC Knowledge State is the canonical source of truth.

### Repository layout on disk

```
your-project/
├── kbvc.lock                   ← commit this (like package-lock.json)
├── .gitignore                  ← kbvc auto-adds .kbvc/secrets
│
├── .kbvc/
│   ├── repo.json               ← repository identity (uuid, format version)
│   ├── HEAD                    ← ref: refs/heads/main
│   ├── config                  ← INI config (backends, retrieval, chunk settings)
│   ├── index                   ← staging area (like git's index)
│   ├── relations.yaml          ← custom relation type definitions
│   │
│   ├── ko_store.json           ← all tracked KO metadata
│   ├── commits/                ← one JSON file per commit (SHA-256 named)
│   ├── ko_versions/<ko_id>/    ← per-KO version snapshots vN.json
│   │
│   ├── graph/
│   │   ├── current.json        ← mutable working graph
│   │   └── graph-vN.json       ← immutable snapshots (one per commit)
│   │
│   ├── prompts/
│   │   ├── current.json        ← pending prompt
│   │   └── p-vN.json           ← immutable prompt snapshots
│   │
│   ├── retrieval/
│   │   └── r-vN.json           ← retrieval config snapshots
│   │
│   └── refs/
│       ├── heads/main          ← branch tip (full SHA-256)
│       └── remotes/origin      ← last pushed commit per remote
│
└── knowledge/                  ← your .md files (normal git-tracked content)
    ├── projects/
    ├── research/
    └── ingested/               ← output of kbvc ingest
```

---

## Knowledge Objects (KOs)

A **Knowledge Object** is the atomic unit of versioned knowledge. One `.md` file → one KO.

```yaml
# projects/manifestai.md
---
id: manifestai                  # required — stable KO identifier
type: project                   # project | education | patent | concept | doc ...
tags: [ai, startup, manifesto]
volatility: slow                # frozen | slow | live
valid_from: "2024-03-01"        # optional temporal validity
valid_to: null
---

## Overview

ManifestAI is an AI-native knowledge platform...

## Technical Stack

Built with LangChain, ChromaDB, and FastAPI...
```

Each KO tracks:

| Field | Description |
|---|---|
| `id` | Stable identifier, used in all relations and vector IDs |
| `version` | Monotonically increasing integer, bumped every commit |
| `chunk_hashes` | SHA-256 of each chunk — drives diff-only re-embedding |
| `vector_ids` | `branch__ko_id__chunk_N` format |
| `volatility` | `frozen` = never re-embed; `slow` = normal; `live` = always re-embed |
| `depends_on` | Other KO IDs this one semantically depends on |
| `entities` | Named entities extracted by `kbvc extract` |
| `valid_from/to` | Temporal knowledge windows |

---

## The Commit Object

Every `kbvc commit` creates a **global commit object** — not per-file like git, but a snapshot of the entire AI knowledge state:

```
Commit a3f7c91
├── KOs changed:
│   ├── manifestai: v1 → v2 (3 chunks re-embedded)
│   └── gemini-docs: v0 → v1 (5 chunks embedded)
├── Graph snapshot: graph-v3
├── Prompt snapshot: p-v1
└── Retrieval snapshot: r-v1 (model: text-embedding-3-small, dims: 1536)
```

The commit ID is a **deterministic SHA-256** of the content — same staged content committed twice produces the same hash. Timestamp is stored but never hashed.

---

## Chunk-Level Versioning (The Cost Saver)

KBVC only re-embeds chunks that actually changed. For a 50-section document where 2 sections were edited, only 2 API calls are made instead of 50.

```
manifestai.md (v1 → v2)

## Overview     hash: a1b2c3  ✓ unchanged — skip
## Team         hash: d4e5f6  ✓ unchanged — skip
## Funding      hash: NEW     ✗ changed  — re-embed  ← 1 API call
## Tech Stack   hash: NEW     ✗ changed  — re-embed  ← 1 API call
## Roadmap      hash: g7h8i9  ✓ unchanged — skip
```

Deleted sections have their vectors removed from the DB to prevent ghost retrievals.

---

## The Separation of Commit vs Push vs Ingest

```
kbvc ingest website https://docs.example.com
  └── Downloads + converts to .md
      └── Writes to ingested/web/docs-example-com.md
          └── NO vector DB touched

kbvc add ingested/
kbvc commit -m "ingest example docs"
  └── Embeds changed chunks into LOCAL vector DB
      └── Creates commit object
          └── NO remote DB touched

kbvc push origin
  └── Syncs committed vectors to REMOTE DB
      └── Tracks what was pushed (refs/remotes/origin)
          └── Incremental — only pushes new commits
```

---

## Commands Reference

### Setup

```bash
kbvc init                          # initialise repo + git init + kbvc.lock
kbvc init --no-git                 # skip git init
kbvc config set embed.backend openai
kbvc config set embed.key sk-...
kbvc config set embed.model text-embedding-3-small
kbvc config set vectordb.backend qdrant
kbvc config set vectordb.url http://localhost:6333
kbvc config list                   # show all config (keys redacted)
```

### Core workflow

```bash
kbvc add projects/manifestai.md    # stage a file
kbvc add knowledge/                # stage a directory
kbvc add .                         # stage all .md files (sorted, deterministic)
kbvc status                        # show staged, unstaged modifications, dirty flags
kbvc commit -m "update funding round details"
kbvc commit -m "..." --dry-run     # preview: show what would be embedded
kbvc log                           # full history (DAG walk, newest first)
kbvc log --oneline                 # compact one-line per commit
kbvc log -- projects/manifestai.md # filter to commits touching this file
```

### Retrieval

```bash
kbvc query "what is ManifestAI's tech stack?"
kbvc query "funding details" --top-k 10
kbvc query "..." --profile graphrag  # vector + graph traversal
```

### Relations and Graph

```bash
kbvc link projects/manifestai.md research/transformers.md \
    --type informed_by \
    --note "ManifestAI's retrieval architecture is based on this paper"

kbvc link a.md b.md --type extends --valid-from 2024-01-01

kbvc graph projects/manifestai.md   # show neighbours
kbvc graph projects/manifestai.md --depth 2 --type informed_by
kbvc graph --all                    # print full graph

kbvc unlink rel-a1b2c3d4            # remove a relation by ID
```

### Relation Type Registry

```bash
kbvc relation list                       # all built-in and custom relation types
kbvc relation list --category core       # core algorithmic types only
kbvc relation list --category temporal   # temporal convenience types
kbvc relation show informed_by           # full detail: inverse, flags, used_by
kbvc relation create deployed_on \
    --inverse hosts \
    --category infrastructure
kbvc relation create similar_to --symmetric --category semantic
kbvc relation create prerequisite_of --transitive --inverse requires
```

**Built-in core relations** (affect KBVC algorithm internals):

| Relation | Inverse | Properties | Used by |
|---|---|---|---|
| `depends_on` | `supports` | transitive | `kbvc impact`, `kbvc stale` |
| `supports` | `depends_on` | transitive | `kbvc explain` |
| `contradicts` | `contradicts` | symmetric | `kbvc contradict` |
| `supersedes` | `superseded_by` | — | `kbvc contradict resolve` |
| `supported_by` | `supports` | — | `kbvc explain` |

**Built-in temporal relations** (traversal only):

`developed_during`, `studied_at`, `worked_at`, `informed_by`, `influenced`, `extends`, `part_of`, `created_at`, `cites`, `used_in`

### Versioning and History

```bash
kbvc history projects/manifestai.md         # per-KO version history
kbvc history projects/manifestai.md --oneline
kbvc diff projects/manifestai.md            # chunk diff vs last commit
kbvc checkout a3f7c91                       # restore full state to commit
kbvc checkout a3f7c91 -- projects/manifestai.md  # restore single KO
kbvc annotate projects/manifestai.md --reason "Series A announced"
```

### Prompt versioning

```bash
kbvc prompt set "Answer using only the provided context. Be concise."
kbvc prompt set "..." --system "You are an expert in AI infrastructure."
kbvc prompt get
kbvc prompt log
kbvc prompt checkout p-v2           # restore a prior prompt
```

### Branches

```bash
kbvc branch create experiment
kbvc branch switch experiment
kbvc branch list
kbvc branch show                    # current branch + HEAD
kbvc branch delete old-branch
```

### Dependencies

```bash
kbvc depends add consumer.md base.md    # consumer depends on base
kbvc depends remove consumer.md base.md
kbvc depends list consumer.md
kbvc impact base.md                     # what breaks if base.md changes?
kbvc impact base.md --depth 2           # transitive impact
```

### Knowledge Intelligence

```bash
# Auto-discover relations
kbvc analyze                              # heuristic mode
kbvc analyze --use-vectors               # cosine similarity mode
kbvc analyze --min-confidence 0.6        # higher threshold
kbvc analyze --apply                     # auto-link all suggestions
kbvc analyze --max-suggestions 5         # prevent graph explosion

# Auto-extract entities
kbvc extract projects/manifestai.md      # show candidates
kbvc extract --all --min-confidence 0.8  # all KOs, high threshold
kbvc extract projects/manifestai.md --apply  # write to ko_store

# Freshness
kbvc stale                               # show stale KOs
kbvc stale --show-fresh                  # full dashboard
kbvc stale --fix                         # stage stale KOs for re-commit

# Analytics
kbvc stats                               # evolution dashboard
kbvc stats --json-out                    # machine-readable output
```

### Ingest (external sources → .md files)

```bash
# Download then review before committing
kbvc ingest website https://docs.anthropic.com
kbvc ingest github https://github.com/langchain-ai/langchain --branch main
kbvc ingest pdf ~/papers/attention-is-all-you-need.pdf
kbvc ingest notion abc123def456 --token $NOTION_TOKEN

# All ingested files land in ingested/ — normal .md, editable
kbvc add ingested/
kbvc commit -m "ingest anthropic docs v2"
```

### Push to remote

```bash
# Configure a remote
kbvc remote add origin \
    --backend qdrant \
    --url https://prod-cluster.qdrant.io \
    --collection kbvc-prod

kbvc remote add staging \
    --backend qdrant \
    --url https://staging.qdrant.io \
    --collection kbvc-staging

kbvc remote list
kbvc remote remove staging

# Push (only sends commits the remote doesn't have yet)
kbvc push                      # push to origin
kbvc push staging              # push to staging
kbvc push --dry-run            # preview without writing
kbvc push --collection prod-v2 # override collection for this push
```

### Lineage and Audit

```bash
kbvc trace main__manifestai__chunk_2
# Output:
# Chunk        main__manifestai__chunk_2
# KO           manifestai (version 3)
# Section      'Technical Stack'
# Committed    a3f7c91  2026-05-15  'update funding round details'
# Embedded     text-embedding-3-small (1536 dims)
# Vector DB    qdrant

kbvc explain main__manifestai__chunk_2
# Full provenance including relations + confidence
```

### Health and Diagnostics

```bash
kbvc doctor                    # basic repo health
kbvc doctor --knowledge        # extended: stale, orphans, missing files,
                               # entity coverage, relation density

# Output includes:
# ✓  KBVC Repository  (format v1, repo_id a3f7c9...)
# ✓  Git Repository   (branch: main)
# ✓  Embed backend    openai
# ✓  VectorDB backend qdrant
# ✓  KO store         42 KO(s) tracked
# ✓  Commit history   18 commit(s)
# ── Knowledge Health ──────────────────
# ✓  No stale KOs
# ✓  No orphan relations
# ✓  All source files present
# ·  Entity coverage  12/42 KOs (29%)  (run: kbvc extract --all)
# ·  Relation density 0.4 relations/KO  (run: kbvc analyze)
```

---

## Supported Backends

### Embedding

| Backend | Install | Notes |
|---|---|---|
| OpenAI | `pip install kbvc[openai]` | `text-embedding-3-small` (1536d), `text-embedding-3-large` (3072d) |
| Google Gemini | `pip install kbvc[gemini]` | Uses `google-genai` SDK. `gemini-embedding-001`/`gemini-embedding-2` (3072d), `text-embedding-004` (768d) |
| Ollama (local) | `pip install kbvc[ollama]` | `nomic-embed-text`, any Ollama model |
| HuggingFace | `pip install kbvc[hf]` | `all-MiniLM-L6-v2` (384d), any sentence-transformers model |

### Vector Databases

| Backend | Install | Notes |
|---|---|---|
| Qdrant | `pip install kbvc[qdrant]` | UUID-based point IDs (no JSON serialisation issues) |
| pgvector | `pip install kbvc[pgvector]` | Supabase, Neon, RDS, self-hosted PostgreSQL |
| Pinecone | `pip install kbvc[pinecone]` | Serverless and pod-based indexes |
| ChromaDB | `pip install kbvc[chroma]` | Local-first, great for development |

---

## The `kbvc.lock` File

Commit `kbvc.lock` to your git repo. It records the exact knowledge infrastructure configuration — like `package-lock.json` for your knowledge base.

```yaml
# kbvc.lock — auto-generated by KBVC. Commit this file. Do not edit manually.
kbvc_version: "0.1.0"
format_version: 1
generated_at: "2026-05-15T09:31:00+00:00"
embedding:
  provider: "openai"
  model: "text-embedding-3-small"
  dims: 1536
vector_store:
  provider: "qdrant"
  collection: kbvc
retrieval:
  profile: vector
  hop_depth: 2
  top_k: 5
  weight_semantic: 0.7
  weight_graph: 0.3
```

---

## Retrieval Profiles

### Vector (default)
Standard nearest-neighbour search. Fast, simple.

### GraphRAG
Combines vector similarity with graph traversal. Returns semantically similar chunks **plus** related KOs via the relation graph.

```bash
kbvc config set retrieval.profile graphrag
kbvc config set retrieval.hop_depth 2
kbvc config set retrieval.weight_semantic 0.7
kbvc config set retrieval.weight_graph 0.3
```

Query flow:
```
query → embed → top-K vector results
                     ↓
              graph.neighbors(ko_id, depth=2)
                     ↓
              merged + re-ranked results
```

---

## Knowledge Intelligence Pipeline

### `kbvc analyze` — Relation Discovery

Automatically proposes relations between your KOs using heuristic (cross-reference, shared tags, keyword patterns) and vector similarity modes. Output capped at `--max-suggestions` to prevent graph explosion.

### `kbvc extract` — Entity Extraction

Detects named entities from KO content:

| Entity Type | Examples |
|---|---|
| `model` | GPT-4, Claude 3, LLaMA-2, Gemini, BERT, Mistral |
| `tool` | LangChain, ChromaDB, FastAPI, PyTorch, Transformers |
| `institution` | OpenAI, Anthropic, Google, Stanford, MIT, DeepMind |
| `concept` | RAG, fine-tuning, RLHF, vector database, attention mechanism |

### `kbvc stale` — Freshness Detection

A KO is stale when a `depends_on` dependency has been committed more recently than the KO itself. `kbvc stale --fix` stages all stale KOs for immediate recommit.

### `kbvc stats` — Evolution Analytics

Prints KO counts by type and volatility, relation density, most-changed and most-connected KOs, monthly growth, and branch overview.

---

## `kbvc backend` — VSAL Management

```bash
kbvc backend init              # idempotently create KBVC schema on configured backend
kbvc backend info              # show current backend configuration (no credentials)
```

---

## `kbvc migrate` — Migration Suite

### Backend-to-Backend

Moves all ChunkRecords from one vector store to another through the universal `ChunkRecord` schema — source and target backends never interact directly.

```bash
kbvc migrate backend --from qdrant --to pgvector
kbvc migrate backend --from qdrant --to pgvector --dry-run
```

### Embedding Model Swap (Zero Downtime)

Re-embeds the entire knowledge base with a new model, overwriting vectors in-place under the same IDs. Updates `kbvc.lock` on completion.

```bash
kbvc migrate embeddings \
    --from text-embedding-3-small \
    --to gemini-embedding-001
```

### Schema Version Bump

```bash
kbvc migrate schema           # bump schema_version in kbvc.lock
kbvc migrate schema --dry-run
```

---

## `kbvc explain` — Knowledge Provenance Chain

```bash
kbvc explain main__caching-strategy__chunk_2
```

Output:
```
vector_id:  main__caching-strategy__chunk_2
  ↓ chunk 2 of KO: caching-strategy @ v4
  ↓ section: Recommended Approach
  ↓ committed:   2026-05-10  commit: a3f7c91
  ↓ message:     "update caching recommendations"
  ↓ embedded with: gemini-embedding-001 (3072 dims)
  ↓ stored in: pgvector

Relations involving caching-strategy:
  supported_by          → benchmark-2026-05       [confidence: 1.00, source: human]
  supported_by          → incident-17              [confidence: 1.00, source: human]
  supersedes            → redis-config             [confidence: 1.00, source: human]
```

---

## `kbvc promote` — Memory → Knowledge Object Promotion

```bash
kbvc promote "Dragonfly fixed the Redis OOM issue in May 2026" \
    --id dragonfly-fix \
    --type lesson \
    --confidence 0.9 \
    --source agent

kbvc status          # review the staged .md file
kbvc commit -m "promote: dragonfly observation"
```

---

## `kbvc gc` — Garbage Collection

Removes orphaned vectors from the vector backend (vectors no longer referenced by any KO), and optionally prunes unreachable snapshot files.

```bash
kbvc gc              # remove orphaned vectors
kbvc gc --dry-run    # preview what would be deleted
kbvc gc --snapshots  # also prune unreachable snapshot files
```

---

## `kbvc sync` — Volatility-Aware Auto-Commit

```bash
kbvc sync                        # commit changed slow + live KOs
kbvc sync --volatility live      # only live KOs
kbvc sync --volatility all       # all non-frozen KOs
kbvc sync --dry-run              # show what would be committed
kbvc sync -m "nightly refresh"   # custom commit message
```

Frozen KOs are always excluded from sync, regardless of filter.

---

## `kbvc contradict` — Contradiction Detection

Scans `contradicts` relations and tracks resolution status. A contradiction is resolved once a `supersedes` relation exists between the two KOs.

```bash
kbvc contradict list
kbvc link winner.md loser.md --type supersedes
kbvc contradict resolve <rel_id>
```

---

## Relation Confidence and Source

Every relation carries:

| Field | Values | Default | Meaning |
|---|---|---|---|
| `confidence` | `0.0`–`1.0` | `1.0` | Certainty of this relation |
| `source` | `human` \| `auto` \| `agent` | `human` | How the relation was created |

---

## How Reproducibility Works

```bash
# Six months later, a RAG response was wrong
kbvc trace main__manifestai__chunk_3
# → committed 2026-03-15, model text-embedding-3-small, commit a3f7c91

# Restore the full state at that commit
kbvc checkout a3f7c91
# → Exact same retrieval results as six months ago
```

---

## Volatility Levels

| Level | Behaviour |
|---|---|
| `frozen` | Never re-embedded, even if content changes. Used for archived/historical KOs. |
| `slow` | Default. Re-embedded only when chunks change. |
| `live` | Always re-embedded on every commit. Used for frequently-updated KOs. |

---

## Installation

```bash
git clone https://github.com/Saiyam-Sandhir-Jain/kbvc
cd kbvc

pip install -e ".[openai]"          # OpenAI embeddings
pip install -e ".[openai,qdrant]"   # OpenAI + Qdrant
pip install -e ".[gemini,pgvector]" # Gemini + pgvector/Supabase
pip install -e ".[ollama,chroma]"   # fully local (no API keys)
pip install -e ".[all]"             # everything

kbvc --version
```

---

## Design Decisions

**Why SHA-256 commit IDs?** Same content committed twice produces the same hash. CI pipelines can detect knowledge changes with a simple hash comparison.

**Why JSON files, not a database?** `.kbvc/` is tracked by git → full history via `git log .kbvc/commits/`. Human-readable and debuggable.

**Why separate ingest from commit?** Ingested content is untrusted by default. The ingest → review → commit → push pipeline gives you a human checkpoint before any content reaches your vector database.

**Why a typed relation registry?** Core algorithmic relations (`depends_on`, `contradicts`, `supersedes`) need enforced semantics — transitivity, symmetry, inverseness. Custom relation types let teams model their specific domain without polluting core behaviour.

**Why not auto-generate all relations?** `kbvc analyze` suggests relations but never creates them silently. Auto-generating causes graph explosion and low-signal edges that hurt GraphRAG retrieval quality.

---

## Roadmap

| Phase | Status | Features |
|---|---|---|
| **Phase 1** | ✅ Complete | init, add, commit, log, config, basic chunking |
| **Phase 2** | ✅ Complete | status, checkout, diff, prompt versioning |
| **Phase 3** | ✅ Complete | link, graph, query, GraphRAG profile |
| **Phase 4** | ✅ Complete | branch, depends, impact, annotate, history, trace, doctor |
| **Phase 5** | ✅ Complete | ingest, push, remote, analyze, extract, stale, stats |
| **Phase 6** | ✅ Complete | VSAL `ChunkRecord` schema, `kbvc backend init/info`, `kbvc migrate backend`, `kbvc explain`, `kbvc promote`, relation `confidence`/`source`, `supersedes`/`supported_by` types |
| **Phase 7** | ✅ Complete | `kbvc gc`, `kbvc migrate embeddings`, `kbvc migrate schema` |
| **Phase 8** | ✅ Complete | `kbvc sync` (volatility-aware auto-commit) |
| **Phase 9** | ✅ Complete | `kbvc contradict list/resolve`, evidence-backed claims via `supported_by` |
| **Beyond P9** | ✅ Complete | Relation Registry (`kbvc relation list/show/create`), Gemini `google-genai` SDK migration, `kbvc clone` with setup checklist |
| **Next** | 🔜 Planned | `kbvc ask` (AI-assisted KM), `kbvc pull --rebuild`, branch merge, Python SDK, web UI |

---

## Test Coverage

```
195 tests · 0 failures · 0 errors
```

| Suite | Tests | What it covers |
|---|---|---|
| Chunker | 9 | Frontmatter parsing, split strategies, hash stability, diff helpers |
| KOStore | 8 | CRUD, persistence, entity round-trip, duplicate guard |
| CommitObject | 8 | SHA-256 determinism, DAG walk, prefix load, save/load |
| RelationGraph | 10 | Add/remove, BFS traversal, snapshot/restore, dirty flag |
| KOVersioner | 5 | Save/load, immutability guard, ordering |
| PromptVersionStore | 8 | Set/snapshot/restore, auto-default, dirty flag |
| RetrievalConfigStore | 2 | Snapshot, versioning |
| StagingIndex | 7 | Stage/unstage, idempotency, persistence |
| KbvcRepo | 10 | init, require (DAG walk), HEAD, config, directory tree |
| CLI Phase 1–4 | 45 | init, config, add, commit, log, status, checkout, diff, prompt, link, graph, query, branch, depends, impact, annotate, history, trace, doctor |
| Edge Cases | 11 | KO id consistency, DAG order, sorted staging, empty checkout, frozen guard, staleness warning |
| ChainTracer | 2 | Full trace after commit, invalid format |
| Ingest | 8 | Website/PDF/GitHub handlers, slugify, force flag |
| Push/Remote | 6 | No commits, no remote, add/list/remove, up-to-date, dry-run |
| Analyze | 6 | Min-KO guard, heuristic discovery, dedup, apply, max-cap |
| Extract | 7 | Model/tool detection, dedup, confidence filter, apply |
| Stale | 8 | Detection, orphan relations, fix flag, frozen exclusion |
| Stats | 5 | Empty repo, after commits, JSON output, most-changed, growth |
| Doctor --knowledge | 5 | Basic, missing files, orphans, entity hint, frozen count |
| GC | 5 | No-backend guard, dry-run orphan report, clean repo, delete orphan, snapshots flag |
| Migrate Embeddings | 5 | dry-run, no chunks, schema bump, schema dry-run, no lock guard |
| Sync | 4 | Nothing changed, detects changed KO, excludes frozen, volatility filter |
| Contradict | 6 | Empty repo, shows pair, resolved by supersedes, resolve no-supersedes error, resolve with supersedes, unknown rel_id |

---

*KBVC — because your AI is only as trustworthy as the knowledge it retrieves.*
