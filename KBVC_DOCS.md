# KBVC — Complete Technical Reference

> **Version:** 0.1.0  
> **Author:** Saiyam Jain (VIT Bhopal)  
> **Repository:** https://github.com/Saiyam-Sandhir-Jain/kbvc

---

## Table of Contents

1. [What Is KBVC?](#1-what-is-kbvc)
2. [Architecture Deep Dive](#2-architecture-deep-dive)
3. [Installation & Setup](#3-installation--setup)
4. [Knowledge Objects (KOs)](#4-knowledge-objects-kos)
5. [The Commit Pipeline](#5-the-commit-pipeline)
6. [The Knowledge Graph](#6-the-knowledge-graph)
7. [Embedding Backends](#7-embedding-backends)
8. [Vector DB Backends](#8-vector-db-backends)
9. [Provenance & Audit Trail](#9-provenance--audit-trail)
10. [Multi-Source Ingestion](#10-multi-source-ingestion)
11. [Intelligence Layer](#11-intelligence-layer)
12. [Infrastructure Operations](#12-infrastructure-operations)
13. [Branches & Remotes](#13-branches--remotes)
14. [Volatility & Auto-Sync](#14-volatility--auto-sync)
15. [Contradiction Detection](#15-contradiction-detection)
16. [AI-Assisted Q&A (kbvc ask)](#16-ai-assisted-qa-kbvc-ask)
17. [The kbvc.lock File](#17-the-kbvclock-file)
18. [.kbvc/ Directory Layout](#18-kbvc-directory-layout)
19. [Internal Data Models](#19-internal-data-models)
20. [Design Invariants](#20-design-invariants)
21. [Configuration Reference](#21-configuration-reference)
22. [CLI Reference](#22-cli-reference)
23. [Extending KBVC](#23-extending-kbvc)

---

## 1. What Is KBVC?

KBVC (Knowledge Base Version Control) is a **Knowledge Operating System** — an infrastructure layer that sits between your Markdown documents and your vector database.

The simplest mental model: **Git for your knowledge base — and then some.**

But that undersells it. KBVC is a complete platform for treating knowledge as auditable, versionable, intelligently-managed infrastructure. Unlike Git (which tracks bytes), KBVC understands the *semantic* structure of your knowledge: how documents relate to each other, which ones contradict, which ones are stale, and which embedding model produced which vector.

### What KBVC Is NOT

- **Not a RAG library** — KBVC does not run LLM inference. It manages the knowledge that feeds your RAG system.
- **Not a vector database** — KBVC stores metadata and commit history. It delegates actual vector storage to pluggable backends (Qdrant, Chroma, etc.).
- **Not a document editor** — KBVC tracks Markdown files you write yourself.

### The Problem Space

Standard RAG pipelines have no answer to:

```
"Why did the LLM say that?"
  → Which chunk produced that answer?
  → Which document was that chunk from?
  → Which version of the document?
  → Which embedding model produced that vector?
  → Was that document current at query time?
```

KBVC answers all of these through its commit DAG, ChainTracer, and vector metadata.

---

## 2. Architecture Deep Dive

```
kbvc/
├── cli.py                  ← Click entry point (all commands wired here)
│
├── core/                   ← Pure Python data models (no I/O except file reads)
│   ├── repo.py             ← KbvcRepo — root object, HEAD, branch helpers
│   ├── ko.py               ← KnowledgeObject + KOStore (flat JSON registry)
│   ├── commit.py           ← CommitObject + walk_dag (SHA-256 DAG)
│   ├── chunker.py          ← Markdown → List[Chunk], hash comparison
│   ├── graph.py            ← RelationGraph — current.json + versioned snapshots
│   ├── index.py            ← StagingIndex — .kbvc/index (staged_files, dirty flags)
│   ├── versioner.py        ← KOVersioner — immutable .kbvc/ko_versions/<id>/vN.json
│   ├── prompt_store.py     ← PromptVersionStore — retrieval prompt versioning
│   ├── retrieval_store.py  ← RetrievalConfigStore — per-commit retrieval snapshots
│   └── lineage.py          ← ChainTracer — vector_id → CommitObject (pure read)
│
├── backends/               ← Pluggable infrastructure adapters
│   ├── __init__.py         ← get_embed_backend() + get_vectordb_backend() factories
│   ├── embed/
│   │   ├── openai.py       ← OpenAIEmbedBackend
│   │   ├── gemini.py       ← GeminiEmbedBackend
│   │   ├── ollama.py       ← OllamaEmbedBackend
│   │   └── huggingface.py  ← HuggingFaceEmbedBackend
│   └── vectordb/
│       ├── __init__.py     ← VectorDBBackend ABC + ChunkRecord (VSAL)
│       ├── qdrant.py       ← Full VSAL implementation
│       ├── pgvector.py     ← Full VSAL implementation
│       ├── chroma.py       ← Full VSAL implementation
│       ├── pinecone.py     ← Full VSAL implementation
│       └── lancedb.py      ← Full VSAL implementation (embedded, no server)
│
├── adapters/               ← Source type processors
│   ├── base.py             ← SourceAdapter ABC
│   └── text_file.py        ← TextFileAdapter (.md, .txt)
│
├── commands/               ← Business logic (called by cli.py)
│   ├── commit.py           ← 12-step commit pipeline
│   ├── ingest.py           ← website / github / pdf / notion / text
│   ├── analyze.py          ← suggest_relations() + extract_entities()
│   ├── push.py             ← run_push() — DAG-aware push to remote
│   ├── gc.py               ← run_gc() — orphan vector cleanup
│   ├── stale.py            ← compute_staleness() + detect_orphans()
│   ├── stats.py            ← compute_stats() → StatsReport
│   ├── contradict.py       ← detect_contradictions() + resolve_contradiction()
│   ├── sync.py             ← run_sync() — volatility-aware auto-commit
│   ├── promote.py          ← agent memory → KO promotion
│   ├── explain.py          ← vector provenance chain
│   └── migrate*.py         ← backend / embeddings / schema migrations
│
└── utils/
    ├── config.py           ← read_config() / write_config_key()
    ├── display.py          ← rich-powered terminal output helpers
    └── lock.py             ← write_lock_file() — kbvc.lock writer
```

### Data Flow Diagram

```
Write path (kbvc commit):
  Markdown files
       │
       ▼
  parse_frontmatter()   ← YAML frontmatter extraction
       │
       ▼
  split_into_chunks()   ← Markdown section chunker
       │
       ▼
  diff against stored   ← Only re-embed changed chunks
  chunk hashes          
       │
       ▼
  embed.embed_batch()   ← Embedding backend (OpenAI / Gemini / etc.)
       │
       ▼
  vdb.upsert_batch()    ← Vector DB backend (VSAL ChunkRecord)
       │
       ▼
  CommitObject (SHA-256) ← Deterministic hash over content
       │
       ▼
  .kbvc/commits/<hash>.json
  .kbvc/ko_versions/<id>/vN.json
  kbvc.lock

Read path (kbvc query / ask):
  Natural language question
       │
       ▼
  embed.embed()          ← Same backend as write
       │
       ▼
  vdb.query()            ← ANN search with branch filter
       │
       ▼
  chunk text + metadata  ← Loaded from source files + vector metadata
       │
       ▼
  Citations: ko_id, version, commit_id, score
```

---

## 3. Installation & Setup

### Install

```bash
# Minimal (no embedding or vector DB — useful for testing CLI)
pip install kbvc

# Recommended local setup (OpenAI + LanceDB, no server)
pip install kbvc[openai,lancedb]

# Production setup (OpenAI + Qdrant)
pip install kbvc[openai,qdrant]

# PostgreSQL shop
pip install kbvc[openai,pgvector]

# All optional backends
pip install kbvc[all]
```

### Initialize a Repository

```bash
mkdir my-knowledge && cd my-knowledge
kbvc init
# Creates: .kbvc/ directory, HEAD, config, ko_store.json, etc.
# Also runs: git init (pass --no-git to skip)
```

### Configure Backends

```bash
# Embedding
kbvc config set embed.backend openai
kbvc config set embed.key sk-...
kbvc config set embed.model text-embedding-3-small   # default

# Vector DB (LanceDB — no server needed)
kbvc config set vectordb.backend lancedb
kbvc config set vectordb.url ./kbvc_lance
kbvc config set vectordb.collection kbvc

# Initialize the schema
kbvc backend init
```

### Verify Setup

```bash
kbvc doctor
# Checks: config, backend connectivity, index integrity, lock file
```

---

## 4. Knowledge Objects (KOs)

A **Knowledge Object** is the fundamental atom of KBVC. It is a Markdown file with YAML frontmatter that declares its identity and metadata.

### Frontmatter Schema

```yaml
---
id: my-document              # REQUIRED — becomes the ko_id
type: document               # project | education | patent | document | lesson | decision | ...
tags: [ai, rag, python]      # free-form tags
volatility: slow             # frozen | slow | live
source_type: file            # file | web | github | pdf | notion | memory
valid_from: "2026-01-01"     # optional: ISO-8601 date when this KO becomes valid
valid_to: null               # optional: null = still valid, or ISO-8601 end date
---
```

**`id`** is the most important field. It is the `ko_id` used throughout KBVC — in commits, graph relations, vector IDs, and version snapshots. If omitted, KBVC derives it from the filename: `my-doc.md` → `my-doc`.

**`volatility`** controls re-embedding behavior:
- `frozen` — never re-embedded, even on `kbvc sync`
- `slow` — re-embedded only on explicit `kbvc add` or `kbvc sync --volatility slow`
- `live` — re-embedded on every `kbvc sync` when content changes

### KO Lifecycle

```
1. Author writes knowledge/my-doc.md with frontmatter
2. kbvc add knowledge/my-doc.md     → StagingIndex updated
3. kbvc commit -m "add doc"         → KO embedded + versioned
4. Edit the file
5. kbvc add knowledge/my-doc.md
6. kbvc commit -m "update doc"      → Only changed chunks re-embedded; version → v2
7. kbvc history my-doc              → Shows v1 and v2
8. kbvc checkout <old_commit>       → Restore v1 state
```

### KO Version Snapshots

Every commit creates an immutable `KOVersionSnapshot` at `.kbvc/ko_versions/<id>/vN.json`:

```json
{
  "ko_id": "caching-strategy",
  "version": 2,
  "commit_id": "a3f2c1d9...",
  "path": "knowledge/caching-strategy.md",
  "chunk_hashes": ["abc123...", "def456..."],
  "volatility": "slow",
  "type": "document",
  "tags": ["architecture", "performance"],
  "ko_snapshot_name": "v2.json"
}
```

These snapshots are immutable by design — once written, they are never overwritten. This guarantees tamper-evident history.

---

## 5. The Commit Pipeline

`kbvc commit` runs a 12-step pipeline in `commands/commit.py`:

```
Step  1: Resolve embed + vectordb backends from config
Step  2: For each staged file:
            parse frontmatter → derive ko_id
            chunk the document
            diff chunks against stored hashes
            embed only changed/new chunks
            upsert to vector DB
            delete removed chunk vectors
            update KOStore
Step  3: Snapshot graph → graph-vN.json (immutable)
Step  4: Snapshot prompt → p-vN.json (commit_id = "pending")
Step  5: Snapshot retrieval config → r-vN.json (commit_id = "pending")
Step  6: Create CommitObject
            SHA-256 over: parent + branch + message + ko_changes + snapshot_names
            Timestamp NOT included in hash (content-addressable, like Git)
Step  7: Back-fill commit_id
            Rewrite p-vN.json + r-vN.json with real commit_id
            vdb.patch_metadata() on all new vectors
Step  8: Persist KO version snapshots → ko_versions/<id>/vN.json
Step  9: Save commit file → commits/<hash>.json
            Advance HEAD → refs/heads/<branch>
Step 10: Write kbvc.lock
Step 11: Clear staging index
Step 12: Print summary
```

**Critical invariant:** The commit hash **excludes the timestamp**. Same content committed twice → same hash. This is intentional — it mirrors Git's content-addressability and ensures reproducibility.

---

## 6. The Knowledge Graph

The knowledge graph is a directed, typed graph of `Relation` objects connecting KOs.

### Creating Relations

```bash
kbvc link <from_ko_id> <relation_type> <to_ko_id>
kbvc link caching-strategy informed_by system-architecture
kbvc link v2-api contradicts v1-api
kbvc link auth-module extends base-security
```

**Standard relation types:**

| Type | Meaning |
|---|---|
| `informed_by` | This KO's content is shaped by the target |
| `contradicts` | This KO makes conflicting claims with the target |
| `extends` | This KO builds on or adds to the target |
| `derived_from` | This KO was generated or extracted from the target |
| `supersedes` | This KO replaces the target (target is obsolete) |
| `depends_on` | This KO requires the target to be valid |
| `related_to` | General semantic relationship |

### Graph Commands

```bash
kbvc graph                       # Summary: node count, edge count
kbvc graph --dot                 # Export as DOT format (pipe to Graphviz)
kbvc graph --dot | dot -Tpng > graph.png

kbvc depends caching-strategy    # What does caching-strategy depend on?
kbvc impact system-architecture  # What KOs would be affected if I change this?
```

### Graph Storage

The working graph lives at `.kbvc/graph/current.json`. Every commit snapshots it to `.kbvc/graph/graph-vN.json` — immutable. The graph is included in the commit hash, so changing graph relations and committing creates a new commit ID.

---

## 7. Embedding Backends

All backends implement the `EmbedBackend` ABC:

```python
class EmbedBackend(ABC):
    @property
    def dimensions(self) -> int: ...
    @property
    def model_name(self) -> str: ...
    def embed(self, text: str) -> List[float]: ...
    def embed_batch(self, texts: List[str]) -> List[List[float]]: ...
```

### OpenAI

```bash
kbvc config set embed.backend openai
kbvc config set embed.key sk-...
kbvc config set embed.model text-embedding-3-small   # or text-embedding-3-large, ada-002
```

| Model | Dimensions | Cost |
|---|---|---|
| `text-embedding-3-small` | 1536 | Cheapest |
| `text-embedding-3-large` | 3072 | Best quality |
| `text-embedding-ada-002` | 1536 | Legacy |

### Gemini

```bash
kbvc config set embed.backend gemini
kbvc config set embed.key AIza...
kbvc config set embed.model gemini-embedding-001
```

### Ollama (Local, Free)

```bash
# Start Ollama first: ollama serve
kbvc config set embed.backend ollama
kbvc config set embed.model nomic-embed-text
kbvc config set embed.url http://localhost:11434
```

### HuggingFace (Local, Free)

```bash
kbvc config set embed.backend huggingface
kbvc config set embed.model all-MiniLM-L6-v2
```

---

## 8. Vector DB Backends

All backends implement the `VectorDBBackend` ABC through the VSAL layer. Every backend supports:

- `upsert` / `upsert_batch` — write vectors
- `delete` / `delete_by_prefix` — remove vectors
- `query` — ANN search with optional metadata filter
- `patch_metadata` — update metadata without touching vectors
- `exists_batch` — check which vector IDs exist
- `initialize_schema` — idempotent schema creation
- `export_chunks` — full export as `List[ChunkRecord]`

### LanceDB (Recommended for local development)

```bash
pip install kbvc[lancedb]
kbvc config set vectordb.backend lancedb
kbvc config set vectordb.url ./kbvc_lance    # local path
# Or cloud: s3://my-bucket/kbvc
```

No server required. Data stored in a local directory (Arrow columnar format). Ideal for laptops and single-machine deployments.

### Qdrant

```bash
# Start: docker run -p 6333:6333 qdrant/qdrant
pip install kbvc[qdrant]
kbvc config set vectordb.backend qdrant
kbvc config set vectordb.url http://localhost:6333
kbvc config set vectordb.collection kbvc
```

### pgvector

```bash
# Needs PostgreSQL with pgvector extension
pip install kbvc[pgvector]
kbvc config set vectordb.backend pgvector
kbvc config set vectordb.url "postgresql://user:pass@localhost/mydb"
kbvc config set vectordb.collection kbvc
```

### Chroma

```bash
pip install kbvc[chroma]
kbvc config set vectordb.backend chroma
kbvc config set vectordb.url ./chroma_db
```

### Pinecone

```bash
pip install kbvc[pinecone]
kbvc config set vectordb.backend pinecone
kbvc config set vectordb.key pc-...
kbvc config set vectordb.collection kbvc
```

---

## 9. Provenance & Audit Trail

This is one of KBVC's most powerful and unique capabilities: **complete traceability from any vector back to its source**.

### Vector ID Format

Every vector stored by KBVC has a deterministic ID:

```
<branch>__<ko_id>__chunk_<N>
```

Examples:
```
main__caching-strategy__chunk_0
main__caching-strategy__chunk_1
feature/auth__auth-module__chunk_0
```

The double-underscore is the delimiter. Single underscores may appear within branch names or ko_ids.

### Trace a Vector Back to Source

```bash
kbvc explain main__caching-strategy__chunk_1
```

Output:
```
Vector Provenance Chain
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  vector_id:   main__caching-strategy__chunk_1
  ko_id:       caching-strategy
  ko_version:  3
  chunk_index: 1
  commit_id:   a3f2c1d9...

Commit details:
  branch:  main
  message: "Update caching TTL from 30s to 60s"
  parent:  7b8e9f01...

Embedding config at commit time:
  provider: openai
  model:    text-embedding-3-small
  dims:     1536
```

### KO Version History

```bash
kbvc history caching-strategy
```

Shows all versions of a KO with their commit IDs and timestamps.

### Full Audit Trail

```bash
kbvc trace caching-strategy
```

Walks the entire commit DAG and shows every change ever made to this KO.

---

## 10. Multi-Source Ingestion

KBVC can ingest knowledge from multiple source types:

### Website

```bash
kbvc ingest website https://docs.example.com/architecture
# Scrapes the page, converts to Markdown, creates a KO
```

### GitHub Repository

```bash
kbvc ingest github https://github.com/org/repo
# Imports README + Markdown files from the repo
```

### PDF

```bash
kbvc ingest pdf path/to/document.pdf
# Converts PDF to Markdown, creates a KO
```

### Notion

```bash
kbvc ingest notion https://notion.so/page-id
# Requires NOTION_TOKEN environment variable
```

### Plain Text

```bash
kbvc ingest text path/to/file.txt --id my-doc --type document
```

All ingested KOs are created with `source_type` set appropriately in their frontmatter and are staged automatically for the next commit.

---

## 11. Intelligence Layer

### Semantic Relation Analysis

```bash
kbvc analyze
```

Scans all committed KOs and suggests relations based on:
- Shared tags
- Shared entity names (extracted from content)
- Title/section similarity (heuristic v1; LLM-backed in v2)

Outputs suggested `kbvc link` commands that you can review and run.

### Entity Extraction

```bash
kbvc extract
kbvc extract --ko caching-strategy   # single KO
```

Extracts named entities (technologies, concepts, products) from KO content and outputs them per-KO. Use `--apply` to write entities back to frontmatter tags.

### Staleness Detection

```bash
kbvc stale
```

Identifies:
- KOs whose `valid_to` date has passed
- KOs that depend on other KOs which have been superseded
- Orphaned relations pointing to non-existent KOs

### Knowledge Statistics

```bash
kbvc stats
```

Outputs:
- Total KOs, commits, relations
- Embedding cost per commit
- Most-connected KO
- Stale KO count
- Knowledge growth over time

---

## 12. Infrastructure Operations

### Zero-Downtime Backend Migration

Migrate all vectors from one backend to another without re-embedding:

```bash
kbvc migrate backend --from qdrant --to pgvector
kbvc migrate backend --from chroma --to lancedb --dry-run   # preview
```

All `ChunkRecord` data (including embeddings) is exported from the source and imported to the target. No embedding API calls.

### Zero-Downtime Embedding Model Swap

When you want to upgrade your embedding model:

```bash
kbvc migrate embeddings \
    --from text-embedding-3-small \
    --to text-embedding-3-large
```

This re-embeds all KOs using the new model, updates vector metadata, and writes a new commit. The old vectors are cleaned up by `kbvc gc`.

### Schema Migration

When KBVC format changes between versions:

```bash
kbvc migrate schema
```

Updates `kbvc.lock` to the current schema version. Usually handled automatically.

### Garbage Collection

```bash
kbvc gc
kbvc gc --dry-run
```

Removes orphaned vectors — vectors in the DB that no longer correspond to any committed KO chunk. Happens when KOs are deleted or chunks are removed during a commit.

### Agent Memory Promotion

Promote a transient agent memory into a permanent KO:

```bash
kbvc promote "The rate limit for the OpenAI API is 10,000 RPM on tier 2" \
    --id api-rate-limits \
    --type document \
    --tags [api, limits, openai]
```

Creates a properly-formatted KO file and stages it for the next commit.

---

## 13. Branches & Remotes

### Branch Operations

```bash
kbvc branch                          # List all branches
kbvc branch feature/new-auth         # Create branch
kbvc branch checkout feature/new-auth # Switch to branch

# Each branch has its own vector namespace in the DB
# Vector IDs are prefixed with the branch name
```

### Remotes & Push

```bash
kbvc remote add origin https://github.com/org/knowledge-base.git
kbvc push
kbvc push --remote origin --branch main
```

`kbvc push` re-embeds all KOs and pushes vectors to the configured vector DB backend. It respects the commit DAG and only pushes commits newer than the remote's last known state.

### Cloning

```bash
kbvc clone https://github.com/org/knowledge-base.git
cd knowledge-base
# KBVC reads kbvc.lock and shows you exactly what config to set
kbvc config set embed.key sk-...
kbvc backend init
kbvc push   # rebuild vector state
```

---

## 14. Volatility & Auto-Sync

The `sync` command implements volatility-aware auto-commit:

```bash
kbvc sync                         # Commit changed slow + live KOs
kbvc sync --volatility live       # Only live KOs
kbvc sync --volatility all        # All non-frozen KOs
kbvc sync --dry-run               # Show what would change
kbvc sync --message "nightly"     # Custom commit message
```

**How it works:**

1. Scans all KOs in `ko_store.json`
2. For each KO matching the volatility filter:
   - Re-chunks the source file
   - Compares chunk hashes against the last committed state
   - If any chunk changed → mark as candidate
3. Stages all candidates
4. Runs `run_commit()` with a generated message

Typical usage: run `kbvc sync` as a cron job or CI step for nightly knowledge base updates.

---

## 15. Contradiction Detection

```bash
kbvc contradict list
```

Scans the knowledge graph for `contradicts` relations and lists them with context from both KOs.

```bash
kbvc contradict resolve <rel_id>
```

Resolves a contradiction by removing the relation and optionally updating the `valid_to` field of the superseded KO.

**How contradictions are detected automatically:**

`kbvc analyze` detects potential contradictions by:
1. Comparing tags between KOs — same tags on documents with opposing language patterns
2. Checking for `supersedes` relations — if A supersedes B, B may contradict A
3. Temporal validity — if KO A and KO B cover the same topic but have overlapping validity windows

---

## 16. AI-Assisted Q&A (kbvc ask)

`kbvc ask` closes the loop: KBVC manages your knowledge, and `kbvc ask` lets you query it with full provenance.

```bash
kbvc ask "What is the cache TTL for hot paths?"
kbvc ask "Which projects use transformers?" --top-k 10
kbvc ask "Summarise the auth flow" --branch feature/auth
kbvc ask "What are the rate limits?" --show-ids
```

**How it works:**

1. Embeds your question using the configured backend
2. Performs ANN search against the active branch's vectors
3. Loads chunk text from source files
4. Displays top-K results with:
   - KO ID and version
   - Commit ID (short hash)
   - Similarity score
   - Chunk preview (600 chars)

**This is NOT an LLM call.** `kbvc ask` returns the raw retrieved context — the grounding material. Feed this context to your LLM of choice to generate a final answer. This design is intentional: KBVC manages knowledge; your application manages generation.

For a full audit trail of any returned result: `kbvc explain <vector_id>`

---

## 17. The kbvc.lock File

`kbvc.lock` is the reproducibility guarantee for your knowledge base:

```yaml
kbvc_version: "0.1.0"
format_version: "1"
embedding:
  provider: openai
  model: text-embedding-3-small
  dimensions: 1536
vector_store:
  provider: qdrant
  collection: kbvc
committed_at: "2026-01-15T10:30:00Z"
commit_id: "a3f2c1d9e5f7b2a4c6d8e0f1a3b5c7d9..."
branch: main
```

**Rules:**
- Written on every `kbvc commit`
- **Must be committed to Git** — it is the deployment manifest for your knowledge base
- Never contains API keys (only provider/model names)
- Anyone cloning the repo sees exactly which embedding model + vector store was used

**Analogy:** `kbvc.lock` : `package-lock.json` :: `kbvc commit` : `npm install`

---

## 18. .kbvc/ Directory Layout

```
.kbvc/
├── repo.json               ← Repository identity: repo_id, format_version, name
├── HEAD                    ← "ref: refs/heads/main" or bare commit hash
├── config                  ← INI config (embed.*, vectordb.*, retrieval.*, etc.)
├── index                   ← JSON staging area
├── ko_store.json           ← Flat list of all KnowledgeObjects
│
├── refs/
│   ├── heads/
│   │   ├── main            ← Full SHA-256 of HEAD commit on main
│   │   └── <branch>        ← One file per branch
│   └── remotes/
│       └── <remote_name>   ← Last pushed commit_id per remote
│
├── commits/
│   └── <sha256>.json       ← CommitObject (parent, branch, message, KO changes)
│
├── ko_versions/
│   └── <ko_id>/
│       └── vN.json         ← Immutable KOVersionSnapshot per version
│
├── graph/
│   ├── current.json        ← Mutable working graph (list of Relation dicts)
│   └── graph-vN.json       ← Immutable per-commit snapshots
│
├── prompts/
│   ├── current.json        ← Mutable working prompt
│   └── p-vN.json           ← Immutable per-commit snapshots
│
├── retrieval/
│   └── r-vN.json           ← Immutable per-commit retrieval config snapshots
│
├── migrations/             ← Reserved for migration logs
└── objects/                ← v3 shared object store (currently empty)
```

---

## 19. Internal Data Models

### KnowledgeObject

```python
@dataclass
class KnowledgeObject:
    id: str                    # ko_id (from frontmatter or filename)
    path: str                  # relative path from repo root
    version: int               # current version number
    chunk_hashes: List[str]    # SHA-256[:16] of each current chunk
    volatility: str            # frozen | slow | live
    type: str                  # document type
    tags: List[str]
    source_type: str           # file | web | github | pdf | notion | memory
    valid_from: Optional[str]
    valid_to: Optional[str]
    branch: str
```

### CommitObject

```python
@dataclass
class CommitObject:
    commit_id: str             # SHA-256 over content (not timestamp)
    parent: Optional[str]      # parent commit_id (None for first commit)
    branch: str
    message: str
    changed_kos: List[KOChange]
    graph_snapshot: str        # filename of graph-vN.json
    prompt_snapshot: str       # filename of p-vN.json
    retrieval_snapshot: str    # filename of r-vN.json
    timestamp: str             # ISO-8601 UTC (NOT included in hash)
```

### ChunkRecord (VSAL)

```python
@dataclass
class ChunkRecord:
    vector_id: str             # "<branch>__<ko_id>__chunk_<N>"
    branch: str
    ko_id: str
    ko_version: int
    chunk_index: int
    chunk_hash: str            # SHA-256[:16] of enriched chunk text
    embedding: List[float]     # the actual vector
    metadata: Dict[str, Any]   # extra fields (commit_id, etc.)
    created_at: str            # ISO-8601 UTC
```

---

## 20. Design Invariants

These are non-negotiable. Violating any of them breaks audit integrity.

1. **Commit hash excludes timestamp.** Only content determines the hash. Same content committed twice → same hash. Intentional.

2. **`ko_id` derivation:** Always `frontmatter.get("id") or path.stem.replace(" ", "-").lower()`. Every command uses `_path_to_ko_id()` from `commands/commit.py`.

3. **Vector ID format:** Always `<branch>__<ko_id>__chunk_<N>`. Double underscore is the delimiter. `ChainTracer._parse_vector_id()` depends on this format.

4. **Version snapshots are immutable.** `KOVersioner.save_version()` raises `FileExistsError` if the file exists.

5. **Graph snapshots are immutable.** Once written as `graph-vN.json`, never modified.

6. **`commit_id` starts as `"pending"`.** Back-filled after the CommitObject hash is known (Step 7).

7. **`kbvc.lock` never contains API keys.** Only provider names and model identifiers.

8. **Graph mutations never trigger re-embedding.** `kbvc link` marks `graph_dirty = True` in the index but doesn't touch vectors.

9. **Frozen KOs are never re-embedded.** `volatility == "frozen"` skips embedding unconditionally.

10. **Staging order is deterministic.** `kbvc add .` uses `sorted(repo.root.rglob("*.md"))`.

---

## 21. Configuration Reference

All configuration lives at `.kbvc/config` (INI format).

### Embedding

| Key | Default | Description |
|---|---|---|
| `embed.backend` | — | `openai` \| `gemini` \| `ollama` \| `huggingface` |
| `embed.key` | — | API key (not stored in kbvc.lock) |
| `embed.model` | backend default | Model identifier |
| `embed.url` | — | Base URL (Ollama only) |

### Vector DB

| Key | Default | Description |
|---|---|---|
| `vectordb.backend` | — | `lancedb` \| `qdrant` \| `pgvector` \| `chroma` \| `pinecone` |
| `vectordb.url` | — | Connection URL or path |
| `vectordb.key` | — | API key (Pinecone only) |
| `vectordb.collection` | `kbvc` | Collection/index name |

### Retrieval

| Key | Default | Description |
|---|---|---|
| `retrieval.profile` | `vector` | `vector` \| `graph` \| `hybrid` |
| `retrieval.top_k` | `5` | Default number of results |
| `retrieval.hop_depth` | `2` | Graph traversal depth (GraphRAG) |

### Chunking

| Key | Default | Description |
|---|---|---|
| `chunk.min_tokens` | `20` | Minimum tokens per chunk |
| `chunk.max_tokens` | `400` | Maximum tokens per chunk |

---

## 22. CLI Reference

See `README.md` for the full command list. Run `kbvc <command> --help` for any command.

```bash
kbvc --help              # All top-level commands
kbvc commit --help       # Commit-specific options
kbvc migrate --help      # Migration subcommands
kbvc ingest --help       # Ingestion subcommands
kbvc contradict --help   # Contradiction subcommands
```

---

## 23. Extending KBVC

### Adding a New Embedding Backend

1. Create `src/kbvc/backends/embed/<name>.py`
2. Subclass `EmbedBackend` from `kbvc.backends.embed`
3. Implement `dimensions`, `model_name`, `embed()`, `embed_batch()`, `from_config()`
4. Register in `get_embed_backend()` in `kbvc/backends/__init__.py`
5. Add optional-dependency to `pyproject.toml`

### Adding a New Vector DB Backend

1. Create `src/kbvc/backends/vectordb/<name>.py`
2. Subclass `VectorDBBackend` from `kbvc.backends.vectordb`
3. Implement all abstract methods + `initialize_schema` + `export_chunks` + `from_config()`
4. Register in `get_vectordb_backend()` in `kbvc/backends/__init__.py`
5. Add optional-dependency to `pyproject.toml`

### Adding a New Source Adapter

1. Create `src/kbvc/adapters/<name>.py`
2. Subclass `SourceAdapter` from `kbvc.adapters.base`
3. Implement `can_process()` and `extract_chunks()`
4. Register in `ADAPTER_REGISTRY` in `kbvc/adapters/__init__.py`

---

*KBVC is built and maintained by Saiyam Jain. Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).*
