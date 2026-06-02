<div align="center">

<img src="https://img.shields.io/badge/KBVC-Knowledge%20OS-6366f1?style=for-the-badge&logoColor=white" alt="KBVC"/>

# KBVC — Knowledge Base Version Control

**The Knowledge Operating System for AI/RAG systems.**  
Version control, semantic graph, full audit trails, and reproducible deployment — for your knowledge base.

[![PyPI version](https://img.shields.io/pypi/v/kbvc.svg?style=flat-square&color=6366f1)](https://pypi.org/project/kbvc/)
[![Python](https://img.shields.io/pypi/pyversions/kbvc?style=flat-square)](https://pypi.org/project/kbvc/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/Saiyam-Sandhir-Jain/kbvc/ci.yml?style=flat-square&label=CI)](https://github.com/Saiyam-Sandhir-Jain/kbvc/actions)
[![Tests](https://img.shields.io/badge/tests-195%20passed-brightgreen?style=flat-square)](#)

</div>

---

## What is KBVC?

KBVC is a **Knowledge Operating System** — an infrastructure layer that sits between your documents and your vector database, giving your knowledge base the same discipline that Git gives your code.

It is **not** a RAG library. It is **not** a vector database.  
It is the **missing infrastructure layer** that answers questions no existing tool can:

| Question | Without KBVC | With KBVC |
|---|---|---|
| Which document version produced this embedding? | ❌ Unknown | ✅ `kbvc explain <vector_id>` |
| What was my knowledge state 3 months ago? | ❌ Gone | ✅ `kbvc checkout <commit>` |
| How do my documents relate to each other? | ❌ No idea | ✅ `kbvc graph` |
| Which documents are stale or contradictory? | ❌ Manual audit | ✅ `kbvc stale` / `kbvc contradict list` |
| Can I reproduce the exact RAG pipeline from last quarter? | ❌ Never | ✅ `kbvc.lock` |
| If my LLM gives a bad answer, what document caused it? | ❌ Guessing | ✅ `kbvc trace` + `kbvc explain` |

---

## Architecture Overview

```
 Your Documents (Markdown)
          │
          ▼
 ┌─────────────────────────────────────────────────────────────┐
 │                         KBVC                                │
 │                                                             │
 │  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐  │
 │  │  Commit DAG │  │ Knowledge    │  │  Prompt &         │  │
 │  │  (SHA-256)  │  │ Graph        │  │  Retrieval Store  │  │
 │  │             │  │ (Relations)  │  │  (per commit)     │  │
 │  └─────────────┘  └──────────────┘  └───────────────────┘  │
 │                                                             │
 │  ┌──────────────────────────────────────────────────────┐   │
 │  │       VSAL — Vector Storage Abstraction Layer        │   │
 │  │  ChunkRecord ──► Qdrant / pgvector / Chroma /        │   │
 │  │                   Pinecone / LanceDB                 │   │
 │  └──────────────────────────────────────────────────────┘   │
 │                                                             │
 │  ┌──────────────────────────────────────────────────────┐   │
 │  │        Embed Backends                                │   │
 │  │  OpenAI · Gemini · Ollama · HuggingFace              │   │
 │  └──────────────────────────────────────────────────────┘   │
 └─────────────────────────────────────────────────────────────┘
          │
          ▼
 Your RAG / LLM Application
```

---

## Feature Matrix

| Phase | Commands | What It Unlocks |
|---|---|---|
| **1** | `init` `add` `commit` `log` | Git-style commit loop for knowledge |
| **2** | `status` `checkout` `diff` | Time travel — restore any past state |
| **3** | `link` `query` `graph` | Knowledge graph + semantic search |
| **4** | `branch` `depends` `impact` `annotate` `history` `trace` `doctor` | Audit trails + dependency management |
| **5** | `ingest` `push` `remote` `analyze` `extract` `stale` `stats` | Multi-source ingestion + intelligence |
| **6** | `backend` `explain` `promote` | Infrastructure management + provenance |
| **7** | `gc` `migrate backend` `migrate schema` | Maintenance + zero-downtime ops |
| **8** | `sync` | Volatility-aware auto-commit |
| **9** | `contradict list/resolve` | Contradiction detection + resolution |
| **10** | `ask` | Grounded knowledge Q&A |

---

## Quick Start

### Install

```bash
# Core (pick your embedding + vector DB extras)
pip install kbvc[openai,qdrant]

# Local-first setup (no external server)
pip install kbvc[openai,lancedb]

# Everything
pip install kbvc[all]
```

### Initialize a Knowledge Repository

```bash
mkdir my-knowledge && cd my-knowledge
kbvc init

# Configure backends
kbvc config set embed.backend openai
kbvc config set embed.key sk-...
kbvc config set vectordb.backend lancedb   # no server needed
```

### Create Your First Knowledge Object

Every document tracked by KBVC is a **Knowledge Object (KO)** — a Markdown file with a YAML frontmatter header:

```markdown
---
id: caching-strategy
type: document
tags: [architecture, performance]
volatility: slow
source_type: file
---

## Caching Strategy

We use a two-tier cache: Redis for hot paths (TTL 60s) and
PostgreSQL materialized views for aggregate queries.

## Invalidation

Cache invalidation is triggered by...
```

```bash
kbvc add knowledge/caching-strategy.md
kbvc commit -m "Add caching strategy document"
```

### Explore

```bash
kbvc log                         # View commit history
kbvc status                      # See staged vs. committed state
kbvc query "how does caching work?"   # Semantic search
kbvc ask "what is the cache TTL?"     # Grounded Q&A with citations
kbvc graph                       # Visualize knowledge graph
kbvc stats                       # Analytics report
```

---

## Core Concepts

### Knowledge Objects (KOs)

A KO is the fundamental unit of KBVC. Each KO is:
- A Markdown file with YAML frontmatter
- Chunked into sections (one vector per section)
- Versioned independently (v1, v2, v3...)
- Connected to other KOs via typed relations

**Volatility levels** control how KBVC handles automatic commits:

| Volatility | Meaning | Auto-commit |
|---|---|---|
| `frozen` | Never changes (e.g. archived decisions) | Never re-embedded |
| `slow` | Changes occasionally (most documents) | On explicit `kbvc sync` |
| `live` | Changes frequently (e.g. meeting notes) | On every `kbvc sync` |

### Commit DAG

Every `kbvc commit` creates an immutable `CommitObject` — a SHA-256 hash over:
- Parent commit ID
- Branch name
- Commit message
- All changed KO IDs + their chunk hashes
- Snapshot filenames for graph, prompt, retrieval config

This gives you a fully reproducible, tamper-evident history — just like Git, but for knowledge.

### Knowledge Graph

KOs can be linked with typed relations:

```bash
kbvc link caching-strategy informed_by system-architecture
kbvc link new-api contradicts old-api
kbvc link auth-module extends base-security
```

**Relation types:** `informed_by` · `contradicts` · `extends` · `derived_from` · `supersedes` · `depends_on` · `related_to`

### VSAL — Vector Storage Abstraction Layer

KBVC never touches Qdrant points, pgvector rows, or Pinecone vectors directly. All storage goes through `ChunkRecord` — a universal unit that every backend adapter maps to its native representation.

This means you can migrate between vector databases **without re-embedding**:

```bash
kbvc migrate backend --from qdrant --to pgvector
```

---

## Knowledge Object Frontmatter Reference

```yaml
---
id: my-document              # required; used as ko_id
type: project                # project | education | patent | document | lesson | ...
tags: [ai, rag, python]      # free tags for filtering
volatility: slow             # frozen | slow | live
source_type: file            # file | web | github | pdf | notion | memory
valid_from: "2026-01-01"     # optional temporal validity window
valid_to: null               # null = still valid
---
```

---

## All Commands

### Repository Setup

```bash
kbvc init [--name NAME] [--no-git]   # Initialize a KBVC repo
kbvc clone <url> [directory]          # Clone a KBVC repo from Git
kbvc config set <key> <value>         # Set a config value
kbvc config get <key>                 # Read a config value
kbvc config list                      # List all config
```

### Knowledge Authoring

```bash
kbvc add <path|.>                     # Stage files for commit
kbvc add <path> --reason "why"       # Stage with an annotation
kbvc commit -m "message"             # Commit staged changes
kbvc commit -m "message" --dry-run   # Preview commit (no write)
kbvc status                          # Show staged vs. committed state
```

### History & Diff

```bash
kbvc log                             # Commit history (newest first)
kbvc log --oneline                   # Compact history
kbvc diff                            # Diff staged against HEAD
kbvc diff <commit_id>                # Diff against a specific commit
kbvc history <ko_id>                 # Version history for one KO
kbvc checkout <commit_id>            # Restore knowledge state
kbvc checkout <commit_id> --ko <id>  # Restore single KO
```

### Knowledge Graph

```bash
kbvc link <from> <type> <to>         # Create a relation
kbvc graph                           # Print graph summary
kbvc graph --dot                     # Export as DOT (Graphviz)
kbvc depends <ko_id>                 # Show KO dependencies
kbvc impact <ko_id>                  # What depends on this KO?
```

### Search & Q&A

```bash
kbvc query "natural language question"  # Vector search
kbvc query "..." --top-k 10             # Return 10 results
kbvc ask "natural language question"    # Grounded Q&A with citations
kbvc ask "..." --top-k 5 --show-ids     # Show raw vector IDs
```

### Intelligence & Analytics

```bash
kbvc analyze                         # Suggest relations between KOs
kbvc extract                         # Extract entities from KOs
kbvc stale                           # Find stale/orphaned KOs
kbvc contradict list                 # List contradictions
kbvc contradict resolve <rel_id>     # Resolve a contradiction
kbvc stats                           # Analytics report
```

### Provenance & Audit

```bash
kbvc trace <ko_id>                   # Full version trail for a KO
kbvc explain <vector_id>             # Trace vector → source → commit
kbvc annotate <ko_id> "reason"       # Add change reason before commit
kbvc doctor                          # Repo health check
```

### Ingestion

```bash
kbvc ingest website <url>            # Scrape a website
kbvc ingest github <repo_url>        # Import a GitHub repo
kbvc ingest pdf <file.pdf>           # Convert PDF to KO
kbvc ingest notion <page_url>        # Import a Notion page
kbvc ingest text <file.txt>          # Import a plain text file
```

### Branches & Remotes

```bash
kbvc branch                          # List branches
kbvc branch <name>                   # Create a branch
kbvc branch checkout <name>          # Switch branch
kbvc remote add <name> <url>         # Add a remote
kbvc push                            # Push to configured remote
```

### Infrastructure

```bash
kbvc backend init                    # Create vector DB schema
kbvc backend info                    # Show backend status
kbvc migrate backend --from X --to Y # Migrate vector store
kbvc migrate embeddings --from M --to M2  # Swap embedding model
kbvc migrate schema                  # Update lock file schema
kbvc gc                              # Garbage-collect orphan vectors
kbvc sync                            # Volatility-aware auto-commit
kbvc sync --volatility live          # Only sync live KOs
kbvc sync --dry-run                  # Preview what would be committed
kbvc promote "<memory text>"         # Promote agent memory to KO
```

---

## Backends

### Embedding Backends

| Backend | Install Extra | Models |
|---|---|---|
| **OpenAI** | `kbvc[openai]` | `text-embedding-3-small` (default), `text-embedding-3-large`, `ada-002` |
| **Gemini** | `kbvc[gemini]` | `gemini-embedding-001` |
| **Ollama** | `kbvc[ollama]` | any locally-served model (`nomic-embed-text`, etc.) |
| **HuggingFace** | `kbvc[hf]` | any `sentence-transformers` model |

```bash
# OpenAI
kbvc config set embed.backend openai
kbvc config set embed.key sk-...
kbvc config set embed.model text-embedding-3-small

# Ollama (local, no API key)
kbvc config set embed.backend ollama
kbvc config set embed.model nomic-embed-text
kbvc config set embed.url http://localhost:11434
```

### Vector DB Backends

| Backend | Install Extra | Best For |
|---|---|---|
| **LanceDB** | `kbvc[lancedb]` | Local dev, no server, embedded |
| **Qdrant** | `kbvc[qdrant]` | Production, cloud-native |
| **pgvector** | `kbvc[pgvector]` | PostgreSQL shops |
| **Chroma** | `kbvc[chroma]` | Local dev, Python-first |
| **Pinecone** | `kbvc[pinecone]` | Managed serverless |

```bash
# LanceDB (no server needed — best for local dev)
kbvc config set vectordb.backend lancedb
kbvc config set vectordb.url ./kbvc_lance

# Qdrant
kbvc config set vectordb.backend qdrant
kbvc config set vectordb.url http://localhost:6333
kbvc config set vectordb.collection kbvc

# pgvector
kbvc config set vectordb.backend pgvector
kbvc config set vectordb.url "postgresql://user:pass@localhost/mydb"
```

---

## The kbvc.lock File

Every commit writes a `kbvc.lock` file to your repo root — analogous to `package-lock.json`. It records:

```yaml
kbvc_version: "0.1.0"
embedding:
  provider: openai
  model: text-embedding-3-small
  dimensions: 1536
vector_store:
  provider: qdrant
  collection: kbvc
committed_at: "2026-01-15T10:30:00Z"
commit_id: "a3f2c1d9..."
```

Commit `kbvc.lock` to Git. Anyone cloning your repo with `kbvc clone` will see exactly what backend configuration was used and can reproduce it.

---

## Use Cases

### AI Product Teams
Track your RAG knowledge base like code. Every prompt change, every document update, every embedding model swap is versioned and auditable. Reproduce any past retrieval configuration exactly.

### Research Labs
Manage evolving scientific knowledge with contradiction detection and lineage tracking. Know when a newer paper supersedes an older one. Build multi-hop knowledge graphs connecting related findings.

### Enterprise Knowledge Management
Enforce governance over internal knowledge bases. Who changed what, when, and why. Detect when two documents contradict each other. Get alerts when downstream KOs depend on stale upstream content.

### LLM Application Developers
Debug why your LLM gave a specific answer — trace it back to the exact document chunk, its version, and the commit that created its embedding. Swap embedding models without rewriting your pipeline.

---

## Development Setup

```bash
git clone https://github.com/Saiyam-Sandhir-Jain/kbvc.git
cd kbvc
pip install -e ".[dev,openai,qdrant]"
pytest tests/ -v
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contributor guide.

---

## License

MIT © [Saiyam Jain](https://github.com/Saiyam-Sandhir-Jain)

---

<div align="center">
  <sub>Built by Saiyam Jain · VIT Bhopal</sub>
</div>
