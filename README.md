# KBVC — Knowledge Base Version Control

> **Git for your RAG pipeline.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-195%20passing-brightgreen.svg)](tests/)
[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](pyproject.toml)

KBVC is a version control system for AI knowledge bases. It sits between your `.md` knowledge files and your vector database, giving every document, chunk, embedding decision, relation, and prompt a commit history — a complete audit trail and reproducibility guarantee for your RAG pipeline.

```
Without KBVC:                     With KBVC:

docs/ (changed silently)          kbvc add .
      ↓                           kbvc commit -m "update API docs"
vector DB (stale vectors,    →    .kbvc/commits/a3f7c91
           no audit trail,             KOs: api-docs v1→v2 (2 chunks re-embedded)
           non-reproducible)           Graph: graph-v4
                                       Prompt: p-v2
                                  vector DB (only changed chunks)
```

---

## Why KBVC?

| Problem | Without KBVC | With KBVC |
|---|---|---|
| Document updated | Old vectors persist silently | Only changed chunks re-embedded |
| Wrong AI answer | Cannot trace cause | `kbvc trace` → commit → model → section |
| Reproduce last week's result | Impossible | `kbvc checkout <hash>` |
| Team collaboration | No workflow | Branch → review → commit → push |
| Embedding model swap | Re-embed everything manually | `kbvc migrate embeddings --to new-model` |
| Orphaned vectors accumulate | No cleanup mechanism | `kbvc gc` |

---

## Install

```bash
# Base install (no backends — good for exploring)
pip install kbvc

# With your embedding + vector DB backend
pip install "kbvc[openai,qdrant]"       # OpenAI + Qdrant
pip install "kbvc[gemini,pgvector]"     # Gemini + pgvector / Supabase
pip install "kbvc[ollama,chroma]"       # Fully local, no API keys

# Everything
pip install "kbvc[openai,gemini,ollama,hf,qdrant,pgvector,pinecone,chroma]"
```

### Supported backends

| Layer | Options |
|---|---|
| Embedding | OpenAI · Google Gemini (`google-genai` SDK) · Ollama (local) · HuggingFace |
| Vector DB | Qdrant · pgvector · Pinecone · ChromaDB |

> **Gemini users:** KBVC uses the new `google-genai` SDK (the old `google-generativeai` reached end-of-life Nov 2025). See [GEMINI_MIGRATION.md](GEMINI_MIGRATION.md) if upgrading.

---

## Quickstart

```bash
# 1. Initialise in your knowledge directory
cd my-knowledge-base
kbvc init

# 2. Configure backends
kbvc config set embed.backend openai
kbvc config set embed.key sk-...
kbvc config set vectordb.backend qdrant
kbvc config set vectordb.url http://localhost:6333

# 3. Write a knowledge file
cat > projects/my-project.md << EOF
---
id: my-project
type: project
tags: [ai, llm]
volatility: slow
---

## Overview

My project uses LangChain and ChromaDB to build a RAG system.

## Technical Stack

Built on the Transformer architecture, using OpenAI embeddings.
EOF

# 4. Stage, embed, commit
kbvc add projects/my-project.md
kbvc commit -m "Initial knowledge base"

# 5. Query
kbvc query "what is the tech stack?"

# 6. Discover relations between KOs automatically
kbvc analyze

# 7. Push to production
kbvc remote add origin --backend qdrant --url https://prod.example.com --collection prod
kbvc push
```

---

## Core Concepts

### Knowledge Objects (KOs)
One `.md` file = one Knowledge Object. Each KO carries a stable `id`, a `type`, `tags`, and a `volatility` level (`frozen` / `slow` / `live`) that controls when it gets re-embedded.

### Commits
Every `kbvc commit` creates a global snapshot — not just per-file — capturing changed KOs, the knowledge graph, the prompt, and the retrieval configuration. The commit ID is a deterministic SHA-256: same content always produces the same hash.

### Chunk-Level Versioning
KBVC hashes every chunk. On commit, only chunks whose hash changed are sent to the embedding API. For a 50-section document where 2 sections changed: 2 API calls, not 50.

### The Ingest Pipeline
```
kbvc ingest website https://docs.example.com   # download → .md (no vectors touched)
kbvc add ingested/                              # stage for review
kbvc commit -m "ingest example docs"            # embed + commit locally
kbvc push                                       # sync to remote
```
Untrusted content never reaches your vector database until you commit it.

### VSAL — Vector Storage Abstraction Layer
All storage goes through a universal `ChunkRecord` schema. The backend is an implementation detail — migrate between Qdrant, pgvector, Pinecone, and ChromaDB with a single command:
```bash
kbvc migrate backend --from qdrant --to pgvector
```

### Relation Registry
KBVC ships a typed relation system with two built-in tiers:
- **Core relations** — affect KBVC internals (`depends_on`, `contradicts`, `supersedes`, `supported_by`). KBVC enforces their semantic properties (transitive, symmetric, inverse).
- **Temporal relations** — pre-defined convenience types (`studied_at`, `worked_at`, `developed_during`) for graph traversal and human context.

You can also define your own:
```bash
kbvc relation create deployed_on --inverse hosts --category infrastructure
kbvc relation list --category core
```

---

## Command Reference

<details>
<summary><strong>Setup & core workflow</strong></summary>

```bash
kbvc init                          # initialise repo
kbvc config set embed.backend openai
kbvc add projects/doc.md           # stage a file
kbvc add .                         # stage all .md files
kbvc status                        # show staged / unstaged / stale
kbvc commit -m "message"           # embed changed chunks, snapshot state
kbvc commit -m "message" --dry-run # preview without writing
kbvc log                           # full commit history
kbvc log --oneline
kbvc diff projects/doc.md          # chunk diff vs last commit
kbvc checkout <hash>               # restore full state to a commit
```
</details>

<details>
<summary><strong>Knowledge graph</strong></summary>

```bash
kbvc link a.md b.md --type informed_by --note "based on this paper"
kbvc unlink <rel-id>
kbvc graph projects/doc.md         # show neighbours
kbvc graph projects/doc.md --depth 2
kbvc analyze                       # auto-discover relation candidates
kbvc analyze --use-vectors --apply
kbvc depends add consumer.md base.md
kbvc impact base.md                # what breaks if base.md changes?
```
</details>

<details>
<summary><strong>Relation type management</strong></summary>

```bash
kbvc relation list                 # all built-in + custom relation types
kbvc relation list --category core
kbvc relation show informed_by     # full details including inverse/flags
kbvc relation create deployed_on \
    --inverse hosts \
    --category infrastructure
```
</details>

<details>
<summary><strong>Retrieval</strong></summary>

```bash
kbvc query "what is the tech stack?"
kbvc query "funding details" --top-k 10
kbvc query "..." --profile graphrag   # vector + graph traversal
```
</details>

<details>
<summary><strong>Ingest</strong></summary>

```bash
kbvc ingest website https://docs.anthropic.com
kbvc ingest github https://github.com/langchain-ai/langchain
kbvc ingest pdf ~/papers/paper.pdf
kbvc ingest notion <page-id> --token $NOTION_TOKEN
```
</details>

<details>
<summary><strong>Branches & prompt versioning</strong></summary>

```bash
kbvc branch create experiment
kbvc branch switch experiment
kbvc prompt set "Answer using only the provided context."
kbvc prompt log
kbvc prompt checkout p-v2
```
</details>

<details>
<summary><strong>Maintenance & migration</strong></summary>

```bash
kbvc gc                            # remove orphaned vectors
kbvc gc --dry-run --snapshots
kbvc sync                          # auto-commit all changed KOs
kbvc sync --volatility live
kbvc migrate backend --from qdrant --to pgvector
kbvc migrate embeddings --from text-embedding-3-small --to gemini-embedding-001
kbvc migrate schema
```
</details>

<details>
<summary><strong>Audit & health</strong></summary>

```bash
kbvc trace main__doc__chunk_2      # trace a vector to its commit
kbvc explain main__doc__chunk_2    # full provenance chain
kbvc history projects/doc.md       # per-KO version history
kbvc stale                         # show stale KOs
kbvc stale --fix                   # stage stale KOs for recommit
kbvc stats                         # evolution dashboard
kbvc doctor --knowledge            # full health check
kbvc contradict list               # show contradiction pairs
kbvc contradict resolve <rel-id>
```
</details>

<details>
<summary><strong>Push & remote</strong></summary>

```bash
kbvc remote add origin --backend qdrant --url https://... --collection prod
kbvc remote add staging --backend qdrant --url https://... --collection staging
kbvc push                          # incremental push to origin
kbvc push staging
kbvc push --dry-run
```
</details>

<details>
<summary><strong>Backend management</strong></summary>

```bash
kbvc backend init                  # idempotently create KBVC schema
kbvc backend info                  # show backend config (no credentials)
```
</details>

---

## The `kbvc.lock` File

Commit `kbvc.lock` to your git repository. It records the exact embedding model, vector store, and retrieval configuration — like `package-lock.json` for your knowledge base. Anyone cloning the repo knows exactly what backend to configure.

---

## Reproducibility

Given any commit hash, you can restore the exact knowledge state that produced a retrieval result — document versions, graph snapshot, prompt, retrieval config:

```bash
kbvc trace main__doc__chunk_3
# → committed 2026-03-15, model text-embedding-3-small, commit a3f7c91

kbvc checkout a3f7c91
# Exact same retrieval results as at that point in time
```

---

## Development

```bash
git clone https://github.com/Saiyam-Sandhir-Jain/kbvc
cd kbvc
pip install -e ".[dev]"
pytest tests/
```

```
195 tests · 0 failures · 0 errors
```

---

## Roadmap

| Phase | Status | Features |
|---|---|---|
| 1–6 | ✅ Complete | Core version control, graph, GraphRAG, ingest, push, VSAL, explain, promote |
| 7 | ✅ Complete | `kbvc gc`, `kbvc migrate embeddings`, `kbvc migrate schema` |
| 8 | ✅ Complete | `kbvc sync` |
| 9 | ✅ Complete | `kbvc contradict` |
| + | ✅ Complete | Relation Registry (`kbvc relation list/show/create`), Gemini SDK migration |
| Next | 🔜 Planned | Python SDK, `kbvc pull --rebuild`, branch merge, web UI, `kbvc ask` |

---

## License

[MIT](LICENSE) — see `LICENSE` for details.

---

*KBVC — because your AI is only as trustworthy as the knowledge it retrieves.*
