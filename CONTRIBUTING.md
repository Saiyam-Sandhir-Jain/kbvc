# Contributing to KBVC

Thank you for your interest in contributing. KBVC is a version control system for AI knowledge bases, and every improvement — bug fix, new backend, new command, or doc clarification — makes the whole project more useful.

---

## Getting started

```bash
git clone https://github.com/Saiyam-Sandhir-Jain/kbvc
cd kbvc
pip install -e ".[dev]"
pytest tests/          # 195 tests · 0 failures expected
```

---

## How to contribute

### Reporting bugs

Open a [bug report](https://github.com/Saiyam-Sandhir-Jain/kbvc/issues/new?template=bug_report.yml). Include the KBVC version (`kbvc --version`), Python version, backends in use, and the full traceback.

### Requesting features

Open a [feature request](https://github.com/Saiyam-Sandhir-Jain/kbvc/issues/new?template=feature_request.yml). Check the [roadmap](KBVC_OVERVIEW.md) first — the feature may already be planned.

### Submitting a pull request

1. Fork the repo and create a branch from `main`:
   ```bash
   git checkout -b feat/your-feature
   ```
2. Make your changes. Keep commits focused and atomic.
3. Add or update tests. Every changed behaviour needs a test.
4. Run the full suite:
   ```bash
   pytest tests/ -v
   ```
5. Update `CHANGELOG.md` under `[Unreleased]`.
6. Open a PR against `main`. Fill in the PR template.

---

## Project structure

```
kbvc/
├── src/
│   └── kbvc/
│       ├── cli.py              # Click entry-point; routes to commands/
│       ├── commands/           # One module per command group
│       ├── core/               # Repo, KO, chunker, commit, graph, lineage
│       ├── backends/
│       │   ├── embed/          # OpenAI · Gemini · Ollama · HuggingFace
│       │   └── vectordb/       # Qdrant · pgvector · Pinecone · ChromaDB
│       ├── adapters/           # Source adapters (text file, future: PDF, etc.)
│       └── utils/              # Config, display, lock file helpers
├── tests/
│   └── test_kbvc.py        # Full integration test suite (195 tests)
├── pyproject.toml
├── KBVC_DOCS.md            # Full command reference
└── CHANGELOG.md
```

---

## Adding a new vector DB backend

1. Create `src/kbvc/backends/vectordb/mybackend.py`.
2. Implement the `VectorBackend` interface from `src/kbvc/backends/vectordb/__init__.py`.
3. Register the backend string in `kbvc/utils/config.py`.
4. Add an optional dependency in `pyproject.toml` under `[project.optional-dependencies]`.
5. Write tests covering `upsert`, `query`, `delete`, and `list_all`.

## Adding a new embedding backend

Same pattern under `src/kbvc/backends/embed/`. Implement `EmbedBackend` and register in config.

---

## Code style

KBVC uses [ruff](https://github.com/astral-sh/ruff) for linting and formatting.

```bash
pip install ruff
ruff check src/kbvc/
ruff format src/kbvc/
```

CI will fail on lint errors.

---

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(gc): add --snapshots flag to prune unreachable snapshot files
fix(commit): handle empty staging area without traceback
docs: add GraphRAG retrieval example to README
test(migrate): cover zero-KO edge case in embeddings migration
```

---

## Secrets policy

**Never commit API keys, tokens, or credentials.** KBVC's `.gitignore` excludes `.kbvc/secrets` and `.kbvc/config.local`. If you accidentally commit a secret, rotate it immediately and open an issue.

---

## Questions?

Open a [Discussion](https://github.com/Saiyam-Sandhir-Jain/kbvc/discussions) — we're happy to help.
