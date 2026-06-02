# Contributing to KBVC

Thank you for your interest in contributing to KBVC! This guide covers everything you need to get started.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Submitting a PR](#submitting-a-pr)
- [Design Invariants](#design-invariants)
- [Adding Backends](#adding-backends)
- [Release Process](#release-process)

---

## Code of Conduct

Be respectful, constructive, and inclusive. This is an open-source project maintained in good faith.

---

## Development Setup

```bash
# 1. Fork and clone
git clone https://github.com/<you>/kbvc.git
cd kbvc

# 2. Install in dev mode (editable)
pip install -e ".[dev,openai,qdrant]"

# 3. Verify tests pass
pytest tests/ -v
# Expected: 195 passed

# 4. Run a local smoke test
mkdir /tmp/test-kb && cd /tmp/test-kb
kbvc init --no-git
kbvc config set embed.backend openai
kbvc config set embed.key sk-...
kbvc config set vectordb.backend lancedb
kbvc config set vectordb.url ./kbvc_lance
```

---

## Project Structure

```
src/kbvc/
├── cli.py          ← All Click commands wired here
├── core/           ← Data models (no I/O except files)
├── backends/       ← Pluggable embed + vectordb adapters
├── adapters/       ← Source type processors (text, web, pdf...)
├── commands/       ← Business logic called by cli.py
└── utils/          ← Config, display, lock helpers

tests/
└── test_kbvc.py    ← 2400+ line test suite (all phases)
```

---

## Making Changes

### New CLI Commands

1. Add business logic to a new file in `commands/`
2. Wire the Click command in `cli.py`
3. Follow the docstring convention: include `\b` + examples block
4. All commands that need backends import them via `get_embed_backend()` / `get_vectordb_backend()`

```python
@main.command()
@click.argument("ko_id")
@click.option("--verbose", is_flag=True)
def mycommand(ko_id, verbose):
    """One-line description.

    \b
    Longer description here.

    \b
    Examples:
        kbvc mycommand foo
        kbvc mycommand foo --verbose
    """
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))
    ...
```

### Commit Message Convention

```
<type>(<scope>): <short description>

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `ci`

Examples:
```
feat(backends): add LanceDB vector backend
fix(sync): use StagingIndex.load() instead of raw constructor
docs(readme): add backend comparison table
test(phase9): add contradiction resolution edge cases
```

---

## Testing

All tests are in `tests/test_kbvc.py`. The entire suite runs offline — all external backends are mocked.

```bash
# Run all tests
pytest tests/ -v

# Run a specific phase
pytest tests/ -v -k "Phase3"

# Run with coverage
pytest tests/ --cov=kbvc --cov-report=term-missing
```

### Adding Tests for New Features

Follow the existing `TempRepoTest` base class pattern:

```python
class TestMyFeature(TempRepoTest):
    def setUp(self):
        super().setUp()
        # self.repo is already init'd with --no-git
        # self.runner is a Click test runner
        # self.invoke(cmd, args) is a convenience helper

    @patch("kbvc.commands.my_command.get_embed_backend")
    @patch("kbvc.commands.my_command.get_vectordb_backend")
    def test_basic(self, mock_vdb, mock_embed):
        mock_embed.return_value = fake_embed_backend()
        mock_vdb.return_value = fake_vdb_backend()
        result = self.invoke(mycommand, ["arg1"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("expected output", result.output)
```

Use `fake_embed_backend()` and `fake_vdb_backend()` for mocks — these are defined near the top of `test_kbvc.py`.

---

## Submitting a PR

1. Fork the repo and create a branch: `git checkout -b feat/my-feature`
2. Make your changes
3. Add tests (all new commands must have tests)
4. Update `CHANGELOG.md` under `[Unreleased]`
5. Ensure `pytest tests/ -v` passes
6. Open a PR against `main`

Use the PR template — fill in all sections.

---

## Design Invariants

**Never violate these** — they are core to KBVC's audit integrity:

1. **Commit hash excludes timestamp.** Content-addressable, like Git.
2. **`ko_id` always derived by `_path_to_ko_id()`.** Never re-implement inline.
3. **Vector IDs always `<branch>__<ko_id>__chunk_<N>`.** Double underscore is the delimiter.
4. **Version snapshots are immutable.** Once written, never overwritten.
5. **`kbvc.lock` never contains API keys.**
6. **Frozen KOs never re-embedded.**

See `KBVC_DOCS.md § Design Invariants` for the full list.

---

## Adding Backends

### New Embedding Backend

```python
# src/kbvc/backends/embed/mybackend.py
from kbvc.backends.embed import EmbedBackend

class MyEmbedBackend(EmbedBackend):
    @property
    def dimensions(self) -> int: return 768

    @property
    def model_name(self) -> str: return "my-model"

    def embed(self, text: str) -> list: ...
    def embed_batch(self, texts: list) -> list: ...

    @classmethod
    def from_config(cls, config: dict) -> "MyEmbedBackend":
        return cls(...)
```

Register in `backends/__init__.py` and add to `pyproject.toml` optional-dependencies.

### New Vector DB Backend

Implement all methods in `VectorDBBackend` ABC including `initialize_schema` and `export_chunks`. See `backends/vectordb/lancedb.py` as the reference implementation.

---

## Release Process

Releases are managed by the maintainer (@Saiyam-Sandhir-Jain). The workflow:

1. Merge PRs into `main`
2. Update `CHANGELOG.md` — move `[Unreleased]` items to a new version section
3. Bump version in `pyproject.toml` and `src/kbvc/__init__.py`
4. Create a GitHub Release → triggers the `publish.yml` workflow → auto-publishes to PyPI

Contributors do **not** bump the version number in their PRs.
