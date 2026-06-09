# kbvc/commands/backend.py
"""
kbvc backend — Vector Storage Abstraction Layer management commands.

  kbvc backend init   — idempotently create KBVC schema on the configured backend
  kbvc backend info   — display current backend configuration
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kbvc.core.repo import KbvcRepo


def run_backend_init(repo: "KbvcRepo") -> None:
    """
    Idempotently initialise the KBVC schema on the configured vector backend.

    Reads backend type, URL, and embedding dimensions from config/lock file,
    then calls initialize_schema() on the appropriate adapter.
    """
    import click
    from kbvc.commands.migrate_helpers import load_backend, detect_backend

    cfg = repo.config()
    backend_name = detect_backend(cfg)
    collection = cfg.get("vectordb.collection", "kbvc")
    embed_model = cfg.get("embed.model", "")
    dims_raw = cfg.get("embed.dims", "")

    if not backend_name or backend_name == "unknown":
        raise click.ClickException(
            "No vector backend configured.\n"
            "Run: kbvc config set vectordb.backend <qdrant|pgvector|chroma|pinecone>"
        )

    # Infer dimensions from model name if not explicitly set
    dims = _resolve_dims(dims_raw, embed_model)

    click.echo("Detected:\n")
    click.echo(f"  Backend:    {backend_name}")
    click.echo(f"  Collection: {collection}")
    click.echo(f"  Model:      {embed_model or '(not set)'}")
    click.echo(f"  Dimensions: {dims}")
    click.echo("")
    click.echo("Creating schema…")

    try:
        backend = load_backend(backend_name, cfg)
    except ValueError as exc:
        raise click.ClickException(str(exc))

    try:
        backend.initialize_schema(collection, dims)
    except Exception as exc:
        raise click.ClickException(f"Schema initialisation failed: {exc}")

    click.echo(f"  ✓ {collection}")
    click.echo("")
    click.echo("Schema ready. You can now run:")
    click.echo("  kbvc add .  &&  kbvc commit -m '...'")


def run_backend_info(repo: "KbvcRepo") -> None:
    """Display current backend configuration (no credentials)."""
    import click
    from kbvc.commands.migrate_helpers import detect_backend

    cfg = repo.config()
    backend_name = detect_backend(cfg)
    collection = cfg.get("vectordb.collection", "kbvc")
    embed_provider = cfg.get("embed.backend", "(not set)")
    embed_model = cfg.get("embed.model", "(not set)")
    dims_raw = cfg.get("embed.dims", "")
    dims = _resolve_dims(dims_raw, embed_model)

    click.echo("══════════════════════════════════════")
    click.echo("  KBVC Backend Info")
    click.echo("══════════════════════════════════════")
    click.echo(f"  Vector backend:  {backend_name or '(not set)'}")
    click.echo(f"  Collection:      {collection}")
    click.echo(f"  Embed provider:  {embed_provider}")
    click.echo(f"  Embed model:     {embed_model}")
    click.echo(f"  Dimensions:      {dims}")
    click.echo("══════════════════════════════════════")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODEL_DIMS = {
    # OpenAI
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    # Gemini
    "gemini-embedding-001": 3072,
    "gemini-embedding-2": 3072,   # alias used in config; same dims as gemini-embedding-001
    "text-embedding-004": 768,
    # common HF / Ollama
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
}


def _resolve_dims(dims_raw: str, model: str) -> int:
    if dims_raw:
        try:
            return int(dims_raw)
        except ValueError:
            pass
    return _MODEL_DIMS.get(model, 1536)
