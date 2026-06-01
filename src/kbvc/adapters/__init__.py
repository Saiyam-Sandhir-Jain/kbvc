# kbvc/adapters/__init__.py
"""
Adapter registry — maps source_type strings to SourceAdapter classes.

v1: only "file" → TextFileAdapter is registered.
v2+: adapters register via entry_points("kbvc.adapters") in pyproject.toml.
"""

from __future__ import annotations

from kbvc.adapters.base import SourceAdapter
from kbvc.adapters.text_file import TextFileAdapter

ADAPTER_REGISTRY: dict[str, type[SourceAdapter]] = {
    "file": TextFileAdapter,
}


def get_adapter(source_type: str) -> SourceAdapter | None:
    """Return an adapter instance for the given source_type, or None if unsupported."""
    # Built-in registry
    cls = ADAPTER_REGISTRY.get(source_type)
    if cls:
        return cls()

    # v2+: discover community adapters via entry_points
    try:
        from importlib.metadata import entry_points
        eps = entry_points(group="kbvc.adapters")
        for ep in eps:
            if ep.name == source_type:
                adapter_cls = ep.load()
                return adapter_cls()
    except Exception:
        pass

    return None
