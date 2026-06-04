# kbvc/utils/config.py
"""
Config helpers for reading and writing the INI config at .kbvc/config.
"""

from __future__ import annotations

from configparser import ConfigParser
from pathlib import Path

# All valid section prefixes.
# Validated on write so typos (e.g. "vectordb.urlhttp") fail loudly
# instead of silently creating a garbage config key or a directory on disk.
_VALID_SECTIONS = {
    "embed",
    "vectordb",
    "chunk",
    "retrieval",
    "remote",
    "repo",
}


def read_config(config_path: Path) -> dict:
    """Parse INI config to a flat dot-key dict."""
    parser = ConfigParser()
    parser.read(config_path)
    flat: dict = {}
    for section in parser.sections():
        for key, val in parser.items(section):
            flat[f"{section}.{key}"] = val
    return flat


def write_config_key(config_path: Path, key: str, value: str) -> None:
    """
    Write a single key=value to the INI config.
    Key format: section.name  (e.g. "embed.backend")

    Raises ValueError for:
    - Keys not in section.name format
    - Unknown sections (catches typos like "vectordb.urlhttp://...")
    """
    parts = key.split(".", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Config key must be in section.name format (e.g. embed.backend), got: '{key}'\n"
            f"Valid sections: {', '.join(sorted(_VALID_SECTIONS))}"
        )
    section, name = parts

    # Reject unknown sections — catches common typos immediately
    if section not in _VALID_SECTIONS:
        raise ValueError(
            f"Unknown config section: '{section}'.\n"
            f"Valid sections: {', '.join(sorted(_VALID_SECTIONS))}\n"
            f"Example: kbvc config set vectordb.url http://localhost:8000"
        )

    # Sanity check: value shouldn't look like it was meant to be part of the key
    # e.g. someone typed: kbvc config set vectordb.urlhttp://localhost:8000
    if "://" in name:
        raise ValueError(
            f"Config name '{name}' looks like it contains a URL — did you forget a space?\n"
            f"  Wrong: kbvc config set vectordb.url{value}\n"
            f"  Right: kbvc config set vectordb.url {value}"
        )

    parser = ConfigParser()
    parser.read(config_path)

    if not parser.has_section(section):
        parser.add_section(section)
    parser.set(section, name, value)

    with open(config_path, "w", encoding="utf-8") as f:
        parser.write(f)
