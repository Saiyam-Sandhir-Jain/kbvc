# kbvc/utils/config.py
"""
Config helpers for reading and writing the INI config at .kbvc/config.
"""

from __future__ import annotations

from configparser import ConfigParser
from pathlib import Path


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
    Creates the section if it doesn't exist.
    """
    parts = key.split(".", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Config key must be in section.name format (e.g. embed.backend), got: {key}"
        )
    section, name = parts

    parser = ConfigParser()
    parser.read(config_path)

    if not parser.has_section(section):
        parser.add_section(section)
    parser.set(section, name, value)

    with open(config_path, "w", encoding="utf-8") as f:
        parser.write(f)
