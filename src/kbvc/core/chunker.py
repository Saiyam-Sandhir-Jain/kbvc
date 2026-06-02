# kbvc/core/chunker.py
"""
Markdown chunker: splits a KO source file into sections,
prefixes each with the KO identity string, and hashes with SHA-256.
Compatible with Saiyam's existing frontmatter schema.
"""

import hashlib
import re
import yaml
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Chunk:
    index: int
    section: str
    text: str        # enriched text (identity prefix + section body)
    hash: str        # xxhash hex of the enriched text
    token_count: int # rough estimate: word_count * 4/3


def parse_frontmatter(content: str) -> Tuple[dict, str]:
    """
    Extract YAML frontmatter from a Markdown file.
    Returns (frontmatter_dict, body_str).
    If no frontmatter is found returns ({}, full_content).
    """
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            try:
                fm = yaml.safe_load(parts[1])
            except yaml.YAMLError:
                fm = {}
            return fm or {}, parts[2].strip()
    return {}, content


def split_into_chunks(
    body: str,
    frontmatter: dict,
    split_on: str = "##",
    target_tokens: int = 400,
) -> List[Chunk]:
    """
    Split a KO body into chunks at section boundaries defined by `split_on`.

    `split_on` options:
        ##        — split on level-2 headers (recommended)
        #         — split on level-1 headers
        paragraph — split on blank lines

    Each chunk is prefixed with the KO identity string so that any chunk
    retrieved in isolation still carries context about its origin.

    Args:
        body:          The markdown body text (frontmatter already stripped).
        frontmatter:   Parsed frontmatter dict (provides id, type, title).
        split_on:      Section delimiter.
        target_tokens: Target chunk size for reference (not hard-enforced in v1).

    Returns:
        List of Chunk dataclasses, ordered by occurrence in the document.
    """
    ko_id = frontmatter.get("id", "unknown")
    ko_type = frontmatter.get("type", "document")
    ko_title = frontmatter.get("title", "") or frontmatter.get("name", "")
    # Identity prefix prepended to every chunk so retrieval context is self-contained
    identity = f"[{ko_type.upper()}] {ko_title} (id: {ko_id})\n"

    if split_on == "paragraph":
        pattern = r"(?=\n\n)"
    else:
        # Lookahead at the start of a line matching the header marker
        escaped = re.escape(split_on)
        pattern = rf"(?=^{escaped} )"

    raw_sections = re.split(pattern, body, flags=re.MULTILINE)

    chunks: List[Chunk] = []
    current_section_name = "preamble"
    chunk_idx = 0

    for section_text in raw_sections:
        stripped = section_text.strip()
        if not stripped:
            continue

        # Determine section heading from first line
        first_line = stripped.split("\n")[0]
        if first_line.startswith("#"):
            current_section_name = first_line.lstrip("#").strip()

        enriched = identity + stripped
        hsh = hashlib.sha256(enriched.encode("utf-8")).hexdigest()[:16]
        # Token estimate: words * 4/3 (rough BPE approximation)
        token_count = len(enriched.split()) * 4 // 3

        chunks.append(Chunk(
            index=chunk_idx,
            section=current_section_name,
            text=enriched,
            hash=hsh,
            token_count=token_count,
        ))
        chunk_idx += 1

    return chunks


def compute_changed_chunks(
    new_chunks: List[Chunk],
    stored_hashes: List[str],
) -> List[int]:
    """
    Return indices of chunks that are new (beyond stored count) or hash-changed.
    Does NOT include deleted indices — call compute_deleted_chunks separately.
    """
    changed: List[int] = []
    for chunk in new_chunks:
        if chunk.index >= len(stored_hashes):
            changed.append(chunk.index)
        elif chunk.hash != stored_hashes[chunk.index]:
            changed.append(chunk.index)
    return changed


def compute_deleted_chunks(
    new_chunks: List[Chunk],
    stored_hashes: List[str],
) -> List[int]:
    """
    Return indices of chunks that existed in the old version but no longer exist.
    The caller must issue vector DB deletes for these to avoid orphan vectors.
    """
    new_count = len(new_chunks)
    old_count = len(stored_hashes)
    return list(range(new_count, old_count)) if old_count > new_count else []
