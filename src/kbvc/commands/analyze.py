# kbvc/commands/analyze.py
"""
kbvc analyze — automatic relation discovery via semantic similarity.
kbvc extract — automatic entity extraction pipeline.

These are the "knowledge intelligence" features that elevate KBVC from
a versioning tool to a Knowledge Operating System.

Both commands are read-only — they SUGGEST but never mutate.
The user confirms with 'kbvc link' or 'kbvc extract --apply'.

Design:
  analyze:
    1. Load all KOs from ko_store
    2. For each pair, compare their stored chunk hashes for diversity
       (in a real deployment: query the vector DB for pairwise cosine sim)
    3. Emit scored suggestions ordered by confidence
    4. Never suggest relations that already exist in the graph
    5. Never emit more than max_suggestions to avoid graph explosion

  extract:
    1. Read the source file
    2. Apply heuristic regex + optional LLM extraction
    3. Return Entity candidates with confidence scores
    4. --apply flag writes entities to ko_store
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from kbvc.core.ko import KnowledgeObject


# ---------------------------------------------------------------------------
# RelationSuggestion
# ---------------------------------------------------------------------------

@dataclass
class RelationSuggestion:
    from_id: str
    to_id: str
    suggested_type: str
    confidence: float
    reasoning: str


# ---------------------------------------------------------------------------
# EntityCandidate
# ---------------------------------------------------------------------------

@dataclass
class EntityCandidate:
    name: str
    type: str          # "model" | "concept" | "person" | "institution" | "tool"
    chunk_index: int
    confidence: float
    context: str       # snippet of surrounding text


# ---------------------------------------------------------------------------
# Heuristic relation type inference
# ---------------------------------------------------------------------------

# Maps a keyword pattern → (relation_type, confidence_boost)
_RELATION_SIGNALS: list[tuple[re.Pattern, str, float]] = [
    (re.compile(r"\bbuilt (on|with|using)\b", re.I),       "extends",     0.10),
    (re.compile(r"\binspired by\b", re.I),                  "informed_by", 0.12),
    (re.compile(r"\bfork(ed)? (of|from)\b", re.I),         "extends",     0.15),
    (re.compile(r"\bpart of\b", re.I),                      "part_of",     0.12),
    (re.compile(r"\bcreated (at|during)\b", re.I),         "created_at",  0.10),
    (re.compile(r"\b(references?|cites?)\b", re.I),        "cites",       0.08),
    (re.compile(r"\bcontradicts?\b", re.I),                 "contradicts", 0.12),
    (re.compile(r"\bused in\b", re.I),                      "used_in",     0.10),
    (re.compile(r"\bstudied at\b", re.I),                   "studied_at",  0.12),
    (re.compile(r"\bworked at\b", re.I),                    "worked_at",   0.12),
    (re.compile(r"\bdeveloped during\b", re.I),             "developed_during", 0.12),
]

# ---------------------------------------------------------------------------
# Entity extraction patterns
# ---------------------------------------------------------------------------

_ENTITY_PATTERNS: list[tuple[re.Pattern, str]] = [
    # AI models / techniques
    (re.compile(r"\b(GPT-[\d\.]+|Claude[\s\-]\d|Gemini[\s\-]\w+|LLaMA[\s\-]\d+|"
                r"BERT|T5|Mistral|Mixtral|Phi-\d|Falcon|Qwen\d+)\b", re.I), "model"),
    # Python libraries / tools
    (re.compile(r"\b(LangChain|LlamaIndex|ChromaDB|Qdrant|Pinecone|Weaviate|"
                r"PyTorch|TensorFlow|JAX|scikit-learn|transformers|sentence-transformers|"
                r"FastAPI|Flask|Django)\b", re.I), "tool"),
    # Institutions
    (re.compile(r"\b(OpenAI|Anthropic|Google|Meta|Microsoft|DeepMind|Hugging Face|"
                r"Stanford|MIT|CMU|Oxford|Cambridge|Berkeley)\b"), "institution"),
    # Concepts (common AI/ML concepts)
    (re.compile(r"\b(RAG|GraphRAG|fine-tuning|RLHF|embeddings?|vector (store|database)|"
                r"knowledge graph|prompt engineering|retrieval.augmented|attention mechanism|"
                r"transformer architecture)\b", re.I), "concept"),
]


# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------

def suggest_relations(
    kos: List["KnowledgeObject"],
    existing_relations: set[tuple[str, str]],  # set of (from_id, to_id)
    source_texts: dict[str, str],  # ko_id → raw markdown body
    max_suggestions: int = 10,
    min_confidence: float = 0.40,
    embed_backend=None,   # optional: EmbedBackend for vector similarity
    vdb_backend=None,     # optional: VectorDBBackend for real cosine sim
    collection: str = "kbvc",
) -> List[RelationSuggestion]:
    """
    Suggest relations between KOs.

    Strategy (ordered by preference):
      1. If embed_backend + vdb_backend available: real cosine similarity
         via vector DB nearest-neighbour lookup.
      2. Fallback: heuristic text signal matching between documents.

    Only emits suggestions that don't already exist in the graph.
    Caps output at max_suggestions to prevent graph explosion.
    """
    suggestions: List[RelationSuggestion] = []
    ko_ids = [ko.id for ko in kos]

    if embed_backend and vdb_backend:
        suggestions = _suggest_via_vectors(
            kos, existing_relations, source_texts,
            embed_backend, vdb_backend, collection,
            max_suggestions, min_confidence,
        )
    else:
        suggestions = _suggest_via_heuristics(
            kos, existing_relations, source_texts,
            max_suggestions, min_confidence,
        )

    return suggestions[:max_suggestions]


def _suggest_via_vectors(
    kos, existing_relations, source_texts,
    embed, vdb, collection, max_sug, min_conf,
) -> List[RelationSuggestion]:
    """
    Use actual vector similarity to find related KOs.
    For each KO, embed its first chunk and query the DB for nearest neighbours.
    """
    from kbvc.core.chunker import parse_frontmatter, split_into_chunks

    suggestions: List[RelationSuggestion] = []
    seen_pairs: set[frozenset] = set()

    for ko in kos:
        body = source_texts.get(ko.id, "")
        if not body:
            continue
        fm = {"id": ko.id, "type": ko.type}
        chunks = split_into_chunks(body, fm)
        if not chunks:
            continue

        try:
            vec = embed.embed(chunks[0].text)
            results = vdb.query(collection, vec, top_k=5)
        except Exception:
            continue

        for r in results:
            meta = r.get("metadata", {})
            other_id = meta.get("ko_id", "")
            score = r.get("score", 0.0)

            if not other_id or other_id == ko.id:
                continue
            pair = frozenset({ko.id, other_id})
            if pair in seen_pairs:
                continue
            if (ko.id, other_id) in existing_relations or (other_id, ko.id) in existing_relations:
                continue

            seen_pairs.add(pair)
            # Infer relation type from heuristics on text
            rel_type, text_boost = _infer_relation_type(
                source_texts.get(ko.id, ""),
                source_texts.get(other_id, ""),
            )
            confidence = min(score + text_boost, 0.99)
            if confidence >= min_conf:
                suggestions.append(RelationSuggestion(
                    from_id=ko.id,
                    to_id=other_id,
                    suggested_type=rel_type,
                    confidence=round(confidence, 3),
                    reasoning=f"cosine={score:.3f} + text signals",
                ))

    return sorted(suggestions, key=lambda s: -s.confidence)


def _suggest_via_heuristics(
    kos, existing_relations, source_texts, max_sug, min_conf,
) -> List[RelationSuggestion]:
    """
    Pure text heuristic fallback — no vector DB required.
    Cross-references KO names and relation signal keywords.
    """
    suggestions: List[RelationSuggestion] = []
    seen_pairs: set[frozenset] = set()

    for i, ko_a in enumerate(kos):
        text_a = source_texts.get(ko_a.id, "").lower()
        for ko_b in kos[i + 1:]:
            pair = frozenset({ko_a.id, ko_b.id})
            if pair in seen_pairs:
                continue
            if (ko_a.id, ko_b.id) in existing_relations or \
               (ko_b.id, ko_a.id) in existing_relations:
                continue

            text_b = source_texts.get(ko_b.id, "").lower()

            # Signal 1: does ko_a's text mention ko_b's id?
            cross_ref_score = 0.0
            if ko_b.id.replace("-", " ") in text_a:
                cross_ref_score += 0.30
            if ko_a.id.replace("-", " ") in text_b:
                cross_ref_score += 0.30

            # Signal 2: shared tags
            shared_tags = set(ko_a.tags) & set(ko_b.tags)
            tag_score = min(len(shared_tags) * 0.08, 0.25)

            # Signal 3: relation keywords in either text
            rel_type, signal_boost = _infer_relation_type(
                source_texts.get(ko_a.id, ""),
                source_texts.get(ko_b.id, ""),
            )

            confidence = cross_ref_score + tag_score + signal_boost
            if confidence >= min_conf:
                seen_pairs.add(pair)
                suggestions.append(RelationSuggestion(
                    from_id=ko_a.id,
                    to_id=ko_b.id,
                    suggested_type=rel_type,
                    confidence=round(min(confidence, 0.99), 3),
                    reasoning=(
                        f"cross_ref={cross_ref_score:.2f} "
                        f"shared_tags={len(shared_tags)} "
                        f"text_signals={signal_boost:.2f}"
                    ),
                ))

    return sorted(suggestions, key=lambda s: -s.confidence)


def _infer_relation_type(text_a: str, text_b: str) -> tuple[str, float]:
    """
    Scan text for relation signal keywords; return best-match type and score.
    Returns ("informed_by", 0.0) as the safe default.
    """
    combined = (text_a + " " + text_b).lower()
    best_type = "informed_by"
    best_boost = 0.0
    for pattern, rel_type, boost in _RELATION_SIGNALS:
        if pattern.search(combined):
            if boost > best_boost:
                best_boost = boost
                best_type = rel_type
    return best_type, best_boost


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

def extract_entities(
    ko_id: str,
    body: str,
    chunks: list,
    min_confidence: float = 0.65,
) -> List[EntityCandidate]:
    """
    Extract entity candidates from a KO's body text using pattern matching.

    In v2 this can be upgraded to an LLM call.
    For now: fast, offline, no API required.

    Args:
        ko_id:          The KO being analysed.
        body:           Raw body text (no frontmatter).
        chunks:         Chunk list from split_into_chunks (for chunk_index mapping).
        min_confidence: Minimum confidence threshold.

    Returns:
        Deduplicated list of EntityCandidate.
    """
    candidates: List[EntityCandidate] = []
    seen_names: set[str] = set()

    for pattern, entity_type in _ENTITY_PATTERNS:
        for match in pattern.finditer(body):
            name = match.group(0).strip()
            name_lower = name.lower()
            if name_lower in seen_names:
                continue
            seen_names.add(name_lower)

            # Find which chunk this entity appears in
            char_pos = match.start()
            chunk_idx = _char_pos_to_chunk(char_pos, body, chunks)

            # Context window (50 chars each side)
            ctx_start = max(0, char_pos - 50)
            ctx_end = min(len(body), match.end() + 50)
            context = body[ctx_start:ctx_end].replace("\n", " ").strip()

            # Confidence based on pattern certainty + name length
            base_conf = {"model": 0.85, "tool": 0.80, "institution": 0.88,
                         "concept": 0.72}.get(entity_type, 0.70)
            # Longer / more specific names are more confident
            len_boost = min(len(name) / 60, 0.10)
            confidence = round(min(base_conf + len_boost, 0.99), 3)

            if confidence >= min_confidence:
                candidates.append(EntityCandidate(
                    name=name,
                    type=entity_type,
                    chunk_index=chunk_idx,
                    confidence=confidence,
                    context=f"...{context}...",
                ))

    return sorted(candidates, key=lambda c: -c.confidence)


def _char_pos_to_chunk(char_pos: int, body: str, chunks: list) -> int:
    """
    Find which chunk index a character position maps to.
    Approximates by scanning chunk text in order.
    """
    if not chunks:
        return 0
    running = 0
    for chunk in chunks:
        # chunk.text includes the identity prefix — approximate body start
        chunk_body = chunk.text.split("\n", 2)[-1] if "\n" in chunk.text else chunk.text
        running += len(chunk_body)
        if running >= char_pos:
            return chunk.index
    return chunks[-1].index if chunks else 0
