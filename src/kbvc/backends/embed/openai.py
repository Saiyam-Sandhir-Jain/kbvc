# kbvc/backends/embed/openai.py
"""
OpenAI embedding backend.

Supports all OpenAI text-embedding-* models.
Default model: text-embedding-3-small (1536 dims, good cost/quality ratio).

P5 pitfall: batch size must be ≤ 512 items to stay well under the API limit
and avoid silent rate-limiting. Large commits are batched internally.
"""

from __future__ import annotations

from typing import List, Optional

from kbvc.backends.embed import EmbedBackend


# Model dimension registry — add new models here as they ship
_OPENAI_DIMS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

_DEFAULT_MODEL = "text-embedding-3-small"
_BATCH_SIZE = 512  # safe limit below OpenAI's 2048 maximum (P5)


class OpenAIEmbedBackend(EmbedBackend):

    def __init__(self, api_key: str, model: str = _DEFAULT_MODEL) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package not installed. Run: pip install kbvc[openai]"
            )
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._dims = _OPENAI_DIMS.get(model, 1536)  # default fallback

    # ── EmbedBackend interface ────────────────────────────────────────────────

    def embed(self, text: str) -> List[float]:
        response = self._client.embeddings.create(
            input=[text],
            model=self._model,
        )
        return response.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch-embed with automatic chunking to respect API limits (P5)."""
        if not texts:
            return []
        results: List[List[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            response = self._client.embeddings.create(
                input=batch,
                model=self._model,
            )
            # API returns results in order — safe to extend directly
            results.extend(item.embedding for item in response.data)
        return results

    @property
    def dimensions(self) -> int:
        return self._dims

    @property
    def model_name(self) -> str:
        return self._model

    # ── factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config: dict) -> OpenAIEmbedBackend:
        api_key = config.get("embed.key", "")
        if not api_key:
            raise ValueError(
                "OpenAI API key not configured.\n"
                "Run: kbvc config set embed.key sk-..."
            )
        model = config.get("embed.model", _DEFAULT_MODEL)
        return cls(api_key=api_key, model=model)
