# kbvc/backends/embed/gemini.py
"""
Google Gemini embedding backend — using the new google-genai SDK.

Supports all Gemini embedding models:
  - gemini-embedding-001 (3072 dims)
  - gemini-embedding-2 (3072 dims)
  - text-embedding-004 (768 dims)

Note: Migrated from deprecated google-generativeai (EOL 2025-11-30)
to the new unified google-genai SDK.
"""

from __future__ import annotations

from typing import List

from kbvc.backends.embed import EmbedBackend

_GEMINI_DIMS: dict[str, int] = {
    "gemini-embedding-001": 3072,
    "gemini-embedding-2":   3072,
    "text-embedding-004":   768,
}

_DEFAULT_MODEL = "gemini-embedding-001"
_BATCH_SIZE = 100  # Gemini has a lower per-request limit than OpenAI


class GeminiEmbedBackend(EmbedBackend):

    def __init__(self, api_key: str, model: str = _DEFAULT_MODEL) -> None:
        try:
            from google import genai
        except ImportError:
            raise ImportError(
                "google-genai package not installed. "
                "Run: pip install kbvc[gemini]\n"
                "(Note: uses new google-genai SDK, not deprecated google-generativeai)"
            )
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._dims = _GEMINI_DIMS.get(model, 3072)

    # ── EmbedBackend interface ────────────────────────────────────────────────

    def embed(self, text: str) -> List[float]:
        response = self._client.models.embed_content(
            model=f"models/{self._model}",
            contents=text,
        )
        return response.embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        results: List[List[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            for text in batch:
                response = self._client.models.embed_content(
                    model=f"models/{self._model}",
                    contents=text,
                )
                results.append(response.embedding)
        return results

    @property
    def dimensions(self) -> int:
        return self._dims

    @property
    def model_name(self) -> str:
        return self._model

    # ── factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config: dict) -> GeminiEmbedBackend:
        api_key = config.get("embed.key", "")
        if not api_key:
            raise ValueError(
                "Gemini API key not configured.\n"
                "Run: kbvc config set embed.key AIza..."
            )
        model = config.get("embed.model", _DEFAULT_MODEL)
        return cls(api_key=api_key, model=model)
