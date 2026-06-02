# kbvc/backends/embed/gemini.py
"""
Google Gemini embedding backend — using the new google-genai SDK.

Supports all Gemini embedding models:
  - gemini-embedding-001 (3072 dims)
  - text-embedding-004   (768 dims)

Note: Uses the new unified google-genai SDK (not deprecated google-generativeai).
The EmbedContentResponse has `.embeddings` (list[ContentEmbedding]), each with
`.values` (list[float]).  NOT `.embedding` — that field does not exist.

Bug fixed: response.embedding → response.embeddings[0].values
"""

from __future__ import annotations

from typing import List

from kbvc.backends.embed import EmbedBackend

_GEMINI_DIMS: dict[str, int] = {
    "gemini-embedding-001": 3072,
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
                "google-genai package not installed.\n"
                "Run: pip install kbvc[gemini]\n"
                "(Uses new google-genai SDK, not deprecated google-generativeai)"
            )
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._dims = _GEMINI_DIMS.get(model, 3072)

    # ── helpers ────────────────────────────────────────────────────────────────

    def _extract_vector(self, response) -> List[float]:
        """
        Extract the float vector from an EmbedContentResponse.

        The google-genai SDK returns:
            response.embeddings: list[ContentEmbedding]
            response.embeddings[0].values: list[float]

        NOT response.embedding (that attribute does not exist).
        """
        try:
            embeddings = response.embeddings
            if not embeddings:
                raise ValueError(
                    "Gemini returned an empty embeddings list. "
                    "Check your API key, model name, and quota."
                )
            values = embeddings[0].values
            if values is None:
                raise ValueError(
                    "Gemini ContentEmbedding.values is None. "
                    "The model may not support embedding for this input."
                )
            return list(values)
        except AttributeError as exc:
            raise RuntimeError(
                f"Unexpected Gemini response shape: {exc}.\n"
                f"Response was: {response!r}\n"
                "If the google-genai SDK was recently updated, please file a "
                "KBVC issue at https://github.com/Saiyam-Sandhir-Jain/kbvc/issues"
            ) from exc

    # ── EmbedBackend interface ─────────────────────────────────────────────────

    def embed(self, text: str) -> List[float]:
        response = self._client.models.embed_content(
            model=f"models/{self._model}",
            contents=text,
        )
        return self._extract_vector(response)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        results: List[List[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i: i + _BATCH_SIZE]
            for text in batch:
                response = self._client.models.embed_content(
                    model=f"models/{self._model}",
                    contents=text,
                )
                results.append(self._extract_vector(response))
        return results

    @property
    def dimensions(self) -> int:
        return self._dims

    @property
    def model_name(self) -> str:
        return self._model

    # ── factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config: dict) -> "GeminiEmbedBackend":
        api_key = config.get("embed.key", "")
        if not api_key:
            raise ValueError(
                "Gemini API key not configured.\n"
                "Run: kbvc config set embed.key AIza..."
            )
        model = config.get("embed.model", _DEFAULT_MODEL)
        return cls(api_key=api_key, model=model)
