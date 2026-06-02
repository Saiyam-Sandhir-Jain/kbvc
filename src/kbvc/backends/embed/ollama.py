# kbvc/backends/embed/ollama.py
"""
Ollama embedding backend — for local models (no API key required).

API migration note
------------------
The Ollama Python SDK has two embedding functions:

  DEPRECATED (ollama < 0.4):
      ollama.embeddings(model=..., prompt=...)
      → EmbeddingsResponse with `.embedding` (list[float])

  CURRENT (ollama >= 0.4):
      ollama.embed(model=..., input=...)
      → EmbedResponse with `.embeddings` (list[list[float]])

This backend tries the current API first and falls back to the deprecated
one so both SDK versions are supported.

Bug fixed: was calling deprecated ollama.embeddings() and doing dict
access response["embedding"] which fails on newer SDK objects.
"""

from __future__ import annotations

from typing import List

from kbvc.backends.embed import EmbedBackend

_DEFAULT_MODEL = "nomic-embed-text"
_DEFAULT_URL = "http://localhost:11434"


class OllamaEmbedBackend(EmbedBackend):

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        base_url: str = _DEFAULT_URL,
    ) -> None:
        try:
            import ollama as _ollama
        except ImportError:
            raise ImportError(
                "ollama package not installed. Run: pip install kbvc[ollama]"
            )
        self._ollama = _ollama
        self._model = model
        self._base_url = base_url
        self._dims: int = 0

        # Detect which API the installed SDK exposes
        self._use_new_api = callable(getattr(_ollama, "embed", None))

    # ── helpers ────────────────────────────────────────────────────────────────

    def _call_embed(self, text: str) -> List[float]:
        """Call whichever Ollama embed API is available and return a flat vector."""
        if self._use_new_api:
            # New API: ollama.embed(model, input) → EmbedResponse
            # .embeddings is list[list[float]] — one list per input item
            response = self._ollama.embed(model=self._model, input=text)
            try:
                vec = list(response.embeddings[0])
            except (AttributeError, IndexError, TypeError):
                # Some builds return a Mapping — try dict-style fallback
                vec = list(response["embeddings"][0])
        else:
            # Deprecated API: ollama.embeddings(model, prompt) → EmbeddingsResponse
            # .embedding is list[float] (singular)
            response = self._ollama.embeddings(model=self._model, prompt=text)
            try:
                vec = list(response.embedding)
            except AttributeError:
                vec = list(response["embedding"])

        if not vec:
            raise RuntimeError(
                f"Ollama returned an empty embedding for model '{self._model}'.\n"
                "Make sure the model supports embedding and is pulled:\n"
                f"  ollama pull {self._model}"
            )
        return vec

    # ── EmbedBackend interface ─────────────────────────────────────────────────

    def embed(self, text: str) -> List[float]:
        vec = self._call_embed(text)
        if not self._dims:
            self._dims = len(vec)
        return vec

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if self._use_new_api:
            # New API supports batch input directly
            response = self._ollama.embed(model=self._model, input=texts)
            try:
                results = [list(e) for e in response.embeddings]
            except (AttributeError, TypeError):
                results = [list(e) for e in response["embeddings"]]
            if results and not self._dims:
                self._dims = len(results[0])
            return results
        else:
            # Deprecated API is single-item only
            return [self.embed(t) for t in texts]

    @property
    def dimensions(self) -> int:
        if not self._dims:
            self.embed("probe")  # warm up to detect dims
        return self._dims

    @property
    def model_name(self) -> str:
        return self._model

    # ── factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config: dict) -> "OllamaEmbedBackend":
        model = config.get("embed.model", _DEFAULT_MODEL)
        url = config.get("embed.url", _DEFAULT_URL)
        return cls(model=model, base_url=url)
