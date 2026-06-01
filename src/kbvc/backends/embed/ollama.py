# kbvc/backends/embed/ollama.py
"""
Ollama embedding backend — for local models (no API key required).
"""

from __future__ import annotations

from typing import List

from kbvc.backends.embed import EmbedBackend

_DEFAULT_MODEL = "nomic-embed-text"
_DEFAULT_URL = "http://localhost:11434"
_DEFAULT_DIMS = 768  # nomic-embed-text default


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
        # Probe actual dimensions on first embed
        self._dims: int = 0

    def embed(self, text: str) -> List[float]:
        response = self._ollama.embeddings(model=self._model, prompt=text)
        vec = response["embedding"]
        if not self._dims:
            self._dims = len(vec)
        return vec

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]

    @property
    def dimensions(self) -> int:
        if not self._dims:
            # Probe with a dummy string to get dims
            self.embed("probe")
        return self._dims

    @property
    def model_name(self) -> str:
        return self._model

    @classmethod
    def from_config(cls, config: dict) -> OllamaEmbedBackend:
        model = config.get("embed.model", _DEFAULT_MODEL)
        url = config.get("embed.url", _DEFAULT_URL)
        return cls(model=model, base_url=url)
