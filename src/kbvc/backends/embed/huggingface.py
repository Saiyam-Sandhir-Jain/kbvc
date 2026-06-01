# kbvc/backends/embed/huggingface.py
"""
HuggingFace sentence-transformers embedding backend.
Runs entirely locally — no API key required.
"""

from __future__ import annotations

from typing import List

from kbvc.backends.embed import EmbedBackend

_DEFAULT_MODEL = "all-MiniLM-L6-v2"  # 384 dims, small and fast


class HuggingFaceEmbedBackend(EmbedBackend):

    def __init__(self, model: str = _DEFAULT_MODEL) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Run: pip install kbvc[hf]"
            )
        self._model_id = model
        self._model = SentenceTransformer(model)
        self._dims: int = self._model.get_sentence_embedding_dimension()  # type: ignore

    def embed(self, text: str) -> List[float]:
        return self._model.encode(text, convert_to_numpy=True).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return [e.tolist() for e in embeddings]

    @property
    def dimensions(self) -> int:
        return self._dims

    @property
    def model_name(self) -> str:
        return self._model_id

    @classmethod
    def from_config(cls, config: dict) -> HuggingFaceEmbedBackend:
        model = config.get("embed.model", _DEFAULT_MODEL)
        return cls(model=model)
