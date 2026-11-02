"""Vector embeddings with automatic fallback.

- Primary: Gemini `gemini-embedding-001` (semantic, multilingual, free tier).
- Fallback: TF-IDF (scikit-learn) - fully offline, no API key required.

The whole index always uses ONE backend so document and query vectors live in
the same vector space.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .llm import GeminiClient, LLMError

MAX_EMBED_CHARS = 6000  # gemini-embedding-001 accepts ~2k tokens per input


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class Embedder:
    def __init__(self, llm: GeminiClient | None = None):
        self.llm = llm
        self.backend = "gemini" if (llm and llm.available) else "tfidf"
        self._tfidf: TfidfVectorizer | None = None

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        texts = [t[:MAX_EMBED_CHARS] for t in texts]
        if self.backend == "gemini":
            try:
                vecs = self.llm.embed(texts, task_type="RETRIEVAL_DOCUMENT")  # type: ignore[union-attr]
                return _normalize(np.array(vecs, dtype=np.float32))
            except LLMError:
                self.backend = "tfidf"  # degrade gracefully
        self._tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        return _normalize(self._tfidf.fit_transform(texts).toarray().astype(np.float32))

    def transform_query(self, text: str) -> np.ndarray:
        text = text[:MAX_EMBED_CHARS]
        if self.backend == "gemini":
            vec = self.llm.embed([text], task_type="RETRIEVAL_QUERY")[0]  # type: ignore[union-attr]
            return _normalize(np.array([vec], dtype=np.float32))[0]
        assert self._tfidf is not None, "fit_transform must be called first"
        return _normalize(self._tfidf.transform([text]).toarray().astype(np.float32))[0]

    @property
    def label(self) -> str:
        return "Gemini embeddings (gemini-embedding-001)" if self.backend == "gemini" else "TF-IDF (offline fallback)"
