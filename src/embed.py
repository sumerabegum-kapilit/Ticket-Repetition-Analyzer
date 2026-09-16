"""Local, free, offline embeddings via sentence-transformers.

No API key, no per-ticket cost, nothing leaves the machine. This is the
workhorse of the pipeline since it runs over every ticket.
"""
from __future__ import annotations

import numpy as np

from .config import settings

_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(settings.embedding_model)
    return _model


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    model = get_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 200,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vectors.astype("float32")
