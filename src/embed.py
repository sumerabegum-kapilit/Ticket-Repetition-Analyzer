"""Local, free, offline embeddings via sentence-transformers.

No API key, no per-ticket cost, nothing leaves the machine. This is the
workhorse of the pipeline since it runs over every ticket.
"""
from __future__ import annotations

import importlib.machinery
import sys
import types

import numpy as np

from .config import settings

_model = None


def _stub_sklearn_if_blocked() -> None:
    """sentence-transformers unconditionally imports several sklearn.metrics
    symbols at package-import time, for evaluator classes this project never
    uses (only .encode() is called - see embed_texts below). scikit-learn's
    own import chain pulls in freshly-compiled Cython/scipy internals (e.g.
    sklearn.utils._cyutility) that a Windows Application Control policy can
    block as unrecognized native code - this has been observed to start
    happening on a machine where it worked before, with no project change
    involved. Since nothing here ever calls an sklearn-backed evaluator, try
    a real import first and only fall back to a harmless stub (any attribute
    access returns a placeholder that errors if actually called) if sklearn
    itself is unavailable - this sidesteps the block entirely rather than
    depending on it not recurring."""
    if "sklearn" in sys.modules:
        return
    try:
        import sklearn  # noqa: F401
        return
    except Exception:
        pass

    class _NoOpModule(types.ModuleType):
        def __getattr__(self, name):
            def _unavailable(*_args, **_kwargs):
                raise RuntimeError(
                    f"{self.__name__}.{name} is unavailable - scikit-learn failed to "
                    "import (see src/embed.py) and was stubbed out, since this project "
                    "only needs sentence-transformers' .encode(), never an "
                    "sklearn-backed evaluator."
                )

            return _unavailable

    for name in ("sklearn", "sklearn.metrics", "sklearn.metrics.pairwise"):
        module = _NoOpModule(name)
        # A module sitting in sys.modules with __spec__ left as None (the
        # default) isn't a state the import system otherwise produces on
        # its own - importlib.util.find_spec("sklearn") explicitly rejects
        # it (ValueError: sklearn.__spec__ is None) rather than treating it
        # as "not installed". transformers' own availability check does
        # exactly that find_spec() call, so this stub needs a real (if
        # inert) spec to look like a properly-imported module to callers
        # that only check for sklearn's presence, not just callers that
        # `from sklearn.x import y` it directly.
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        sys.modules.setdefault(name, module)


def get_model():
    global _model
    if _model is None:
        _stub_sklearn_if_blocked()
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
