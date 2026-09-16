"""A minimal on-disk vector store: L2-normalized embeddings + ids, persisted
as a .npy + .json pair. No server, no database engine, no extra dependency -
tens of thousands of vectors fit comfortably in memory and in a single file,
which is all this pipeline's scale needs. Swap for a real vector DB (Qdrant,
Milvus...) only if the ticket volume grows into the millions.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np


class VectorStore:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.vectors_path = self.directory / "vectors.npy"
        self.ids_path = self.directory / "ids.json"
        self.ids: list[str] = []
        self.vectors: Optional[np.ndarray] = None
        self._load()

    def _load(self) -> None:
        if self.vectors_path.exists() and self.ids_path.exists():
            self.vectors = np.load(self.vectors_path)
            self.ids = json.loads(self.ids_path.read_text(encoding="utf-8"))

    def save(self) -> None:
        if self.vectors is not None:
            np.save(self.vectors_path, self.vectors)
        self.ids_path.write_text(json.dumps(self.ids), encoding="utf-8")

    def replace_all(self, ids: list[str], vectors: np.ndarray) -> None:
        self.ids = ids
        self.vectors = vectors
        self.save()

    def append(self, ids: list[str], vectors: np.ndarray) -> None:
        """Add new vectors without touching existing ones - what an
        incremental (new-tickets-only) pipeline run uses instead of
        re-embedding and replacing the whole store."""
        if not ids:
            return
        if self.vectors is None:
            self.vectors = vectors
            self.ids = list(ids)
        else:
            self.vectors = np.vstack([self.vectors, vectors])
            self.ids.extend(ids)
        self.save()
