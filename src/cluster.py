"""Group tickets that describe the same recurring issue.

Two tickets are "the same issue" if their embeddings are close in cosine
distance (semantic similarity), so different wording of the same complaint
still lands in one cluster. Similarity search is chunked brute-force cosine
(pure numpy - vectors are already L2-normalized, so cosine similarity is
just a dot product), followed by union-find to turn pairwise similarity into
groups. Chunking keeps memory bounded (the full n x n similarity matrix is
never built at once) even at tens of thousands of tickets; no compiled
tree-search dependency (e.g. scikit-learn) required.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import settings

CHUNK_SIZE = 1000


@dataclass
class Cluster:
    ticket_indices: list[int] = field(default_factory=list)
    representative_index: int = -1


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def cluster_tickets(
    vectors: np.ndarray,
    categories: list[str],
    threshold: float | None = None,
) -> list[Cluster]:
    """Categories act as a soft guard rail: two tickets only merge if they
    also share a category, so an IT ticket never gets folded into an HR
    ticket just because the wording happens to be vaguely similar."""
    threshold = threshold if threshold is not None else settings.similarity_threshold
    n = vectors.shape[0]
    if n == 0:
        return []

    categories_arr = np.asarray(categories)
    uf = _UnionFind(n)
    for start in range(0, n, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n)
        sims = vectors[start:end] @ vectors.T  # (chunk, n) cosine similarities
        for local_i, global_i in enumerate(range(start, end)):
            neighbors = np.nonzero(sims[local_i] >= threshold)[0]
            if neighbors.size == 0:
                continue
            same_category = categories_arr[neighbors] == categories[global_i]
            for j in neighbors[same_category]:
                if j != global_i:
                    uf.union(global_i, int(j))

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)

    clusters: list[Cluster] = []
    for members in groups.values():
        centroid = vectors[members].mean(axis=0)
        sims = vectors[members] @ centroid
        representative_index = members[int(np.argmax(sims))]
        clusters.append(Cluster(ticket_indices=members, representative_index=representative_index))
    return clusters
