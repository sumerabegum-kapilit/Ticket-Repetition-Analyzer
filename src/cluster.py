"""Group tickets that describe the same recurring issue.

Two tickets are "the same issue" if their embeddings are close in cosine
distance (semantic similarity), so different wording of the same complaint
still lands in one cluster.

Clustering is centroid-based, not single-linkage: a ticket joins the
existing cluster whose *running average* vector it's most similar to (if
that similarity clears the threshold), otherwise it starts a new cluster.
An earlier version used union-find over any pairwise match, which chains -
if A~B and B~C both clear the threshold, A and C end up grouped even if
they're unrelated, and on real (verbose, templated) ticket text this
chaining collapsed the entire dataset into one giant cluster once the
threshold dropped low enough to catch paraphrases/synonyms. Comparing
against a cluster's centroid instead of any single member means one
borderline ticket can't drag two otherwise-unrelated clusters together.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import settings


@dataclass
class Cluster:
    ticket_indices: list[int] = field(default_factory=list)
    representative_index: int = -1


def cluster_tickets(
    vectors: np.ndarray,
    threshold: float | None = None,
) -> list[Cluster]:
    """Two tickets merge purely on semantic similarity - no category/domain
    guard. An earlier version required an exact category match too, but on
    real ticket data category is often an inconsistent free-text label (e.g.
    "Easychit" vs "EasyChit Client" for the same product) rather than a
    clean taxonomy, so that guard was silently blocking genuine repeats
    worded almost identically (0.88+ similarity) just because of a category
    string mismatch. The similarity threshold alone is the correct signal
    for "same issue"."""
    threshold = threshold if threshold is not None else settings.similarity_threshold
    n = vectors.shape[0]
    if n == 0:
        return []

    members_by_cluster: list[list[int]] = []
    sums: list[np.ndarray] = []  # running (unnormalized) sum of member vectors

    # centroid_mat holds L2-normalized centroids in its first `k` rows.
    # Preallocated with doubling growth so most iterations just slice a view
    # (no copy) instead of rebuilding the whole (k, dim) array from a list
    # on every single ticket - that rebuild is what made this O(n*k) blow up
    # into minutes on real-sized datasets.
    k = 0
    capacity = 256
    dim = vectors.shape[1]
    centroid_mat = np.zeros((capacity, dim), dtype=vectors.dtype)

    for i in range(n):
        v = vectors[i]
        best_cluster, best_sim = -1, -1.0
        if k:
            sims = centroid_mat[:k] @ v
            best_cluster = int(np.argmax(sims))
            best_sim = float(sims[best_cluster])

        if best_sim >= threshold:
            members_by_cluster[best_cluster].append(i)
            sums[best_cluster] = sums[best_cluster] + v
            centroid_mat[best_cluster] = sums[best_cluster] / (np.linalg.norm(sums[best_cluster]) + 1e-12)
        else:
            if k == capacity:
                capacity *= 2
                grown = np.zeros((capacity, dim), dtype=vectors.dtype)
                grown[:k] = centroid_mat[:k]
                centroid_mat = grown
            centroid_mat[k] = v
            sums.append(v.copy())
            members_by_cluster.append([i])
            k += 1

    clusters: list[Cluster] = []
    for members in members_by_cluster:
        centroid = vectors[members].mean(axis=0)
        sims = vectors[members] @ centroid
        representative_index = members[int(np.argmax(sims))]
        clusters.append(Cluster(ticket_indices=members, representative_index=representative_index))
    return clusters
