#!/usr/bin/env python3
"""AUC and paired-event uncertainty for tracker two-sample comparisons."""

import numpy as np


def weighted_auc(labels, scores, weights=None):
    """Probability that a weighted class-1 score exceeds a class-0 score."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    weights = (np.ones(len(labels), dtype=np.float64) if weights is None
               else np.asarray(weights, dtype=np.float64))
    if (labels.shape != scores.shape or labels.shape != weights.shape
            or np.any(~np.isin(labels, (0, 1)))
            or np.any(~np.isfinite(scores)) or np.any(~np.isfinite(weights))
            or np.any(weights < 0)):
        raise ValueError("AUC inputs must be finite aligned scores and nonnegative weights")
    total_negative = weights[labels == 0].sum()
    total_positive = weights[labels == 1].sum()
    if total_negative <= 0 or total_positive <= 0:
        raise ValueError("AUC requires positive weight in both classes")
    order = np.argsort(scores, kind="stable")
    s, y, w = scores[order], labels[order], weights[order]
    starts = np.r_[0, np.flatnonzero(np.diff(s)) + 1]
    negative_here = np.add.reduceat(w * (y == 0), starts)
    positive_here = np.add.reduceat(w * (y == 1), starts)
    negative_below = np.cumsum(negative_here) - negative_here
    favorable = np.sum(positive_here * (negative_below + 0.5 * negative_here))
    return float(favorable / (total_negative * total_positive))


def paired_event_bootstrap(labels, scores, weights, groups, n_events,
                           n_draws=1000, seed=12345):
    """Resample SIM/COUNT event pairs, keeping all tracks in each event together.

    Group 0..n_events-1 is sample A; n_events..2*n_events-1 is sample B.
    The interval describes variation conditional on the finite source pool.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    weights = np.asarray(weights)
    groups = np.asarray(groups)
    if n_events < 2 or n_draws < 1:
        return None
    if not (labels.shape == scores.shape == weights.shape == groups.shape):
        raise ValueError("bootstrap inputs must have aligned shapes")
    if np.any(groups < 0) or np.any(groups >= 2 * n_events):
        raise ValueError("group IDs must identify prepared event pairs")
    pair_indices = [np.flatnonzero((groups == event) | (groups == event + n_events))
                    for event in range(n_events)]
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_draws):
        sampled = rng.integers(0, n_events, n_events)
        indices = np.concatenate([pair_indices[event] for event in sampled])
        try:
            aucs.append(weighted_auc(labels[indices], scores[indices], weights[indices]))
        except ValueError:  # An all-empty class can occur in a small track sample.
            continue
    if not aucs:
        return None
    low, high = np.quantile(aucs, [0.025, 0.975])
    return {"lower": float(low), "upper": float(high),
            "valid_draws": len(aucs), "requested_draws": n_draws,
            "interpretation": "paired-event bootstrap conditional on the reused source pool"}
