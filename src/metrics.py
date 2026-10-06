"""Metrics used in the preregistered evaluation protocol."""
from __future__ import annotations
import numpy as np

def rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    pred, truth = np.asarray(pred), np.asarray(truth)
    return float(np.sqrt(np.mean((pred - truth) ** 2)))

def empirical_coverage(samples: np.ndarray, truth: np.ndarray, alpha: float = 0.1):
    samples = np.asarray(samples)
    truth = np.asarray(truth)
    lo = np.quantile(samples, alpha / 2, axis=0)
    hi = np.quantile(samples, 1 - alpha / 2, axis=0)
    covered = (truth >= lo) & (truth <= hi)
    return float(np.mean(covered)), float(np.mean(hi - lo))

def event_probability(samples: np.ndarray, predicate) -> float:
    samples = np.asarray(samples)
    return float(np.mean([bool(predicate(path)) for path in samples]))

def brier_score(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    p, y = np.asarray(probabilities), np.asarray(outcomes)
    return float(np.mean((p - y) ** 2))

def reliability_bins(probabilities: np.ndarray, outcomes: np.ndarray, bins: int = 10):
    probabilities = np.asarray(probabilities)
    outcomes = np.asarray(outcomes)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= left) & (probabilities < right if right < 1 else probabilities <= right)
        if np.any(mask):
            rows.append({"bin_left": float(left), "bin_right": float(right), "mean_probability": float(np.mean(probabilities[mask])), "event_rate": float(np.mean(outcomes[mask])), "count": int(np.sum(mask))})
    return rows
