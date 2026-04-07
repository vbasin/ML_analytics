"""Label generation for 3/5-class classification with tie-breaking."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.training.config import LabelConfig


def _classify_3(
    up_move: pd.Series,
    down_move: pd.Series,
    threshold: float,
) -> pd.Series:
    """3-class: BIG_DOWN(0) / CONSOLIDATION(1) / BIG_UP(2)."""
    label = pd.Series(1, index=up_move.index, dtype=np.int32)  # default CONSOLIDATION
    label[up_move >= threshold] = 2    # BIG_UP
    label[down_move >= threshold] = 0  # BIG_DOWN

    # Tie-breaking: if both triggered, asymmetric — up wins (max_up > max_down)
    both = (up_move >= threshold) & (down_move >= threshold)
    label[both & (up_move >= down_move)] = 2
    label[both & (up_move < down_move)] = 0
    return label


def _classify_5(
    up_move: pd.Series,
    down_move: pd.Series,
    big_threshold: float,
    small_threshold: float,
) -> pd.Series:
    """5-class: STRONG_DOWN(0)/WEAK_DOWN(1)/CONSOLIDATION(2)/WEAK_UP(3)/STRONG_UP(4)."""
    label = pd.Series(2, index=up_move.index, dtype=np.int32)  # CONSOLIDATION

    # Weak moves
    label[(up_move >= small_threshold) & (up_move < big_threshold)] = 3   # WEAK_UP
    label[(down_move >= small_threshold) & (down_move < big_threshold)] = 1  # WEAK_DOWN

    # Strong moves
    label[up_move >= big_threshold] = 4    # STRONG_UP
    label[down_move >= big_threshold] = 0  # STRONG_DOWN

    # Tie-breaking: asymmetric max_up > max_down
    both_strong = (up_move >= big_threshold) & (down_move >= big_threshold)
    label[both_strong & (up_move >= down_move)] = 4
    label[both_strong & (up_move < down_move)] = 0

    both_weak = (
        (up_move >= small_threshold) & (up_move < big_threshold)
        & (down_move >= small_threshold) & (down_move < big_threshold)
    )
    label[both_weak & (up_move >= down_move)] = 3
    label[both_weak & (up_move < down_move)] = 1

    return label


def generate_labels(
    prices: pd.Series,
    config: LabelConfig | None = None,
) -> pd.DataFrame:
    """Generate classification labels from a price series.

    For each timestep, computes the maximum forward price move within each
    forward window and assigns labels according to config.label_type.
    """
    config = config or LabelConfig()
    threshold = config.big_move_threshold
    labels = pd.DataFrame(index=prices.index)

    for window in config.forward_windows:
        max_future = prices.rolling(window, min_periods=1).max().shift(-window)
        min_future = prices.rolling(window, min_periods=1).min().shift(-window)

        up_move = max_future - prices
        down_move = prices - min_future

        col = f"label_{window}"
        if config.label_type == "classification_5":
            labels[col] = _classify_5(up_move, down_move, threshold, config.small_move_threshold)
        else:
            labels[col] = _classify_3(up_move, down_move, threshold)

    # Primary label = shortest forward window
    labels["label"] = labels[f"label_{config.forward_windows[0]}"]

    return labels


def compute_class_weights(y: np.ndarray, num_classes: int = 3) -> dict[int, float]:
    """Compute inverse-frequency class weights to handle imbalance."""
    counts = np.bincount(y.astype(int), minlength=num_classes)
    total = counts.sum()
    weights = {}
    for cls in range(num_classes):
        weights[cls] = total / (num_classes * max(counts[cls], 1))
    return weights
