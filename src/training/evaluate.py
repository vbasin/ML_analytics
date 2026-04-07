"""Evaluation utilities: metrics, confusion matrix, per-class reporting."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

logger = logging.getLogger("vtech.training.evaluate")

CLASS_NAMES_3 = ["BIG_DOWN", "CONSOLIDATION", "BIG_UP"]


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str] | None = None,
) -> dict:
    """Run evaluation on a test set and return a metrics dict."""
    if class_names is None:
        class_names = CLASS_NAMES_3
    num_classes = len(class_names)
    labels_list = list(range(num_classes))

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)

    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    f1_per_class = f1_score(y_test, y_pred, average=None, labels=labels_list, zero_division=0).tolist()
    cm = confusion_matrix(y_test, y_pred, labels=labels_list).tolist()

    report = classification_report(
        y_test, y_pred, target_names=class_names, labels=labels_list, output_dict=True, zero_division=0,
    )

    metrics = {
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
        "test_f1_macro": f1_macro,
        "test_f1_per_class": dict(zip(class_names, f1_per_class)),
        "confusion_matrix": cm,
        "classification_report": report,
    }
    logger.info("evaluation", extra={
        "accuracy": float(accuracy), "f1_macro": f1_macro,
    })
    return metrics


def save_metrics(metrics: dict, save_dir: str | Path) -> Path:
    """Write metrics to a JSON file."""
    path = Path(save_dir) / "training_result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    return path
