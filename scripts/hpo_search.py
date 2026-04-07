#!/usr/bin/env python3
"""Optuna hyperparameter optimization search for the NQ momentum model.

Usage:
    # Quick test (5 trials)
    python scripts/hpo_search.py --n-trials 5

    # Full search (200 trials, resumable)
    python scripts/hpo_search.py --n-trials 200

    # Inspect best result
    python scripts/hpo_search.py --show-best
"""

from __future__ import annotations

import argparse
import logging
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # suppress TF info/warnings

import sys
sys.path.insert(0, "/opt/vtech")

import optuna

from src.training.config import MLConfig
from src.training.trainer import Trainer

logger = logging.getLogger("vtech.hpo")


def objective(trial: optuna.Trial) -> float:
    """Single Optuna trial: sample params → train → return F1."""
    config = MLConfig.from_env()

    # Architecture
    config.model.lstm_units = [
        trial.suggest_categorical("lstm_0", [64, 128, 256]),
        trial.suggest_categorical("lstm_1", [32, 64, 128]),
    ]
    config.model.attention_heads = trial.suggest_categorical("attn_heads", [2, 4, 8])
    config.model.dense_dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.05)
    config.model.use_attention = trial.suggest_categorical("use_attention", [True, False])

    # Training
    config.training.learning_rate = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    config.training.batch_size = trial.suggest_categorical("batch", [32, 64, 128])
    config.model.sequence_length = trial.suggest_categorical("seq_len", [32, 64, 128])

    # Label
    config.labels.big_move_threshold = trial.suggest_int("threshold", 30, 75, step=5)

    # Feature groups
    for grp in ("book_pressure", "order_flow", "iv_surface", "daily_context",
                "vpin", "wavelets", "cross_asset", "macro_sentiment", "equity_context"):
        config.features.enabled_groups[grp] = trial.suggest_categorical(f"feat_{grp}", [True, False])

    if not any(config.features.enabled_groups.values()):
        raise optuna.TrialPruned()

    data_dir = os.environ.get("VTECH_DATA_DIR", "/opt/vtech/data")
    save_dir = f"{data_dir}/checkpoints/hpo/trial_{trial.number:04d}"

    trainer = Trainer(config)
    start = os.environ.get("VTECH_HPO_START", "2026-04-02")
    end = os.environ.get("VTECH_HPO_END", "2026-04-03")

    try:
        metrics = trainer.run(start, end, save_dir=save_dir, data_dir=data_dir)
    except (ValueError, FileNotFoundError) as exc:
        logger.warning("trial_failed", extra={"trial": trial.number, "error": str(exc)})
        raise optuna.TrialPruned()

    f1 = metrics.get("test_f1_macro", 0.0)

    logger.info(
        "trial_done",
        extra={"trial": trial.number, "f1": f1, "threshold": config.labels.big_move_threshold},
    )
    return f1


def show_best(storage: str) -> None:
    study = optuna.load_study(study_name="nq_momentum_v1", storage=storage)
    print(f"Completed trials: {len(study.trials)}")
    print(f"\nBest F1: {study.best_value:.4f}")
    print("Parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    best5 = sorted(study.trials, key=lambda t: t.value or 0, reverse=True)[:5]
    print("\nTop 5:")
    for t in best5:
        print(f"  Trial {t.number}: F1={t.value:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="HPO search for NQ momentum model")
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--show-best", action="store_true")
    parser.add_argument("--storage", default="sqlite:////opt/vtech/data/optuna.db")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    if args.show_best:
        show_best(args.storage)
        return

    study = optuna.create_study(
        study_name="nq_momentum_v1",
        direction="maximize",
        storage=args.storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    print(f"\nBest F1: {study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
