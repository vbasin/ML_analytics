"""End-to-end training pipeline orchestrator.

Run as: python -m src.training.trainer --start=2025-04-01 --end=2026-04-01
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.training.config import MLConfig
from src.training.data_pipeline import DataPipeline
from src.training.evaluate import evaluate_model, save_metrics
from src.training.labels import compute_class_weights, generate_labels

logger = logging.getLogger("vtech.training.trainer")

CLASS_NAMES_3 = ["BIG_DOWN", "CONSOLIDATION", "BIG_UP"]
CLASS_NAMES_5 = ["STRONG_DOWN", "WEAK_DOWN", "CONSOLIDATION", "WEAK_UP", "STRONG_UP"]

# Maps config feature-group names → column prefixes in the Parquet files
FEATURE_GROUP_PREFIXES = {
    "book_pressure": "bp_",
    "order_flow": "of_",
    "iv_surface": "os_",
    "daily_context": "dc_",
    "vpin": "ms_",
    "wavelets": "wv_",
    "candle_structure": "cs_",
    "time_context": "tc_",
    "vwap": "vw_",
    "cross_asset": "ca_",
    "macro_sentiment": "mx_",
    "equity_context": "eq_",
}


class Trainer:
    """10-step training pipeline, reading from Parquet feature cache."""

    def __init__(self, config: MLConfig | None = None) -> None:
        self.config = config or MLConfig()

    def run(
        self,
        start_date: str,
        end_date: str,
        save_dir: str | None = None,
        data_dir: str | None = None,
    ) -> dict:
        """Execute full pipeline and return metrics dict."""
        import os

        t0 = time.time()
        save_path = Path(save_dir or "checkpoints/latest")
        save_path.mkdir(parents=True, exist_ok=True)

        data_dir = data_dir or os.environ.get("VTECH_DATA_DIR", "/opt/vtech/data")
        pipeline = DataPipeline(data_dir, self.config.data)

        num_classes = self.config.model.num_classes
        class_names = CLASS_NAMES_5 if num_classes == 5 else CLASS_NAMES_3

        # 1. Load features
        logger.info("step", extra={"n": 1, "action": "load_features"})
        features_df = pipeline.load_features(start_date, end_date)
        feature_cols = [c for c in features_df.columns if not c.startswith("label") and not c.startswith("_")]

        # Filter feature columns by enabled groups
        enabled = self.config.features.enabled_groups
        disabled_prefixes = [
            FEATURE_GROUP_PREFIXES[grp]
            for grp, on in enabled.items()
            if not on and grp in FEATURE_GROUP_PREFIXES
        ]
        if disabled_prefixes:
            feature_cols = [c for c in feature_cols if not any(c.startswith(p) for p in disabled_prefixes)]
        if not feature_cols:
            raise ValueError("All feature groups disabled — nothing to train on")

        # 2. Generate labels from _close price column
        logger.info("step", extra={"n": 2, "action": "generate_labels"})
        if "_close" not in features_df.columns:
            raise ValueError(
                "Feature Parquet missing '_close' column. "
                "Rebuild features with the updated engine."
            )
        prices = features_df["_close"]
        labels_df = generate_labels(prices, self.config.labels)

        full_df = pd.concat([features_df[feature_cols], labels_df], axis=1).dropna(subset=["label"])

        # Log class distribution
        dist = full_df["label"].value_counts().sort_index()
        for cls_id, count in dist.items():
            name = class_names[int(cls_id)] if int(cls_id) < len(class_names) else str(cls_id)
            logger.info("class_dist", extra={"class": name, "count": int(count),
                        "pct": f"{count / len(full_df) * 100:.1f}%"})

        logger.info("samples", extra={"n": len(full_df), "features": len(feature_cols)})

        # 3. Temporal split
        logger.info("step", extra={"n": 3, "action": "split"})
        train_df, val_df, test_df = [s.copy() for s in pipeline.split_temporal(full_df)]

        # 4. Preprocess — outlier detection + scaling
        logger.info("step", extra={"n": 4, "action": "preprocess"})
        # Outlier clipping: z-score > 5 on training set
        train_mean = train_df[feature_cols].mean()
        train_std = train_df[feature_cols].std().replace(0, 1)
        z_threshold = 5.0
        for df_part in [train_df, val_df, test_df]:
            z = (df_part[feature_cols] - train_mean) / train_std
            clipped = df_part[feature_cols].clip(
                lower=(train_mean - z_threshold * train_std),
                upper=(train_mean + z_threshold * train_std),
                axis=1,
            )
            df_part[feature_cols] = clipped

        scaler_type = self.config.data.scaler_type if hasattr(self.config.data, "scaler_type") else "robust"
        pipeline.fit_scaler(train_df, feature_cols, scaler_type=scaler_type)
        train_s = pipeline.transform(train_df)
        val_s = pipeline.transform(val_df)
        test_s = pipeline.transform(test_df)

        # Replace any remaining NaN/inf with 0 after scaling
        for df_part in [train_s, val_s, test_s]:
            df_part[feature_cols] = df_part[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

        # 5. Create sequences
        logger.info("step", extra={"n": 5, "action": "sequences"})
        seq_len = self.config.model.sequence_length
        X_train, y_train = DataPipeline.create_sequences(train_s[feature_cols].values, train_s["label"].values, seq_len)
        X_val, y_val = DataPipeline.create_sequences(val_s[feature_cols].values, val_s["label"].values, seq_len)
        X_test, y_test = DataPipeline.create_sequences(test_s[feature_cols].values, test_s["label"].values, seq_len)
        logger.info("sequence_shapes", extra={
            "train": list(X_train.shape), "val": list(X_val.shape), "test": list(X_test.shape),
        })

        if X_train.shape[0] == 0:
            raise ValueError(f"No training sequences — need at least {seq_len} rows in training split")

        # 6. Class weights
        class_weights = None
        if self.config.training.use_class_weights:
            class_weights = compute_class_weights(y_train, num_classes)

        # 7. Build model
        logger.info("step", extra={"n": 7, "action": "build_model"})
        model = self._build_model((X_train.shape[1], X_train.shape[2]))

        # 8. Train
        logger.info("step", extra={"n": 8, "action": "train"})
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

        callbacks = [
            EarlyStopping(
                patience=self.config.training.early_stop_patience,
                min_delta=self.config.training.early_stop_min_delta,
                restore_best_weights=True,
            ),
            ReduceLROnPlateau(
                patience=self.config.training.lr_reduce_patience,
                factor=self.config.training.lr_reduce_factor,
                min_lr=self.config.training.min_lr,
            ),
            ModelCheckpoint(
                str(save_path / "model.keras"),
                save_best_only=True,
                monitor="val_loss",
            ),
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.training.epochs,
            batch_size=self.config.training.batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1,
        )

        # 9. Evaluate
        logger.info("step", extra={"n": 9, "action": "evaluate"})
        metrics = evaluate_model(model, X_test, y_test, class_names=class_names)
        metrics["training_time_seconds"] = time.time() - t0
        metrics["history"] = {k: [float(v) for v in vals] for k, vals in history.history.items()}

        # 10. Save artifacts
        logger.info("step", extra={"n": 10, "action": "save"})
        model.save(str(save_path / "model.keras"))
        save_metrics(metrics, save_path)
        with open(save_path / "features.json", "w") as f:
            json.dump({"features": feature_cols, "scaler": scaler_type}, f, indent=2)

        logger.info("done", extra={
            "accuracy": metrics["test_accuracy"],
            "f1_macro": metrics["test_f1_macro"],
            "seconds": metrics["training_time_seconds"],
        })
        return metrics

    # ── model builder ────────────────────────────────────────────

    def _build_model(self, input_shape: tuple[int, int]):
        from src.models.attention_lstm import build_attention_lstm

        cfg = self.config.model
        return build_attention_lstm(
            input_shape=input_shape,
            num_classes=cfg.num_classes,
            lstm_units=cfg.lstm_units,
            attention_heads=cfg.attention_heads,
            dense_units=cfg.dense_units,
            dropout=cfg.dense_dropout,
            use_attention=cfg.use_attention,
            learning_rate=self.config.training.learning_rate,
        )


# ── CLI ──────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train NQ momentum model")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    config = MLConfig.from_env()
    trainer = Trainer(config)
    trainer.run(args.start, args.end, save_dir=args.save_dir, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
