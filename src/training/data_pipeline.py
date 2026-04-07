"""Data pipeline: load Parquet features, split, preprocess, create sequences."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.training.config import DataConfig

logger = logging.getLogger("vtech.training.data_pipeline")

SCALERS = {
    "standard": StandardScaler,
    "robust": RobustScaler,
    "minmax": MinMaxScaler,
}


class DataPipeline:
    """Loads cached Parquet features and prepares them for training."""

    def __init__(self, data_dir: str | Path, config: DataConfig | None = None) -> None:
        self.data_dir = Path(data_dir)
        self.config = config or DataConfig()
        self.scaler = None
        self.scaler_type: str = "robust"
        self.feature_columns: list[str] = []

    # ── load ──────────────────────────────────────────────────────

    def load_features(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load and concatenate daily Parquet feature files for [start, end)."""
        parquet_dir = self.data_dir / "parquet"
        dates = pd.date_range(start_date, end_date, freq="B")  # business days

        frames: list[pd.DataFrame] = []
        for d in dates:
            path = parquet_dir / f"features_{d.date()}.parquet"
            if path.exists():
                frames.append(pd.read_parquet(path))
            else:
                logger.debug("missing_parquet", extra={"date": str(d.date())})

        if not frames:
            raise FileNotFoundError(f"No feature files found in {parquet_dir}")

        df = pd.concat(frames).sort_index()
        logger.info("loaded_features", extra={"rows": len(df), "cols": df.shape[1]})
        return df

    # ── split ─────────────────────────────────────────────────────

    def split_temporal(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Temporal train/val/test split (no shuffle — preserves time order)."""
        n = len(df)
        train_end = int(n * self.config.train_ratio)
        val_end = train_end + int(n * self.config.val_ratio)
        return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]

    # ── preprocess ────────────────────────────────────────────────

    def fit_scaler(self, df: pd.DataFrame, feature_cols: list[str], scaler_type: str = "robust") -> None:
        """Fit a scaler on the training set."""
        self.feature_columns = feature_cols
        self.scaler_type = scaler_type
        scaler_cls = SCALERS.get(scaler_type, RobustScaler)
        self.scaler = scaler_cls()
        self.scaler.fit(df[feature_cols].values)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted scaler to feature columns."""
        if self.scaler is None:
            raise RuntimeError("Call fit_scaler() first")
        out = df.copy()
        out[self.feature_columns] = self.scaler.transform(df[self.feature_columns].values)
        return out

    # ── sequences ─────────────────────────────────────────────────

    @staticmethod
    def create_sequences(
        X: np.ndarray,
        y: np.ndarray,
        seq_len: int = 128,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Slide a window of *seq_len* over X to produce 3-D sequence arrays."""
        Xs, ys = [], []
        for i in range(seq_len, len(X)):
            Xs.append(X[i - seq_len : i])
            ys.append(y[i])
        return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int32)
