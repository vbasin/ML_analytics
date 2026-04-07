"""ML configuration dataclasses for the Databento pipeline.

Mirrors ml/config.py but reads defaults from environment variables
and is adapted for the new data sources.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from os import environ


@dataclass
class FeatureConfig:
    wavelet_window: int = 128
    wavelet_scales: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    max_spread_pct: float = 0.25
    min_option_price: float = 0.25
    large_trade_threshold: int = 10  # contracts — threshold for lt_ features
    # Feature groups enabled by default
    enabled_groups: dict[str, bool] = field(default_factory=lambda: {
        "book_pressure": True,
        "order_flow": True,
        "trade_location": True,
        "same_side": True,
        "large_trade_cvd": True,
        "microstructure": True,
        "trade_arrival": True,
        "realized_vol": True,
        "sub_bar_dynamics": True,
        "volume_profile": True,
        "iv_surface": True,
        "dealer_gex": True,
        "daily_context": True,
        "wavelets": True,
        "candle_structure": True,
        "time_context": True,
        "vwap": True,
        "cross_asset": True,
        "economic_calendar": True,
        "hawkes_clustering": True,
        "higher_timeframe": True,
    })


@dataclass
class DataConfig:
    feature_timestep: str = "5s"
    lookback_minutes: int = 128
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    trading_start: str = "08:00"
    trading_end: str = "11:00"
    timezone: str = "America/Chicago"
    scaler_type: str = "robust"  # "standard", "robust", or "minmax"


@dataclass
class LabelConfig:
    big_move_threshold: float = 50.0
    forward_windows: list[int] = field(default_factory=lambda: [60, 120, 180, 360])
    label_type: str = "classification_3"
    small_move_threshold: float = 25.0


@dataclass
class ModelConfig:
    sequence_length: int = 128
    lstm_units: list[int] = field(default_factory=lambda: [64, 32])
    lstm_dropout: float = 0.2
    lstm_recurrent_dropout: float = 0.1
    use_attention: bool = True
    attention_heads: int = 4
    dense_units: list[int] = field(default_factory=lambda: [64, 32])
    dense_dropout: float = 0.3
    num_classes: int = 3


@dataclass
class TrainingConfig:
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 0.001
    early_stop_patience: int = 10
    early_stop_min_delta: float = 0.001
    lr_reduce_patience: int = 5
    lr_reduce_factor: float = 0.5
    min_lr: float = 1e-6
    use_class_weights: bool = True
    save_best_only: bool = True


@dataclass
class MLConfig:
    features: FeatureConfig = field(default_factory=FeatureConfig)
    data: DataConfig = field(default_factory=DataConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_env(cls) -> MLConfig:
        fwd = environ.get("VTECH_FORWARD_WINDOWS", "60,120,180,360")
        return cls(
            data=DataConfig(
                feature_timestep=environ.get("VTECH_FEATURES_TIMESTEP", "5s"),
                trading_start=environ.get("VTECH_TRADING_HOURS_START", "08:00"),
                trading_end=environ.get("VTECH_TRADING_HOURS_END", "11:00"),
                timezone=environ.get("VTECH_TIMEZONE", "America/Chicago"),
            ),
            labels=LabelConfig(
                big_move_threshold=float(environ.get("VTECH_LABEL_THRESHOLD", "50")),
                forward_windows=[int(x) for x in fwd.split(",")],
            ),
            model=ModelConfig(
                sequence_length=int(environ.get("VTECH_SEQUENCE_LENGTH", "128")),
            ),
        )
