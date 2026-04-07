"""Environment-driven application configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from os import environ


@dataclass(frozen=True)
class AppConfig:
    """Immutable application configuration loaded from environment variables."""

    databento_api_key: str = field(repr=False)
    pg_dsn: str = field(repr=False)
    data_dir: str = "/opt/vtech/data"
    log_level: str = "INFO"
    features_timestep: str = "5s"
    label_threshold: int = 50
    sequence_length: int = 128
    forward_windows: tuple[int, ...] = (60, 120, 180, 360)
    trading_hours_start: str = "08:00"
    trading_hours_end: str = "11:00"
    timezone: str = "America/Chicago"

    @classmethod
    def from_env(cls) -> AppConfig:
        fwd = environ.get("VTECH_FORWARD_WINDOWS", "60,120,180,360")
        pg_dsn = (
            f"postgresql://{environ['PGUSER']}:{environ['PGPASSWORD']}"
            f"@{environ.get('PGHOST', 'localhost')}:{environ.get('PGPORT', '5432')}"
            f"/{environ.get('PGDATABASE', 'vtech_market')}"
        )
        return cls(
            databento_api_key=environ["DATABENTO_API_KEY"],
            pg_dsn=pg_dsn,
            data_dir=environ.get("VTECH_DATA_DIR", "/opt/vtech/data"),
            log_level=environ.get("VTECH_LOG_LEVEL", "INFO"),
            features_timestep=environ.get("VTECH_FEATURES_TIMESTEP", "5s"),
            label_threshold=int(environ.get("VTECH_LABEL_THRESHOLD", "50")),
            sequence_length=int(environ.get("VTECH_SEQUENCE_LENGTH", "128")),
            forward_windows=tuple(int(x) for x in fwd.split(",")),
            trading_hours_start=environ.get("VTECH_TRADING_HOURS_START", "08:00"),
            trading_hours_end=environ.get("VTECH_TRADING_HOURS_END", "11:00"),
            timezone=environ.get("VTECH_TIMEZONE", "America/Chicago"),
        )
