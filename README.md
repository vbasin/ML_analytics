# ML Analytics

ML-driven futures analytics pipeline for **NQ futures and options** — feature
engineering, model training, and live inference on top of
[Databento](https://databento.com/) market data and TimescaleDB.

## Architecture

```
Databento  ──►  Live Streamer  ──►  TimescaleDB (bbo_1s / tbbo)
                                        │
                                   Feature Engine  ──►  Parquet
                                        │
                                   Trainer (Attention-LSTM)
                                        │
                                   Checkpoints / MLflow
```

| Layer | Packages |
|---|---|
| Data acquisition | `acquisition` — live streaming and historical backfill via Databento |
| Ingestion | `ingestion` — loader and data-quality checks |
| Feature engineering | `features` — 20+ microstructure, order-flow, and options-surface features |
| Models | `models` — Attention-LSTM (TensorFlow/Keras) |
| Training | `training` — data pipeline, labelling, HPO via Optuna, MLflow tracking |
| Config / infra | `common` — env-driven config, DB pool, structured logging |

## Quickstart

```bash
# Clone
git clone https://github.com/vbasin/ML_analytics.git
cd ML_analytics

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install core + training extras
pip install -e ".[training]"

# Copy and fill in environment variables
cp .env.example .env
# edit .env with your credentials

# Run the feature engine
python -m features.engine
```

## Environment variables

See [`.env.example`](.env.example) for the full list. Key variables:

| Variable | Purpose |
|---|---|
| `DATABENTO_API_KEY` | Databento API key |
| `PGUSER` / `PGPASSWORD` / `PGDATABASE` | PostgreSQL / TimescaleDB credentials |
| `VTECH_DATA_DIR` | Base path for raw data, parquet, and checkpoints |

## Project layout

```
src/
  acquisition/     # Databento live + historical
  common/          # Config, DB, logging
  features/        # 20+ feature modules + engine
  ingestion/       # Data loader, quality checks
  models/          # Attention-LSTM
  training/        # Data pipeline, labels, trainer, HPO
scripts/           # One-off backfill and verification scripts
systemd/           # systemd unit files for production services
docs/              # Architecture and design documents
```

## License

MIT
