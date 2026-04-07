## CRITICAL BUG: Forward Windows Are Bars, Not Minutes

**Blueprint says:** Forward windows of 5, 10, 15, 30 **minutes**
**Code does:** `forward_windows: [5, 10, 15, 30]` applied to `prices.rolling(window).max().shift(-window)` — at 10s timestep, these are **50 seconds, 100 seconds, 150 seconds, 300 seconds**.

Your model is predicting 50-point moves within 50 seconds–5 minutes, not 5–30 minutes as the blueprint intended. This fundamentally changes the prediction task. To match the blueprint, the windows should be `[30, 60, 90, 180]` bars (= 5, 10, 15, 30 minutes at 10s resolution).

**Recommendation: FIX.** This is the single highest-impact change. Whether you want bar-based or minute-based windows is a design choice, but it needs to be *intentional*. The current defaults accidentally create an extremely short-horizon prediction that's much harder for the model to learn — predicting 50 NQ points in 50 seconds requires catching flash crashes rather than momentum moves.

---

## Full Gap Analysis: Blueprint vs Implementation

### TIER 1 — Implement Now (High Impact, Moderate Effort)

| # | Gap | Blueprint | Current | Why Implement |
|---|-----|-----------|---------|---------------|
| **1** | **Forward window units** | 5,10,15,30 min | 5,10,15,30 bars (50s–5min) | See above — fundamentally wrong prediction horizon |
| **2** | **Close session missing** | 8:30–11:00 + **14:00–15:00 CT** | 08:00–11:00 only | The blueprint explicitly targets the close session as a second trading window. The model never sees 14:00–15:00 data. This is a simple config change in `DataConfig.trading_end` plus adding close-session blocks in time_context.py. |
| **3** | **Confidence thresholding** | Only trade when softmax > 0.85 | Every bar gets a prediction | This is *core* to the strategy: "1–2 trades/day, only when confidence is extreme." Without thresholding, the model output is meaningless for execution. Add a post-prediction filter — doesn't change the model itself, just how predictions are consumed. |
| **4** | **Probability calibration** | Platt scaling / isotonic regression | Raw softmax | Softmax probabilities are notoriously uncalibrated — a 0.85 softmax output may only correspond to 0.60 true probability. Without calibration, confidence thresholding is unreliable. scikit-learn's `CalibratedClassifierCV` or custom Platt scaling on the validation set. Must be implemented alongside #3. |
| **5** | **Walk-forward validation** | Expanding/sliding window retraining | Single 70/15/15 split | A single temporal split is fragile — your test set covers only the last ~2 months. Markets change regime. Walk-forward (retrain monthly on expanding window, test on next month) gives you 10+ independent test periods and reveals whether the model adapts or decays. This is the #1 best-practice gap. |
| **6** | **Realized volatility + vol-of-vol** | Rolling std of log returns + std of rolling vol | Not implemented anywhere | Your feature set has *no direct volatility measure*. Candle range is a proxy but not the same. Realized vol and vol-of-vol are fundamental regime indicators — compression→expansion cycles are exactly what produces 50-point moves. Simple to compute from existing data. |
| **7** | **Adaptive label threshold** | Scale 50 pts by current ATR | Fixed 50 points | 50 NQ points during a 15-ATR day is a routine move; during a 5-ATR day it's a crash. Fixed thresholds create label noise — a "BIG_UP" on a quiet day is structurally different from a "BIG_UP" during an FOMC announcement. Scale threshold as `base_threshold * current_ATR / median_ATR`. |

### TIER 2 — Implement Next (High Impact, More Effort)

| # | Gap | Blueprint | Current | Why Implement |
|---|-----|-----------|---------|---------------|
| **8** | **Trade location / effective spread** | `(trade_price - mid) / half_spread` and `2 * ‖trade-mid‖` | Not computed | These are the strongest microstructure signals for urgency. A trade that crosses the full spread signals desperation — the buyer/seller needs to get filled NOW. This directly predicts momentum ignition. Computable from existing TBBO data (price, bid_px, ask_px). |
| **9** | **Fleeting liquidity detection** | Rolling variance of bid_sz/ask_sz over 5–10s | Not computed | Flickering quotes mean the displayed depth is fake — market makers will yank those orders when stressed. High bid_sz variance + declining bid_sz mean = the bid will evaporate. Computable from BBO-1s data. |
| **10** | **Amihud illiquidity ratio** | `‖return‖ / dollar_volume` rolling | Not computed | Spikes when price moves on thin volume — a fragile state where the next large order causes outsized impact. This is a "readiness" indicator that complements VPIN and Kyle's Lambda. Simple to compute from TBBO. |
| **11** | **Consecutive same-side count** | Run length of same-direction trades | Not computed | 15 consecutive buy trades is very different from 15 alternating buy/sell trades even if the net CVD is the same. Run length captures momentum ignition — when one side is relentlessly hitting, a cascade is likely. |
| **12** | **Large trade CVD** | CVD filtered to trades > 10 contracts | Not computed | Standard CVD is dominated by 1-lot retail/algo noise. Filtering to 10+ contracts isolates institutional flow. When large-trade CVD diverges from total CVD, smart money is positioning differently from the crowd. |
| **13** | **Volume profile (POC, Value Area)** | Compute from TBBO at each price level | Not implemented | Point of Control (highest-volume price) acts as a magnet; breakout from the Value Area (70% of volume) is directional. These are the most widely used institutional reference levels and the model currently has no concept of price-level volume distribution. |
| **14** | **Multi-resolution features** | 1s microstructure + 10s features | 10s only | The blueprint explicitly notes: "microstructure features change at sub-second frequencies — aggregating to 10s loses signal." Book pressure velocity, fleeting liquidity, and trade arrival rate are most informative at 1–3 second resolution. Either add a 1s parallel feature path or use finer-grained aggregation for P1/P2 features within the 10s bar. |

### TIER 3 — Implement When Core Is Solid (Medium Impact)

| # | Gap | Blueprint | Current | Why Implement |
|---|-----|-----------|---------|---------------|
| **15** | **XGBoost baseline** | Non-sequential diagnostic model | Not present | If an XGBoost on the same features matches the LSTM, then the sequence structure doesn't matter and you can use a simpler, faster model. If LSTM significantly outperforms, the temporal patterns are real. 2 hours of work, massive diagnostic value. |
| **16** | **SHAP / feature importance** | Permutation importance per group | Not implemented | You have 138 features but no idea which ones drive predictions. Some feature groups may be pure noise. SHAP reveals whether your "Priority 1" features (microstructure) actually matter more than "Priority 7" (time context). Also validates the blueprint's signal hierarchy. |
| **17** | **Economic calendar encoding** | FOMC/CPI/NFP day flags | Not implemented | These days produce categorically different trading behavior. The model conflates FOMC-morning with regular-morning. A simple binary flag (`is_fomc_day`, `is_cpi_day`, `minutes_to_announcement`) lets the model learn distinct regimes. |
| **18** | **Trade arrival rate + acceleration** | Trades/second rolling + d(rate)/dt | Only `of_trade_count` | Trade count per 10s bar is a coarse proxy. The rate of change (acceleration) of arrival is the transition signal — when arrival rate doubles in 10 seconds, something is happening. |
| **19** | **VWAP slope** | d(VWAP)/dt | Not computed | A rising VWAP means the volume-weighted average price is moving up — new buying is at higher prices. VWAP slope distinguishes trending from mean-reverting regimes. One-line addition. |
| **20** | **30-minute opening range** | OR high/low for first 30 min | Only 5m and 15m | The 30-minute opening range is the most widely watched institutional level. Adding it is a trivial change in time_context.py. |
| **21** | **Spread volatility** | Rolling std of spread | Not computed | Spread compression → expansion is a leading signal. You have spread mean and spread z-score, but not spread stability. Low spread-std followed by a spike = breakout imminent. |
| **22** | **Book pressure velocity** | d(pressure)/dt | Only depth change | Rate of change of the imbalance ratio captures *acceleration* of one side thinning. You have the static level but not the derivative. |
| **23** | **Multi-horizon ensemble** | One model per horizon → fuse | Single model, primary label = shortest window | A 5-minute move and a 30-minute move are structurally different setups. Training separate models per horizon and ensembling (weighted average of probabilities) captures both fast and slow signals. Moderate complexity increase. |
| **24** | **Execution cost modeling** | Subtract spread + slippage from metric | Not modeled | Ensures the edge survives real-world trading costs. A model with 55% accuracy and 2-tick slippage may have negative expectancy. Should be added to evaluate.py as a net-P&L metric alongside F1. |

### TIER 4 — Defer (High Complexity, Requires Missing Data)

| # | Gap | Blueprint | Current | Status |
|---|-----|-----------|---------|--------|
| **25** | **Dealer GEX estimation** | Gamma exposure from options surface | Can't compute — NQ.OPT data not loaded | Blueprint calls this "the holy grail feature." It requires options data (definitions + BBO-1s for all NQ options) which is a $15–30 Databento cost and ~56 GB download. Defer until core model is validated. |
| **26** | **Hawkes process intensity** | Self-exciting event model | Not implemented | Theoretically strong (large moves cluster) but complex to implement and tune. An exponentially-weighted count of large-move events is a simpler proxy that captures 80% of the signal. |
| **27** | **Regime-conditioned prediction** | Two-stage: classify regime → predict within | Single model | Adds a full second model. Defer until you have evidence that a single model underperforms in specific regimes (you'll get this evidence from walk-forward validation). |
| **28** | **Full delta-based IV surface** | 25-delta risk reversal, term structure | Moneyness-based approximation | Requires loaded options data (same as GEX). Current implementation is structurally correct but approximates delta with moneyness. |

### Already Implemented (No Gap)

| Blueprint Item | Status |
|---|---|
| VPIN | ✅ `ms_vpin` in microstructure.py |
| Kyle's Lambda | ✅ `ms_kyle_lambda_30` |
| CVD + delta windows | ✅ `of_cvd_chg_*`, `of_delta_*` |
| Attention-LSTM architecture | ✅ Matches blueprint spec |
| Wavelet decomposition (Haar, multi-scale) | ✅ 6 scales + energy + slope |
| 3-class + 5-class labeling | ✅ Both modes |
| Class weights (inverse-frequency) | ✅ |
| RobustScaler | ✅ (plus standard, minmax options) |
| ES cross-asset (correlation, beta, lead-lag) | ✅ 14 features |
| Opening range (5m, 15m) | ✅ with breakout signals |
| Session blocks + cyclical time | ✅ |
| HPO (Optuna) | ✅ Not in original blueprint — you went beyond |
| Macro sentiment (VIXY) | ✅ Not in original blueprint — you went beyond |
| Equity context (NVDA/TSLA/XLK/SMH) | ✅ Not in original blueprint — you went beyond |

---

## Recommended Implementation Order

```
PHASE A (immediate — fixes + quick wins):
  1. Fix forward window units (bars→minutes or make intentional)
  2. Add close session (14:00-15:00 CT) to config
  3. Add realized volatility + vol-of-vol features
  4. Add VWAP slope
  5. Add 30-min opening range
  6. Add spread volatility + book pressure velocity

PHASE B (validation infrastructure):
  7. Walk-forward validation
  8. XGBoost baseline
  9. SHAP feature importance
  10. Confidence thresholding + probability calibration

PHASE C (microstructure enrichment):
  11. Trade location + effective spread
  12. Consecutive same-side count + large-trade CVD
  13. Fleeting liquidity detection
  14. Amihud illiquidity
  15. Trade arrival acceleration

PHASE D (advanced):
  16. Adaptive label threshold (ATR-normalized)
  17. Volume profile (POC, Value Area)
  18. Economic calendar flags
  19. Multi-resolution input (1s + 10s)
  20. Multi-horizon ensemble
```

**Phases A+B will have the most impact per hour of work.** The forward window fix alone may dramatically change model performance since you're currently training on an unintended prediction horizon. 

Completed: *Write comprehensive comparison analysis* (4/4)