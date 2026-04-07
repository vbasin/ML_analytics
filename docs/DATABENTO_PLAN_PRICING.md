# Databento Plan & Pricing Reference

> **Last updated**: 2026-04-06

## Subscription

- **Dataset**: GLBX.MDP3 — **Standard plan** ($179/month, renews 2026-05-01)
- **Live**: CME personal license included (up to 2 devices)

## Included Data Windows

| Level | Schemas | Included History |
|:-----:|---------|:----------------:|
| L0 | OHLCV-*, Definition, Statistics, Status | **15+ years** |
| L1 | MBP-1, TBBO, BBO-1s/1m, Trades | **12 months (rolling)** |
| L2 | MBP-10 | 1 month |
| L3 | MBO, Imbalance | 1 month |

Beyond included window → pay-as-you-go.

## `get_cost()` API Behavior

**`get_cost()` DOES respect subscription inclusions.** It returns `$0.00` for
data within the rolling included window. If it returns non-zero, the requested
date range extends beyond the window.

The L1 window is a **rolling 12 months from today**. Example:
- On 2026-04-02: `start=2025-04-01` → `$0.00` (within window)
- On 2026-04-06: `start=2025-04-01` → non-zero (5 days outside window)
- On 2026-04-06: `start=2025-04-06` → `$0.00` (within window)

The cost check script uses dynamic dates (`today - 365d` → `today`) to stay
within the window.

## Caveats

1. **Rolling window**: L1 start date must stay within 12 months of today or
   `get_cost()` will show charges for the out-of-window portion.
2. **NQ.OPT batch job**: `GLBX-20260402-S4EMNFKM8K` (bbo-1s, 56 GB compressed)
   expires **2026-05-02** — download before then.
3. **Batch downloads**: Billed once at submission; unlimited re-downloads for 30 days.
   Recommended for data >5 GB.
