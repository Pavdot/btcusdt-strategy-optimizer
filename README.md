# BTCUSDT Strategy Optimizer

Research workspace for optimizing and validating BTCUSDT trading strategies with deterministic backtests, walk-forward checks, Monte Carlo stress tests, and reproducible reports.

## What is included

- `my_strategy.py`: reconstructed manual V1.2 continuation strategy compatible with `backtesting.py`.
- `manual_bt_run/optimize_strategy.py`: adapted optimizer based on the provided script.
- `manual_bt_run/validate_best_manual.py`: full-history, walk-forward, and Monte Carlo validation for the best candidate found.
- `strategy_factory_btc_search.py`: broader strategy factory used for EMA, Donchian, RSI/Bollinger, and momentum families.
- `optimize_strategy_factory.py`: multi-round orchestrator for the strategy factory.
- `results/`: curated summaries, walk-forward table, trades table, and equity/walk-forward plot.

Large raw Binance datasets are intentionally excluded from git. Recreate them before running the scripts, or place a compatible CSV at:

```text
manual_bt_run/data/BTCUSDT_1h.csv
```

Expected CSV columns:

```text
Date,Open,High,Low,Close,Volume
```

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run the manual optimizer

```bash
cd manual_bt_run
python optimize_strategy.py --csv data/BTCUSDT_1h.csv --max-tries 40 --random-state 42
```

## Validate the best manual candidate

```bash
cd manual_bt_run
python validate_best_manual.py
```

## Run CPCV anti-overfitting checks

```bash
python manual_bt_run/cpcv_validate_manual.py --mode fixed --csv manual_bt_run/data/BTCUSDT_1h.csv
```

Full conditional pipeline:

```bash
python manual_bt_run/cpcv_validate_manual.py --mode full --max-tries 120 --random-states 42 1337 2026
```

The `full` mode first validates the current best parameters with combinatorial purged cross-validation. It only launches the robust re-optimization pass when the fixed-parameter CPCV gate passes.

## Latest validation snapshot

Best manual V1.2 candidate on BTCUSDT 1h:

- Full-history return: `+50.82%`
- Annualized return: `+22.43%`
- Sharpe: `1.26`
- Sortino: `2.50`
- Max drawdown: `-10.09%`
- Profit factor: `1.77`
- Trades: `103`
- Walk-forward profitable folds: `72.22%`
- Walk-forward average monthly return per fold: `1.99%`
- Monte Carlo probability of loss: `2.99%`

## Latest CPCV snapshot

Strict CPCV settings: `8` chronological groups, `2` test groups per split, `28` splits, `1200` purged bars, `336` embargo bars.

- Current parameters CPCV median monthly return: `+1.79%`
- Current parameters profitable CPCV splits: `92.86%`
- Current parameters median CPCV profit factor: `1.86`
- Current parameters CPCV gate: `PASS`
- Re-optimized candidate CPCV median monthly return: `+1.77%`
- Re-optimized candidate PBO approximation: `0.357`
- Re-optimized candidate CPCV gate: `FAIL` because PBO is above the strict `<0.20` threshold
- Re-optimized walk-forward average monthly return: `+1.26%`
- Re-optimized Monte Carlo probability of loss: `7.38%`

This is a research artifact, not financial advice and not a live-trading recommendation.
