import json
import math
import os
import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

MPL_CONFIG_DIR = Path("matplotlib_cache")
MPL_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR.resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INTERVAL = "15m"
KLINES_PATH = Path("btcusdt_15m_klines_long.json")
FUNDING_PATH = Path("btcusdt_funding_long.json")

SEARCH_RESULTS_PATH = Path("strategy_search_results_btcusdt_15m.csv")
TOP_EVAL_PATH = Path("strategy_top_eval_btcusdt_15m.csv")
WF_PATH = Path("strategy_walkforward_btcusdt_15m.csv")
MC_PATH = Path("strategy_monte_carlo_btcusdt_15m.csv")
SUMMARY_PATH = Path("strategy_factory_summary_btcusdt_15m.txt")
PLOT_PATH = Path("strategy_factory_top_equity_15m.png")

BARS_PER_DAY = 24 * 4
BARS_PER_YEAR = 365.25 * BARS_PER_DAY
COST_PER_SIDE = 0.00055
TARGET_MONTHLY_RETURN = 0.20
RANDOM_SEED = 20260611

DISCOVERY_DAYS = 365
HOLDOUT_DAYS = 180
WALKFORWARD_TRAIN_DAYS = 180
WALKFORWARD_TEST_DAYS = 30
MONTE_CARLO_RUNS = 10_000


def configure(interval: str) -> None:
    global INTERVAL
    global KLINES_PATH
    global SEARCH_RESULTS_PATH
    global TOP_EVAL_PATH
    global WF_PATH
    global MC_PATH
    global SUMMARY_PATH
    global PLOT_PATH
    global BARS_PER_DAY
    global BARS_PER_YEAR

    supported = {"15m": 24 * 4, "5m": 24 * 12}
    if interval not in supported:
        raise ValueError(f"Unsupported interval {interval}. Use one of: {', '.join(supported)}")

    INTERVAL = interval
    BARS_PER_DAY = supported[interval]
    BARS_PER_YEAR = 365.25 * BARS_PER_DAY
    KLINES_PATH = Path(f"btcusdt_{interval}_klines_long.json")
    SEARCH_RESULTS_PATH = Path(f"strategy_search_results_btcusdt_{interval}.csv")
    TOP_EVAL_PATH = Path(f"strategy_top_eval_btcusdt_{interval}.csv")
    WF_PATH = Path(f"strategy_walkforward_btcusdt_{interval}.csv")
    MC_PATH = Path(f"strategy_monte_carlo_btcusdt_{interval}.csv")
    SUMMARY_PATH = Path(f"strategy_factory_summary_btcusdt_{interval}.txt")
    PLOT_PATH = Path(f"strategy_factory_top_equity_{interval}.png")


@dataclass(frozen=True)
class StrategyParams:
    strategy_id: str
    template: str
    fast: int
    slow: int
    lookback: int
    rsi_window: int
    rsi_low: int
    rsi_high: int
    z_window: int
    z_entry: float
    momentum_lookback: int
    momentum_threshold: float
    trend_window: int
    sl_pcent: float
    tp_pcent: float
    max_hold_bars: int
    leverage: float
    funding_filter: str
    funding_threshold: float


def normalize_kline_row(row):
    if isinstance(row, dict) and "value" in row:
        return row["value"]
    return row


def load_klines(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8-sig") as f:
        raw = json.load(f)
    rows = [normalize_kline_row(row) for row in raw]
    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]
    df = pd.DataFrame(rows, columns=columns)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["open", "high", "low", "close", "volume"]].dropna()


def load_funding(path: Path, index: pd.DatetimeIndex) -> pd.Series:
    if not path.exists():
        return pd.Series(0.0, index=index, name="funding_rate")

    with path.open("r", encoding="utf-8-sig") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw = [raw]

    rows = []
    for row in raw:
        if "fundingTime" not in row or "fundingRate" not in row:
            continue
        rows.append(
            {
                "timestamp": pd.to_datetime(int(row["fundingTime"]), unit="ms", utc=True),
                "funding_rate": float(row["fundingRate"]),
            }
        )

    if not rows:
        return pd.Series(0.0, index=index, name="funding_rate")

    funding = pd.DataFrame(rows).drop_duplicates("timestamp").set_index("timestamp")
    funding = funding.sort_index()["funding_rate"]
    return funding.reindex(index, method="ffill").fillna(0.0).rename("funding_rate")


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_gain = up.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = down.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_feature_cache(df: pd.DataFrame) -> dict:
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    windows = sorted({5, 8, 10, 14, 20, 24, 30, 40, 50, 60, 80, 100, 150, 200})
    cache = {
        "ema": {w: ema(close, w).to_numpy(dtype=float) for w in windows},
        "rsi": {w: rsi(close, w).to_numpy(dtype=float) for w in [7, 14, 21]},
        "roll_high": {
            w: high.shift(1).rolling(w, min_periods=w).max().to_numpy(dtype=float)
            for w in windows
        },
        "roll_low": {
            w: low.shift(1).rolling(w, min_periods=w).min().to_numpy(dtype=float)
            for w in windows
        },
        "z": {},
        "mom": {},
        "vol_mean": {},
    }

    for w in [20, 40, 60, 100]:
        mean = close.rolling(w, min_periods=w).mean()
        std = close.rolling(w, min_periods=w).std()
        cache["z"][w] = ((close - mean) / std.replace(0, np.nan)).to_numpy(dtype=float)

    for w in [8, 12, 20, 40, 60]:
        cache["mom"][w] = close.pct_change(w).to_numpy(dtype=float)

    for w in [20, 50, 100]:
        cache["vol_mean"][w] = volume.rolling(w, min_periods=w).mean().to_numpy(dtype=float)

    return cache


def apply_funding_filter(signal: np.ndarray, funding: np.ndarray, params: StrategyParams) -> np.ndarray:
    if params.funding_filter == "none":
        return signal

    filtered = signal.copy()
    threshold = params.funding_threshold
    if params.funding_filter == "contrarian":
        long_ok = funding <= threshold
        short_ok = funding >= -threshold
    elif params.funding_filter == "avoid_extreme":
        long_ok = np.abs(funding) <= threshold
        short_ok = long_ok
    else:
        return filtered

    filtered[(filtered == 1) & ~long_ok] = 0
    filtered[(filtered == -1) & ~short_ok] = 0
    return filtered


def build_signal(df: pd.DataFrame, cache: dict, funding: np.ndarray, params: StrategyParams) -> np.ndarray:
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    signal = np.zeros(len(df), dtype=np.int8)

    if params.template == "ema_cross":
        fast = cache["ema"][params.fast]
        slow = cache["ema"][params.slow]
        trend = np.where(fast > slow, 1, -1)
        trend[np.isnan(fast) | np.isnan(slow)] = 0
        prev = np.roll(trend, 1)
        prev[0] = 0
        signal[(trend == 1) & (prev <= 0)] = 1
        signal[(trend == -1) & (prev >= 0)] = -1

    elif params.template == "donchian_breakout":
        roll_high = cache["roll_high"][params.lookback]
        roll_low = cache["roll_low"][params.lookback]
        if params.trend_window > 0:
            trend = cache["ema"][params.trend_window]
            long_ok = close > trend
            short_ok = close < trend
        else:
            long_ok = np.ones(len(df), dtype=bool)
            short_ok = long_ok
        signal[(close > roll_high) & long_ok] = 1
        signal[(close < roll_low) & short_ok] = -1

    elif params.template == "rsi_reversion":
        values = cache["rsi"][params.rsi_window]
        if params.trend_window > 0:
            trend = cache["ema"][params.trend_window]
            long_ok = close >= trend
            short_ok = close <= trend
        else:
            long_ok = np.ones(len(df), dtype=bool)
            short_ok = long_ok
        signal[(values <= params.rsi_low) & long_ok] = 1
        signal[(values >= params.rsi_high) & short_ok] = -1

    elif params.template == "bollinger_reversion":
        z = cache["z"][params.z_window]
        signal[z <= -params.z_entry] = 1
        signal[z >= params.z_entry] = -1

    elif params.template == "momentum_breakout":
        mom = cache["mom"][params.momentum_lookback]
        if params.trend_window > 0:
            trend = cache["ema"][params.trend_window]
            long_ok = close >= trend
            short_ok = close <= trend
        else:
            long_ok = np.ones(len(df), dtype=bool)
            short_ok = long_ok
        volume_ok = df["volume"].to_numpy(dtype=float) >= cache["vol_mean"][50]
        signal[(mom >= params.momentum_threshold) & long_ok & volume_ok] = 1
        signal[(mom <= -params.momentum_threshold) & short_ok & volume_ok] = -1

    signal = apply_funding_filter(signal, funding, params)
    shifted = np.roll(signal, 1)
    shifted[0] = 0
    shifted[np.isnan(close)] = 0
    return shifted.astype(np.int8)


def build_exit_signals(df: pd.DataFrame, cache: dict, params: StrategyParams) -> tuple[np.ndarray, np.ndarray]:
    exit_long = np.zeros(len(df), dtype=bool)
    exit_short = np.zeros(len(df), dtype=bool)

    if params.template == "rsi_reversion":
        values = cache["rsi"][params.rsi_window]
        exit_long = values >= 50
        exit_short = values <= 50
    elif params.template == "bollinger_reversion":
        z = cache["z"][params.z_window]
        exit_long = z >= 0
        exit_short = z <= 0
    elif params.template == "momentum_breakout":
        mom = cache["mom"][params.momentum_lookback]
        exit_long = mom <= 0
        exit_short = mom >= 0

    exit_long = np.roll(exit_long, 1)
    exit_short = np.roll(exit_short, 1)
    exit_long[0] = False
    exit_short[0] = False
    return exit_long.astype(bool), exit_short.astype(bool)


def run_backtest(df: pd.DataFrame, cache: dict, funding: np.ndarray, params: StrategyParams):
    open_arr = df["open"].to_numpy(dtype=float)
    high_arr = df["high"].to_numpy(dtype=float)
    low_arr = df["low"].to_numpy(dtype=float)
    close_arr = df["close"].to_numpy(dtype=float)
    signal_arr = build_signal(df, cache, funding, params)
    exit_long_arr, exit_short_arr = build_exit_signals(df, cache, params)
    n = len(df)

    returns = np.zeros(n, dtype=float)
    costs = np.zeros(n, dtype=float)
    positions = np.zeros(n, dtype=np.int8)
    trades = []

    position = 0
    entry_i = -1
    entry_price = np.nan

    for i in range(1, n):
        signal = int(signal_arr[i])
        current_open = open_arr[i]
        current_high = high_arr[i]
        current_low = low_arr[i]
        current_close = close_arr[i]
        prev_close = close_arr[i - 1]
        bar_return = 0.0
        cost = 0.0
        exited = False

        if position != 0:
            hold_bars = i - entry_i
            if position == 1:
                sl = entry_price * (1 - params.sl_pcent)
                tp = entry_price * (1 + params.tp_pcent)
                stop_hit = current_low <= sl
                target_hit = current_high >= tp
                exit_price = None
                reason = None
                if stop_hit:
                    exit_price = sl
                    reason = "stop_loss"
                elif target_hit:
                    exit_price = tp
                    reason = "take_profit"
            else:
                sl = entry_price * (1 + params.sl_pcent)
                tp = entry_price * (1 - params.tp_pcent)
                stop_hit = current_high >= sl
                target_hit = current_low <= tp
                exit_price = None
                reason = None
                if stop_hit:
                    exit_price = sl
                    reason = "stop_loss"
                elif target_hit:
                    exit_price = tp
                    reason = "take_profit"

            if exit_price is None and (
                (position == 1 and exit_long_arr[i])
                or (position == -1 and exit_short_arr[i])
            ):
                exit_price = current_open
                reason = "neutral_exit"
            if exit_price is None and hold_bars >= params.max_hold_bars:
                exit_price = current_open
                reason = "max_hold"
            if exit_price is None and signal == -position:
                exit_price = current_open
                reason = "opposite_signal"

            if exit_price is not None:
                raw_bar_return = position * (exit_price - prev_close) / prev_close
                gross_trade_return = position * (exit_price - entry_price) / entry_price
                bar_return = raw_bar_return * params.leverage
                cost += COST_PER_SIDE * params.leverage
                trades.append(
                    {
                        "entry_time": df.index[entry_i],
                        "exit_time": df.index[i],
                        "side": position,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "gross_return": gross_trade_return,
                        "net_return": gross_trade_return * params.leverage
                        - 2 * COST_PER_SIDE * params.leverage,
                        "exit_reason": reason,
                        "hold_bars": hold_bars,
                    }
                )
                position = 0
                entry_i = -1
                entry_price = np.nan
                exited = True
            else:
                bar_return = position * (current_close - prev_close) / prev_close * params.leverage

        if position == 0 and signal != 0 and not exited:
            position = signal
            entry_i = i
            entry_price = current_open
            cost += COST_PER_SIDE * params.leverage

            if position == 1:
                sl = entry_price * (1 - params.sl_pcent)
                tp = entry_price * (1 + params.tp_pcent)
                if current_low <= sl:
                    exit_price = sl
                    reason = "stop_loss"
                elif current_high >= tp:
                    exit_price = tp
                    reason = "take_profit"
                else:
                    exit_price = None
                    reason = None
            else:
                sl = entry_price * (1 + params.sl_pcent)
                tp = entry_price * (1 - params.tp_pcent)
                if current_high >= sl:
                    exit_price = sl
                    reason = "stop_loss"
                elif current_low <= tp:
                    exit_price = tp
                    reason = "take_profit"
                else:
                    exit_price = None
                    reason = None

            if exit_price is not None:
                gross_trade_return = position * (exit_price - entry_price) / entry_price
                bar_return = gross_trade_return * params.leverage
                cost += COST_PER_SIDE * params.leverage
                trades.append(
                    {
                        "entry_time": df.index[entry_i],
                        "exit_time": df.index[i],
                        "side": position,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "gross_return": gross_trade_return,
                        "net_return": gross_trade_return * params.leverage
                        - 2 * COST_PER_SIDE * params.leverage,
                        "exit_reason": reason,
                        "hold_bars": 0,
                    }
                )
                position = 0
                entry_i = -1
                entry_price = np.nan
            else:
                bar_return = position * (current_close - entry_price) / entry_price * params.leverage

        returns[i] = bar_return - cost
        costs[i] = cost
        positions[i] = position

    equity = np.cumprod(1 + returns)
    result = pd.DataFrame(
        {
            "signal": signal_arr,
            "position": positions,
            "strategy_return_net": returns,
            "cost": costs,
            "equity_curve": equity,
            "close": close_arr,
        },
        index=df.index,
    )
    trades_df = pd.DataFrame(trades)
    return result, trades_df


def max_drawdown(equity: pd.Series | np.ndarray) -> float:
    values = np.asarray(equity, dtype=float)
    if len(values) == 0:
        return 0.0
    peaks = np.maximum.accumulate(values)
    dd = values / peaks - 1
    return float(np.nanmin(dd))


def metrics_from_result(result: pd.DataFrame, trades: pd.DataFrame) -> dict:
    equity = result["equity_curve"]
    returns = result["strategy_return_net"]
    total_return = float(equity.iloc[-1] - 1)
    days = max((equity.index[-1] - equity.index[0]).total_seconds() / 86400, 1e-9)
    monthly_return = (1 + total_return) ** (30.4375 / days) - 1 if total_return > -1 else -1
    annual_return = (1 + total_return) ** (365.25 / days) - 1 if total_return > -1 else -1
    annual_vol = float(returns.std() * math.sqrt(BARS_PER_YEAR))
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0.0
    negative = returns[returns < 0]
    downside = float(negative.std() * math.sqrt(BARS_PER_YEAR)) if len(negative) > 1 else 0.0
    sortino = annual_return / downside if downside > 0 else 0.0
    dd = max_drawdown(equity)

    if trades.empty:
        pf = 0.0
        win_rate = 0.0
        avg_trade = 0.0
        trade_count = 0
    else:
        gains = trades.loc[trades["net_return"] > 0, "net_return"].sum()
        losses = trades.loc[trades["net_return"] < 0, "net_return"].sum()
        pf = float(gains / abs(losses)) if losses < 0 else float("inf")
        win_rate = float((trades["net_return"] > 0).mean())
        avg_trade = float(trades["net_return"].mean())
        trade_count = int(len(trades))

    return {
        "total_return_pcent": total_return * 100,
        "compound_monthly_return_pcent": monthly_return * 100,
        "annual_return_pcent": annual_return * 100,
        "annual_vol_pcent": annual_vol * 100,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd_pcent": dd * 100,
        "profit_factor": pf,
        "win_rate_pcent": win_rate * 100,
        "avg_trade_pcent": avg_trade * 100,
        "trade_count": trade_count,
        "final_equity": float(equity.iloc[-1]),
    }


def score_metrics(metrics: dict) -> float:
    return (
        metrics["compound_monthly_return_pcent"]
        + 2.0 * metrics["sharpe"]
        + 0.5 * metrics["profit_factor"]
        + 0.25 * metrics["max_dd_pcent"]
    )


def generate_candidates(n: int) -> list[StrategyParams]:
    rng = np.random.default_rng(RANDOM_SEED)
    templates = [
        "ema_cross",
        "donchian_breakout",
        "rsi_reversion",
        "bollinger_reversion",
        "momentum_breakout",
    ]
    fast_values = [5, 8, 10, 14, 20, 30, 40]
    slow_values = [30, 40, 50, 60, 80, 100, 150, 200]
    lookbacks = [20, 30, 40, 50, 60, 80, 100, 150]
    z_windows = [20, 40, 60, 100]
    trend_windows = [0, 50, 100, 150, 200]
    sl_values = [0.006, 0.008, 0.01, 0.015, 0.02, 0.03]
    tp_values = [0.012, 0.018, 0.025, 0.035, 0.05, 0.08]
    max_holds = [16, 32, 48, 96, 192, 384]
    leverages = [1.0, 1.5, 2.0, 3.0]
    funding_filters = ["none", "contrarian", "avoid_extreme"]
    funding_thresholds = [0.00005, 0.0001, 0.0002, 0.0004]

    candidates = []
    for i in range(n):
        template = str(rng.choice(templates))
        fast = int(rng.choice(fast_values))
        slow = int(rng.choice([x for x in slow_values if x > fast]))
        params = StrategyParams(
            strategy_id=f"btc{INTERVAL}_{i+1:04d}",
            template=template,
            fast=fast,
            slow=slow,
            lookback=int(rng.choice(lookbacks)),
            rsi_window=int(rng.choice([7, 14, 21])),
            rsi_low=int(rng.choice([20, 25, 30, 35, 40])),
            rsi_high=int(rng.choice([60, 65, 70, 75, 80])),
            z_window=int(rng.choice(z_windows)),
            z_entry=float(rng.choice([1.0, 1.25, 1.5, 2.0, 2.5])),
            momentum_lookback=int(rng.choice([8, 12, 20, 40, 60])),
            momentum_threshold=float(rng.choice([0.004, 0.006, 0.008, 0.012, 0.016])),
            trend_window=int(rng.choice(trend_windows)),
            sl_pcent=float(rng.choice(sl_values)),
            tp_pcent=float(rng.choice(tp_values)),
            max_hold_bars=int(rng.choice(max_holds)),
            leverage=float(rng.choice(leverages)),
            funding_filter=str(rng.choice(funding_filters)),
            funding_threshold=float(rng.choice(funding_thresholds)),
        )
        candidates.append(params)
    return candidates


def params_from_row(row: pd.Series) -> StrategyParams:
    return StrategyParams(
        strategy_id=str(row["strategy_id"]),
        template=str(row["template"]),
        fast=int(row["fast"]),
        slow=int(row["slow"]),
        lookback=int(row["lookback"]),
        rsi_window=int(row["rsi_window"]),
        rsi_low=int(row["rsi_low"]),
        rsi_high=int(row["rsi_high"]),
        z_window=int(row["z_window"]),
        z_entry=float(row["z_entry"]),
        momentum_lookback=int(row["momentum_lookback"]),
        momentum_threshold=float(row["momentum_threshold"]),
        trend_window=int(row["trend_window"]),
        sl_pcent=float(row["sl_pcent"]),
        tp_pcent=float(row["tp_pcent"]),
        max_hold_bars=int(row["max_hold_bars"]),
        leverage=float(row["leverage"]),
        funding_filter=str(row["funding_filter"]),
        funding_threshold=float(row["funding_threshold"]),
    )


def evaluate_params(df: pd.DataFrame, cache: dict, funding: np.ndarray, params: StrategyParams) -> dict:
    result, trades = run_backtest(df, cache, funding, params)
    metrics = metrics_from_result(result, trades)
    row = asdict(params)
    row.update(metrics)
    row["score"] = score_metrics(metrics)
    return row


def run_search(df: pd.DataFrame, funding_series: pd.Series, candidate_count: int = 360) -> pd.DataFrame:
    discovery = df.iloc[-(DISCOVERY_DAYS + HOLDOUT_DAYS) * BARS_PER_DAY : -HOLDOUT_DAYS * BARS_PER_DAY]
    discovery_funding = funding_series.reindex(discovery.index).fillna(0.0).to_numpy(dtype=float)
    cache = build_feature_cache(discovery)
    candidates = generate_candidates(candidate_count)

    rows = []
    for i, params in enumerate(candidates, start=1):
        rows.append(evaluate_params(discovery, cache, discovery_funding, params))
        if i % 30 == 0:
            best = max(rows, key=lambda x: x["score"])
            print(
                f"searched={i}/{candidate_count} best_monthly="
                f"{best['compound_monthly_return_pcent']:.2f}% "
                f"template={best['template']} id={best['strategy_id']}",
                flush=True,
            )

    results = pd.DataFrame(rows).sort_values("score", ascending=False)
    results.to_csv(SEARCH_RESULTS_PATH, index=False)
    return results


def evaluate_top_candidates(df: pd.DataFrame, funding_series: pd.Series, search_results: pd.DataFrame) -> pd.DataFrame:
    discovery = df.iloc[-(DISCOVERY_DAYS + HOLDOUT_DAYS) * BARS_PER_DAY : -HOLDOUT_DAYS * BARS_PER_DAY]
    holdout = df.iloc[-HOLDOUT_DAYS * BARS_PER_DAY :]
    full_cache = build_feature_cache(df)
    discovery_cache = build_feature_cache(discovery)
    holdout_cache = build_feature_cache(holdout)
    full_funding = funding_series.reindex(df.index).fillna(0.0).to_numpy(dtype=float)
    discovery_funding = funding_series.reindex(discovery.index).fillna(0.0).to_numpy(dtype=float)
    holdout_funding = funding_series.reindex(holdout.index).fillna(0.0).to_numpy(dtype=float)

    rows = []
    top = search_results.head(30)
    for _, row in top.iterrows():
        params = params_from_row(row)
        for label, sample, cache, funding in [
            ("discovery", discovery, discovery_cache, discovery_funding),
            ("holdout", holdout, holdout_cache, holdout_funding),
            ("full", df, full_cache, full_funding),
        ]:
            result, trades = run_backtest(sample, cache, funding, params)
            metrics = metrics_from_result(result, trades)
            out = asdict(params)
            out["sample"] = label
            out.update(metrics)
            rows.append(out)

    eval_df = pd.DataFrame(rows)
    eval_df.to_csv(TOP_EVAL_PATH, index=False)
    return eval_df


def fixed_walk_forward(df: pd.DataFrame, funding_series: pd.Series, params: StrategyParams) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test_bars = WALKFORWARD_TEST_DAYS * BARS_PER_DAY
    train_bars = WALKFORWARD_TRAIN_DAYS * BARS_PER_DAY
    rows = []
    all_returns = []
    all_trades = []
    start = 0
    fold = 1

    while start + train_bars + test_bars <= len(df):
        test = df.iloc[start + train_bars : start + train_bars + test_bars]
        funding = funding_series.reindex(test.index).fillna(0.0).to_numpy(dtype=float)
        cache = build_feature_cache(test)
        result, trades = run_backtest(test, cache, funding, params)
        metrics = metrics_from_result(result, trades)
        row = {
            "fold": fold,
            "test_start": test.index[0],
            "test_end": test.index[-1],
        }
        row.update(metrics)
        rows.append(row)
        returns = result[["strategy_return_net"]].copy()
        returns["fold"] = fold
        all_returns.append(returns)
        if not trades.empty:
            trades = trades.copy()
            trades["fold"] = fold
            all_trades.append(trades)
        start += test_bars
        fold += 1

    folds = pd.DataFrame(rows)
    returns_df = pd.concat(all_returns) if all_returns else pd.DataFrame()
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    folds.to_csv(WF_PATH, index=False)
    return folds, returns_df, trades_df


def monte_carlo(trade_returns: pd.Series) -> tuple[pd.DataFrame, dict]:
    returns = trade_returns.dropna().to_numpy(dtype=float)
    if len(returns) == 0:
        return pd.DataFrame(), {}

    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for i in range(MONTE_CARLO_RUNS):
        sample = rng.choice(returns, size=len(returns), replace=True)
        equity = np.insert(np.cumprod(1 + sample), 0, 1.0)
        dd = max_drawdown(equity)
        rows.append(
            {
                "run": i + 1,
                "final_return_pcent": (equity[-1] - 1) * 100,
                "max_dd_pcent": dd * 100,
            }
        )
    mc = pd.DataFrame(rows)
    mc.to_csv(MC_PATH, index=False)
    summary = {
        "mc_runs": MONTE_CARLO_RUNS,
        "mc_trades": len(returns),
        "mc_final_return_p05": mc["final_return_pcent"].quantile(0.05),
        "mc_final_return_p50": mc["final_return_pcent"].quantile(0.50),
        "mc_final_return_p95": mc["final_return_pcent"].quantile(0.95),
        "mc_probability_loss_pcent": (mc["final_return_pcent"] < 0).mean() * 100,
        "mc_max_dd_p05": mc["max_dd_pcent"].quantile(0.05),
        "mc_max_dd_p50": mc["max_dd_pcent"].quantile(0.50),
    }
    return mc, summary


def passes_gates(holdout_metrics: dict, wf_summary: dict, mc_summary: dict) -> tuple[bool, list[str]]:
    failures = []
    if holdout_metrics["compound_monthly_return_pcent"] < TARGET_MONTHLY_RETURN * 100:
        failures.append("holdout monthly return < 20%")
    if wf_summary["wf_monthly_return_pcent"] < TARGET_MONTHLY_RETURN * 100:
        failures.append("walk-forward monthly return < 20%")
    if wf_summary["wf_profit_factor"] < 1.20:
        failures.append("walk-forward PF < 1.20")
    if wf_summary["wf_sharpe"] < 1.0:
        failures.append("walk-forward Sharpe < 1.0")
    if wf_summary["wf_sortino"] < 1.5:
        failures.append("walk-forward Sortino < 1.5")
    if abs(wf_summary["wf_max_dd_pcent"]) > 12.0:
        failures.append("walk-forward max DD > 12%")
    if wf_summary["wf_profitable_folds_pcent"] < 70.0:
        failures.append("profitable folds < 70%")
    if wf_summary["wf_trade_count"] < 250:
        failures.append("walk-forward trades < 250")
    if mc_summary:
        allowed_mc_dd = abs(wf_summary["wf_max_dd_pcent"]) * 1.4
        if abs(mc_summary["mc_max_dd_p05"]) > allowed_mc_dd:
            failures.append("Monte Carlo 5% DD > 1.4x WF DD")
    return len(failures) == 0, failures


def summarize_walk_forward(returns_df: pd.DataFrame, trades_df: pd.DataFrame, folds: pd.DataFrame) -> dict:
    if returns_df.empty:
        return {}
    equity = (1 + returns_df["strategy_return_net"]).cumprod()
    result = pd.DataFrame({"equity_curve": equity, "strategy_return_net": returns_df["strategy_return_net"]}, index=returns_df.index)
    metrics = metrics_from_result(result, trades_df)
    return {
        "wf_total_return_pcent": metrics["total_return_pcent"],
        "wf_monthly_return_pcent": metrics["compound_monthly_return_pcent"],
        "wf_sharpe": metrics["sharpe"],
        "wf_sortino": metrics["sortino"],
        "wf_max_dd_pcent": metrics["max_dd_pcent"],
        "wf_profit_factor": metrics["profit_factor"],
        "wf_trade_count": metrics["trade_count"],
        "wf_win_rate_pcent": metrics["win_rate_pcent"],
        "wf_profitable_folds_pcent": (folds["total_return_pcent"] > 0).mean() * 100,
        "wf_median_fold_return_pcent": folds["total_return_pcent"].median(),
        "wf_worst_fold_return_pcent": folds["total_return_pcent"].min(),
        "wf_best_fold_return_pcent": folds["total_return_pcent"].max(),
    }


def plot_candidate(df: pd.DataFrame, funding_series: pd.Series, params: StrategyParams) -> None:
    cache = build_feature_cache(df)
    funding = funding_series.reindex(df.index).fillna(0.0).to_numpy(dtype=float)
    result, _ = run_backtest(df, cache, funding, params)
    buy_hold = df["close"] / df["close"].iloc[0]

    plt.figure(figsize=(13, 7))
    plt.plot(result.index, result["equity_curve"], label="Strategy net equity", linewidth=1.5)
    plt.plot(buy_hold.index, buy_hold, label="BTC buy & hold", linewidth=1.1, alpha=0.75)
    plt.axvline(df.index[-HOLDOUT_DAYS * BARS_PER_DAY], color="black", linestyle="--", linewidth=1, label="Holdout start")
    plt.title(f"BTCUSDT {INTERVAL} top candidate: {params.strategy_id} / {params.template}")
    plt.xlabel("Date")
    plt.ylabel("Equity multiple")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()


def main() -> None:
    global RANDOM_SEED

    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", choices=["15m", "5m"], default="15m")
    parser.add_argument("--candidates", type=int, default=360)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    configure(args.interval)
    RANDOM_SEED = args.seed

    df = load_klines(KLINES_PATH)
    funding = load_funding(FUNDING_PATH, df.index)
    print(f"loaded bars={len(df)} start={df.index[0]} end={df.index[-1]}", flush=True)

    search = run_search(df, funding, candidate_count=args.candidates)
    eval_df = evaluate_top_candidates(df, funding, search)

    holdout = eval_df[eval_df["sample"] == "holdout"].copy()
    target_hits = holdout[holdout["compound_monthly_return_pcent"] >= TARGET_MONTHLY_RETURN * 100]
    if target_hits.empty:
        chosen_row = holdout.sort_values("compound_monthly_return_pcent", ascending=False).iloc[0]
    else:
        chosen_row = target_hits.sort_values("score" if "score" in target_hits.columns else "compound_monthly_return_pcent", ascending=False).iloc[0]

    params = params_from_row(chosen_row)
    folds, wf_returns, wf_trades = fixed_walk_forward(df, funding, params)
    wf_summary = summarize_walk_forward(wf_returns, wf_trades, folds)
    mc_df, mc_summary = monte_carlo(wf_trades["net_return"] if not wf_trades.empty else pd.Series(dtype=float))

    holdout_result = holdout[holdout["strategy_id"] == params.strategy_id].iloc[0].to_dict()
    passed, failures = passes_gates(holdout_result, wf_summary, mc_summary)
    plot_candidate(df, funding, params)

    lines = []
    lines.append(f"Strategy factory BTCUSDT {INTERVAL} search")
    lines.append(f"bars: {len(df)}")
    lines.append(f"start: {df.index[0].isoformat()}")
    lines.append(f"end: {df.index[-1].isoformat()}")
    lines.append(f"candidate_count: {len(search)}")
    lines.append(f"target_monthly_return_pcent: {TARGET_MONTHLY_RETURN * 100:.2f}")
    lines.append("")
    lines.append("Chosen candidate")
    for key, value in asdict(params).items():
        lines.append(f"{key}: {value}")
    lines.append("")
    lines.append("Holdout metrics")
    for key in [
        "compound_monthly_return_pcent",
        "total_return_pcent",
        "sharpe",
        "sortino",
        "max_dd_pcent",
        "profit_factor",
        "trade_count",
        "win_rate_pcent",
    ]:
        value = holdout_result.get(key)
        if isinstance(value, float):
            lines.append(f"{key}: {value:.4f}")
        else:
            lines.append(f"{key}: {value}")
    lines.append("")
    lines.append("Walk-forward metrics")
    for key, value in wf_summary.items():
        if isinstance(value, float):
            lines.append(f"{key}: {value:.4f}")
        else:
            lines.append(f"{key}: {value}")
    lines.append("")
    lines.append("Monte Carlo metrics")
    for key, value in mc_summary.items():
        if isinstance(value, float):
            lines.append(f"{key}: {value:.4f}")
        else:
            lines.append(f"{key}: {value}")
    lines.append("")
    lines.append(f"passed_all_gates: {passed}")
    if failures:
        lines.append("failures:")
        lines.extend([f"- {failure}" for failure in failures])

    text = "\n".join(lines) + "\n"
    SUMMARY_PATH.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
