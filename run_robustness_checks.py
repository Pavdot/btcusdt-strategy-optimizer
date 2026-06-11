import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path("btcusdt_1h_klines_long.json")
FOLDS_PATH = Path("walkforward_folds_btcusdt_1h.csv")
TRADES_PATH = Path("walkforward_oos_trades_btcusdt_1h.csv")
MC_PATH = Path("monte_carlo_btcusdt_1h.csv")
SUMMARY_PATH = Path("robustness_summary_btcusdt_1h.txt")

SPREAD = 0.0002
FEES_PCENT = 0.00075
COST_PER_ORDER = SPREAD + FEES_PCENT

SHORT_WINDOWS = [10, 20, 30]
LONG_WINDOWS = [50, 100, 200]
SL_PCENTS = [0.02, 0.03]
TP_PCENTS = [0.03, 0.05, 0.08]

TRAIN_BARS = 24 * 120
TEST_BARS = 24 * 30
STEP_BARS = TEST_BARS
MIN_TRAIN_TRADES = 4
MONTE_CARLO_RUNS = 10_000
RANDOM_SEED = 42


@dataclass(frozen=True)
class Params:
    short_window: int
    long_window: int
    sl_pcent: float
    tp_pcent: float


def normalize_kline_row(row):
    if isinstance(row, dict) and "value" in row:
        return row["value"]
    return row


def load_binance_klines(path: Path) -> pd.DataFrame:
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


def generate_trend_signals(df: pd.DataFrame, params: Params) -> pd.Series:
    short_ma = df["close"].rolling(
        params.short_window, min_periods=params.short_window
    ).mean()
    long_ma = df["close"].rolling(
        params.long_window, min_periods=params.long_window
    ).mean()
    in_uptrend = short_ma > long_ma
    was_uptrend = in_uptrend.shift(1, fill_value=False).astype(bool)

    raw_signal = pd.Series(0, index=df.index, dtype=int)
    raw_signal.loc[in_uptrend & ~was_uptrend] = 1
    raw_signal.loc[~in_uptrend & was_uptrend] = -1

    # Execute next bar to avoid using a close that was not known at entry time.
    return raw_signal.shift(1).fillna(0).astype(int)


def run_backtest(df: pd.DataFrame, params: Params) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = df.index
    open_arr = df["open"].to_numpy(dtype=float)
    high_arr = df["high"].to_numpy(dtype=float)
    low_arr = df["low"].to_numpy(dtype=float)
    close_arr = df["close"].to_numpy(dtype=float)
    signal_arr = generate_trend_signals(df, params).to_numpy(dtype=int)

    n = len(df)
    position_arr = np.zeros(n, dtype=int)
    return_arr = np.zeros(n, dtype=float)
    cost_arr = np.zeros(n, dtype=float)

    trades = []
    position = 0
    entry_i = None
    entry_price = np.nan

    for i in range(1, n):
        signal = signal_arr[i]
        current_open = open_arr[i]
        current_high = high_arr[i]
        current_low = low_arr[i]
        current_close = close_arr[i]
        prev_close = close_arr[i - 1]

        cost = 0.0
        bar_return = 0.0
        exited_this_bar = False

        if position == 1:
            sl_level = entry_price * (1 - params.sl_pcent)
            tp_level = entry_price * (1 + params.tp_pcent)

            if current_low <= sl_level:
                exit_price = sl_level
                bar_return = (exit_price - prev_close) / prev_close
                cost += COST_PER_ORDER
                gross = (exit_price - entry_price) / entry_price
                trades.append(
                    {
                        "entry_time": index[entry_i],
                        "exit_time": index[i],
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "gross_return": gross,
                        "net_return": gross - 2 * COST_PER_ORDER,
                        "exit_reason": "stop_loss",
                    }
                )
                position = 0
                entry_i = None
                entry_price = np.nan
                exited_this_bar = True
            elif current_high >= tp_level:
                exit_price = tp_level
                bar_return = (exit_price - prev_close) / prev_close
                cost += COST_PER_ORDER
                gross = (exit_price - entry_price) / entry_price
                trades.append(
                    {
                        "entry_time": index[entry_i],
                        "exit_time": index[i],
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "gross_return": gross,
                        "net_return": gross - 2 * COST_PER_ORDER,
                        "exit_reason": "take_profit",
                    }
                )
                position = 0
                entry_i = None
                entry_price = np.nan
                exited_this_bar = True
            elif signal == -1:
                exit_price = current_open
                bar_return = (exit_price - prev_close) / prev_close
                cost += COST_PER_ORDER
                gross = (exit_price - entry_price) / entry_price
                trades.append(
                    {
                        "entry_time": index[entry_i],
                        "exit_time": index[i],
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "gross_return": gross,
                        "net_return": gross - 2 * COST_PER_ORDER,
                        "exit_reason": "trend_exit",
                    }
                )
                position = 0
                entry_i = None
                entry_price = np.nan
                exited_this_bar = True
            else:
                bar_return = (current_close - prev_close) / prev_close

        if position == 0 and signal == 1 and not exited_this_bar:
            position = 1
            entry_i = i
            entry_price = current_open
            cost += COST_PER_ORDER

            sl_level = entry_price * (1 - params.sl_pcent)
            tp_level = entry_price * (1 + params.tp_pcent)
            if current_low <= sl_level:
                exit_price = sl_level
                bar_return = (exit_price - entry_price) / entry_price
                cost += COST_PER_ORDER
                gross = (exit_price - entry_price) / entry_price
                trades.append(
                    {
                        "entry_time": index[entry_i],
                        "exit_time": index[i],
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "gross_return": gross,
                        "net_return": gross - 2 * COST_PER_ORDER,
                        "exit_reason": "stop_loss",
                    }
                )
                position = 0
                entry_i = None
                entry_price = np.nan
            elif current_high >= tp_level:
                exit_price = tp_level
                bar_return = (exit_price - entry_price) / entry_price
                cost += COST_PER_ORDER
                gross = (exit_price - entry_price) / entry_price
                trades.append(
                    {
                        "entry_time": index[entry_i],
                        "exit_time": index[i],
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "gross_return": gross,
                        "net_return": gross - 2 * COST_PER_ORDER,
                        "exit_reason": "take_profit",
                    }
                )
                position = 0
                entry_i = None
                entry_price = np.nan
            else:
                bar_return = (current_close - entry_price) / entry_price

        position_arr[i] = position
        return_arr[i] = bar_return
        cost_arr[i] = cost

    out = pd.DataFrame(
        {
            "close": close_arr,
            "signal": signal_arr,
            "position": position_arr,
            "strategy_return": return_arr,
            "cost": cost_arr,
        },
        index=index,
    )
    out["strategy_return_net"] = out["strategy_return"] - out["cost"]
    out["equity_curve_net"] = (1 + out["strategy_return_net"]).cumprod()
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df["short_window"] = params.short_window
        trades_df["long_window"] = params.long_window
        trades_df["sl_pcent"] = params.sl_pcent
        trades_df["tp_pcent"] = params.tp_pcent
    return out, trades_df


def calculate_metrics(equity: pd.Series) -> dict:
    equity = equity.dropna()
    if len(equity) < 2:
        return {
            "total_return_pcent": 0.0,
            "annualized_return_pcent": 0.0,
            "annualized_volatility_pcent": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_pcent": 0.0,
        }

    returns = equity.pct_change().dropna()
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    hours = (equity.index[-1] - equity.index[0]).total_seconds() / 3600
    ann_factor = (24 * 365.25) / hours if hours > 0 else 0
    annual_return = (1 + total_return) ** ann_factor - 1 if hours > 0 else 0
    annual_vol = returns.std() * np.sqrt(24 * 365.25)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0.0
    drawdown = equity / equity.cummax() - 1

    return {
        "total_return_pcent": total_return * 100,
        "annualized_return_pcent": annual_return * 100,
        "annualized_volatility_pcent": annual_vol * 100,
        "sharpe_ratio": sharpe,
        "max_drawdown_pcent": drawdown.min() * 100,
    }


def score_training_run(result: pd.DataFrame, trades: pd.DataFrame) -> float:
    if trades.empty or len(trades) < MIN_TRAIN_TRADES:
        return -np.inf

    metrics = calculate_metrics(result["equity_curve_net"])
    if metrics["max_drawdown_pcent"] < -35:
        return -np.inf

    return metrics["sharpe_ratio"]


def parameter_grid() -> list[Params]:
    grid = []
    for short_window in SHORT_WINDOWS:
        for long_window in LONG_WINDOWS:
            if long_window <= short_window:
                continue
            for sl_pcent in SL_PCENTS:
                for tp_pcent in TP_PCENTS:
                    grid.append(Params(short_window, long_window, sl_pcent, tp_pcent))
    return grid


def run_walk_forward(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grid = parameter_grid()
    folds = []
    oos_returns = []
    oos_trades = []

    start = 0
    fold_id = 1
    while start + TRAIN_BARS + TEST_BARS <= len(df):
        train = df.iloc[start : start + TRAIN_BARS]
        test = df.iloc[start + TRAIN_BARS : start + TRAIN_BARS + TEST_BARS]
        print(
            f"fold={fold_id} train={train.index[0]}->{train.index[-1]} "
            f"test={test.index[0]}->{test.index[-1]}",
            flush=True,
        )

        best_params = None
        best_score = -np.inf
        best_train_metrics = None

        for params in grid:
            train_result, train_trades = run_backtest(train, params)
            score = score_training_run(train_result, train_trades)
            if score > best_score:
                best_score = score
                best_params = params
                best_train_metrics = calculate_metrics(train_result["equity_curve_net"])

        if best_params is None:
            start += STEP_BARS
            fold_id += 1
            continue

        test_result, test_trades = run_backtest(test, best_params)
        test_metrics = calculate_metrics(test_result["equity_curve_net"])
        if not test_trades.empty:
            test_trades = test_trades.copy()
            test_trades["fold"] = fold_id
            oos_trades.append(test_trades)

        ret = test_result["strategy_return_net"].copy()
        ret.name = "strategy_return_net"
        oos_returns.append(ret)

        folds.append(
            {
                "fold": fold_id,
                "train_start": train.index[0],
                "train_end": train.index[-1],
                "test_start": test.index[0],
                "test_end": test.index[-1],
                "short_window": best_params.short_window,
                "long_window": best_params.long_window,
                "sl_pcent": best_params.sl_pcent,
                "tp_pcent": best_params.tp_pcent,
                "train_score_sharpe": best_score,
                "train_return_pcent": best_train_metrics["total_return_pcent"],
                "train_max_dd_pcent": best_train_metrics["max_drawdown_pcent"],
                "test_return_pcent": test_metrics["total_return_pcent"],
                "test_sharpe": test_metrics["sharpe_ratio"],
                "test_max_dd_pcent": test_metrics["max_drawdown_pcent"],
                "test_closed_trades": 0 if test_trades.empty else len(test_trades),
                "test_win_rate_pcent": 0.0
                if test_trades.empty
                else (test_trades["net_return"] > 0).mean() * 100,
            }
        )

        start += STEP_BARS
        fold_id += 1

    folds_df = pd.DataFrame(folds)
    trades_df = pd.concat(oos_trades, ignore_index=True) if oos_trades else pd.DataFrame()
    returns_df = pd.concat(oos_returns).to_frame() if oos_returns else pd.DataFrame()
    return folds_df, trades_df, returns_df


def run_monte_carlo(trade_returns: pd.Series) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(RANDOM_SEED)
    returns = trade_returns.dropna().to_numpy(dtype=float)
    if len(returns) == 0:
        return pd.DataFrame(), {}

    rows = []
    sample_size = len(returns)
    for run_id in range(MONTE_CARLO_RUNS):
        sampled = rng.choice(returns, size=sample_size, replace=True)
        equity = np.cumprod(1 + sampled)
        equity = np.insert(equity, 0, 1.0)
        drawdown = equity / np.maximum.accumulate(equity) - 1
        rows.append(
            {
                "run": run_id + 1,
                "final_return_pcent": (equity[-1] - 1) * 100,
                "max_drawdown_pcent": drawdown.min() * 100,
            }
        )

    mc = pd.DataFrame(rows)
    summary = {
        "mc_runs": MONTE_CARLO_RUNS,
        "mc_trade_sample_size": sample_size,
        "mc_final_return_p05": mc["final_return_pcent"].quantile(0.05),
        "mc_final_return_p50": mc["final_return_pcent"].quantile(0.50),
        "mc_final_return_p95": mc["final_return_pcent"].quantile(0.95),
        "mc_probability_loss_pcent": (mc["final_return_pcent"] < 0).mean() * 100,
        "mc_max_dd_p50": mc["max_drawdown_pcent"].quantile(0.50),
        "mc_max_dd_p05": mc["max_drawdown_pcent"].quantile(0.05),
    }
    return mc, summary


def format_dict(title: str, values: dict) -> list[str]:
    lines = [title]
    for key, value in values.items():
        if isinstance(value, float):
            lines.append(f"{key}: {value:.4f}")
        else:
            lines.append(f"{key}: {value}")
    return lines


def main() -> None:
    df = load_binance_klines(DATA_PATH)
    folds_df, trades_df, returns_df = run_walk_forward(df)

    if returns_df.empty:
        raise RuntimeError("No walk-forward returns were generated.")

    oos_equity = (1 + returns_df["strategy_return_net"]).cumprod()
    oos_metrics = calculate_metrics(oos_equity)

    fold_summary = {
        "data_rows": len(df),
        "data_start": df.index[0].isoformat(),
        "data_end": df.index[-1].isoformat(),
        "folds": len(folds_df),
        "train_days": TRAIN_BARS / 24,
        "test_days": TEST_BARS / 24,
        "profitable_folds_pcent": (folds_df["test_return_pcent"] > 0).mean() * 100,
        "median_fold_return_pcent": folds_df["test_return_pcent"].median(),
        "worst_fold_return_pcent": folds_df["test_return_pcent"].min(),
        "best_fold_return_pcent": folds_df["test_return_pcent"].max(),
        "oos_closed_trades": 0 if trades_df.empty else len(trades_df),
        "oos_win_rate_pcent": 0.0
        if trades_df.empty
        else (trades_df["net_return"] > 0).mean() * 100,
    }

    mc_df, mc_summary = run_monte_carlo(
        trades_df["net_return"] if not trades_df.empty else pd.Series(dtype=float)
    )

    folds_df.to_csv(FOLDS_PATH, index=False)
    trades_df.to_csv(TRADES_PATH, index=False)
    mc_df.to_csv(MC_PATH, index=False)

    param_counts = (
        folds_df[["short_window", "long_window", "sl_pcent", "tp_pcent"]]
        .value_counts()
        .reset_index(name="count")
        .head(10)
    )

    lines = []
    lines.extend(format_dict("Walk-forward setup", fold_summary))
    lines.append("")
    lines.extend(format_dict("Out-of-sample equity metrics", oos_metrics))
    lines.append("")
    lines.extend(format_dict("Monte Carlo summary", mc_summary))
    lines.append("")
    lines.append("Selected parameter counts")
    lines.append(param_counts.to_string(index=False))

    text = "\n".join(lines) + "\n"
    SUMMARY_PATH.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
