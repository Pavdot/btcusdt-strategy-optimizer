from pathlib import Path

import numpy as np
import pandas as pd
from backtesting.lib import FractionalBacktest

from my_strategy import MyStrategy


INITIAL_CASH = 10_000
COMMISSION = 0.001
BARS_PER_DAY = 24
TRAIN_DAYS = 180
TEST_DAYS = 30
MONTE_CARLO_RUNS = 10_000


BEST_PARAMS = {
    "ema_fast_period": 300,
    "ema_slow_period": 900,
    "rsi_period": 14,
    "atr_period": 14,
    "vp_tactical_lookback": 336,
    "vp_macro_lookback": 750,
    "vp_bins": 24,
    "breakout_lookback": 20,
    "rsi_rank_lookback": 50,
    "ema_slope_lookback": 20,
    "min_body_ratio": 0.8,
    "min_vol_ratio": 0.5,
    "min_break_dist_atr": 0.03,
    "min_node_percentile": 0.35,
    "min_room_to_target_atr": 0.5,
    "min_rsi_percentile_long": 0.35,
    "max_rsi_percentile_short": 0.65,
    "min_close_position_long": 0.5,
    "max_close_position_short": 0.5,
    "stop_atr_buffer": 0.3,
    "tp_fraction_to_next_level": 1.0,
}


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["Date"], index_col="Date").sort_index()


def run_bt(df: pd.DataFrame):
    bt = FractionalBacktest(
        df,
        MyStrategy,
        cash=INITIAL_CASH,
        commission=COMMISSION,
        trade_on_close=False,
        exclusive_orders=True,
    )
    return bt.run(**BEST_PARAMS)


def metric_line(stats: pd.Series) -> dict:
    return {
        "return_pct": float(stats["Return [%]"]),
        "return_ann_pct": float(stats["Return (Ann.) [%]"]),
        "sharpe": float(stats["Sharpe Ratio"]),
        "sortino": float(stats["Sortino Ratio"]),
        "max_dd_pct": float(stats["Max. Drawdown [%]"]),
        "profit_factor": float(stats["Profit Factor"]),
        "trades": int(stats["# Trades"]),
        "win_rate_pct": float(stats["Win Rate [%]"]),
        "expectancy_pct": float(stats["Expectancy [%]"]),
        "sqn": float(stats["SQN"]),
    }


def equity_metrics(equity: pd.Series) -> dict:
    equity = equity.dropna()
    ret = equity.iloc[-1] / equity.iloc[0] - 1
    days = max((equity.index[-1] - equity.index[0]).total_seconds() / 86400, 1e-9)
    monthly = (1 + ret) ** (30.4375 / days) - 1 if ret > -1 else -1
    running_max = equity.cummax()
    max_dd = (equity / running_max - 1).min()
    returns = equity.pct_change().dropna()
    ann_return = (1 + ret) ** (365.25 / days) - 1 if ret > -1 else -1
    ann_vol = returns.std() * np.sqrt(365.25 * 24)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    return {
        "return_pct": ret * 100,
        "monthly_return_pct": monthly * 100,
        "max_dd_pct": max_dd * 100,
        "sharpe": sharpe,
    }


def walk_forward(df: pd.DataFrame):
    train_bars = TRAIN_DAYS * BARS_PER_DAY
    test_bars = TEST_DAYS * BARS_PER_DAY
    rows = []
    all_trade_returns = []

    fold = 1
    start = 0
    while start + train_bars + test_bars <= len(df):
        sample = df.iloc[start : start + train_bars + test_bars]
        test_start = sample.index[train_bars]
        stats = run_bt(sample)
        equity = stats["_equity_curve"]["Equity"]
        test_equity = equity[equity.index >= test_start]
        metrics = equity_metrics(test_equity)
        trades = stats["_trades"]
        if not trades.empty:
            exits = pd.to_datetime(trades["ExitTime"])
            test_trades = trades[exits >= test_start]
            if "ReturnPct" in test_trades.columns:
                all_trade_returns.extend(test_trades["ReturnPct"].astype(float).to_list())
        rows.append(
            {
                "fold": fold,
                "test_start": test_equity.index[0],
                "test_end": test_equity.index[-1],
                **metrics,
                "test_trades": 0 if trades.empty else int((pd.to_datetime(trades["ExitTime"]) >= test_start).sum()),
            }
        )
        start += test_bars
        fold += 1

    folds = pd.DataFrame(rows)
    return folds, pd.Series(all_trade_returns, dtype=float)


def monte_carlo(trade_returns: pd.Series):
    trade_returns = trade_returns.dropna().to_numpy(dtype=float)
    if len(trade_returns) == 0:
        return pd.DataFrame(), {}
    rng = np.random.default_rng(42)
    rows = []
    for i in range(MONTE_CARLO_RUNS):
        sampled = rng.choice(trade_returns, size=len(trade_returns), replace=True)
        equity = np.insert(np.cumprod(1 + sampled), 0, 1.0)
        dd = equity / np.maximum.accumulate(equity) - 1
        rows.append(
            {
                "run": i + 1,
                "final_return_pct": (equity[-1] - 1) * 100,
                "max_dd_pct": dd.min() * 100,
            }
        )
    mc = pd.DataFrame(rows)
    summary = {
        "mc_runs": MONTE_CARLO_RUNS,
        "mc_trades": len(trade_returns),
        "mc_final_return_p05": mc["final_return_pct"].quantile(0.05),
        "mc_final_return_p50": mc["final_return_pct"].quantile(0.50),
        "mc_final_return_p95": mc["final_return_pct"].quantile(0.95),
        "mc_probability_loss_pct": (mc["final_return_pct"] < 0).mean() * 100,
        "mc_max_dd_p05": mc["max_dd_pct"].quantile(0.05),
        "mc_max_dd_p50": mc["max_dd_pct"].quantile(0.50),
    }
    return mc, summary


def main():
    df = load_data(Path("data/BTCUSDT_1h.csv"))
    full_stats = run_bt(df)
    full_summary = metric_line(full_stats)
    folds, trade_returns = walk_forward(df)
    mc, mc_summary = monte_carlo(trade_returns)

    folds.to_csv("manual_best_walkforward.csv", index=False)
    mc.to_csv("manual_best_monte_carlo.csv", index=False)
    full_stats["_equity_curve"].to_csv("manual_best_equity_curve.csv")
    full_stats["_trades"].to_csv("manual_best_trades.csv", index=False)

    wf_summary = {
        "folds": len(folds),
        "wf_total_return_pct_sum": folds["return_pct"].sum(),
        "wf_median_fold_return_pct": folds["return_pct"].median(),
        "wf_best_fold_return_pct": folds["return_pct"].max(),
        "wf_worst_fold_return_pct": folds["return_pct"].min(),
        "wf_profitable_folds_pct": (folds["return_pct"] > 0).mean() * 100,
        "wf_avg_monthly_return_pct": folds["monthly_return_pct"].mean(),
        "wf_avg_max_dd_pct": folds["max_dd_pct"].mean(),
        "wf_total_test_trades": int(folds["test_trades"].sum()),
    }

    lines = ["Manual V1.2 validation", "", "Full history"]
    for key, value in full_summary.items():
        lines.append(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    lines.append("")
    lines.append("Walk-forward")
    for key, value in wf_summary.items():
        lines.append(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    lines.append("")
    lines.append("Monte Carlo")
    for key, value in mc_summary.items():
        lines.append(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

    text = "\n".join(lines) + "\n"
    Path("manual_best_validation_summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
