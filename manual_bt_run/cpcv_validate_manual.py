from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from backtesting.lib import FractionalBacktest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from my_strategy import MyStrategy  # noqa: E402


INITIAL_CASH = 10_000
COMMISSION = 0.001
MIN_TRADES = 15
BARS_PER_DAY = 24
TRAIN_DAYS = 180
TEST_DAYS = 30
MONTE_CARLO_RUNS = 10_000

DEFAULT_CSV = Path("manual_bt_run/data/BTCUSDT_1h.csv")
DEFAULT_OUTPUT_DIR = Path("manual_bt_run")

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

PARAM_GRID = {
    "ema_fast_period": [200, 250, 300],
    "ema_slow_period": [600, 750, 900],
    "rsi_period": [14],
    "atr_period": [14],
    "vp_tactical_lookback": [252, 336],
    "vp_macro_lookback": [750, 1000],
    "vp_bins": [16, 24],
    "breakout_lookback": [20],
    "rsi_rank_lookback": [50],
    "ema_slope_lookback": [20],
    "min_body_ratio": [0.4, 0.6, 0.8, 1.0],
    "min_vol_ratio": [0.5, 0.8, 1.0],
    "min_break_dist_atr": [0.00, 0.02, 0.03, 0.05],
    "min_node_percentile": [0.35, 0.50, 0.65],
    "min_room_to_target_atr": [0.5, 0.75, 1.0, 1.25],
    "min_rsi_percentile_long": [0.25, 0.35, 0.45],
    "max_rsi_percentile_short": [0.55, 0.65, 0.75],
    "min_close_position_long": [0.50, 0.55, 0.60],
    "max_close_position_short": [0.40, 0.45, 0.50],
    "stop_atr_buffer": [0.3, 0.5, 0.75],
    "tp_fraction_to_next_level": [0.65, 0.85, 1.0],
}


@dataclass(frozen=True)
class CpcvSplit:
    split_id: int
    test_groups: tuple[int, ...]
    test_ranges: tuple[tuple[int, int], ...]
    train_mask: np.ndarray


def resolve_path(path: Path) -> Path:
    candidates = [
        path,
        REPO_ROOT / path,
        Path(__file__).resolve().parent / path,
        Path(__file__).resolve().parent / "data" / path.name,
        REPO_ROOT.parent / "manual_bt_run" / "data" / path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (REPO_ROOT / path).resolve()


def load_data(csv_path: Path) -> pd.DataFrame:
    resolved = resolve_path(csv_path)
    if not resolved.exists():
        raise FileNotFoundError(f"CSV not found: {resolved}")

    df = pd.read_csv(resolved, parse_dates=["Date"], index_col="Date").sort_index()
    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")
    return df[["Open", "High", "Low", "Close", "Volume"]]


def make_activated_strategy(activation_time):
    if activation_time is None:
        return MyStrategy

    class ActivatedMyStrategy(MyStrategy):
        def next(self):
            if self.data.index[-1] < activation_time:
                return
            super().next()

    ActivatedMyStrategy.__name__ = "ActivatedMyStrategy"
    return ActivatedMyStrategy


def run_bt(df: pd.DataFrame, params: dict, activation_time=None):
    strategy_cls = make_activated_strategy(activation_time)
    bt = FractionalBacktest(
        df,
        strategy_cls,
        cash=INITIAL_CASH,
        commission=COMMISSION,
        trade_on_close=False,
        exclusive_orders=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bt.run(**params)


def build_group_ranges(length: int, n_groups: int) -> list[tuple[int, int]]:
    groups = np.array_split(np.arange(length), n_groups)
    return [(int(group[0]), int(group[-1]) + 1) for group in groups if len(group)]


def contiguous_ranges(mask: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return []
    cuts = np.where(np.diff(indices) > 1)[0] + 1
    chunks = np.split(indices, cuts)
    return [(int(chunk[0]), int(chunk[-1]) + 1) for chunk in chunks if len(chunk)]


def build_cpcv_splits(
    length: int,
    n_groups: int,
    test_groups: int,
    purge_bars: int,
    embargo_bars: int,
) -> tuple[list[tuple[int, int]], list[CpcvSplit]]:
    group_ranges = build_group_ranges(length, n_groups)
    splits = []
    for split_id, combo in enumerate(itertools.combinations(range(len(group_ranges)), test_groups), start=1):
        train_mask = np.ones(length, dtype=bool)
        test_ranges = tuple(group_ranges[group] for group in combo)
        for start, end in test_ranges:
            purge_start = max(0, start - purge_bars)
            embargo_end = min(length, end + embargo_bars)
            train_mask[purge_start:embargo_end] = False
        splits.append(
            CpcvSplit(
                split_id=split_id,
                test_groups=tuple(combo),
                test_ranges=test_ranges,
                train_mask=train_mask,
            )
        )
    return group_ranges, splits


def equity_metrics(equity: pd.Series) -> dict:
    equity = equity.dropna()
    if len(equity) < 2:
        return {
            "return_pct": 0.0,
            "monthly_return_pct": 0.0,
            "max_dd_pct": 0.0,
            "sharpe": 0.0,
            "days": 0.0,
        }

    ret = equity.iloc[-1] / equity.iloc[0] - 1
    days = max((equity.index[-1] - equity.index[0]).total_seconds() / 86400, 1e-9)
    monthly = (1 + ret) ** (30.4375 / days) - 1 if ret > -1 else -1
    running_max = equity.cummax()
    max_dd = (equity / running_max - 1).min()
    returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    ann_return = (1 + ret) ** (365.25 / days) - 1 if ret > -1 else -1
    ann_vol = returns.std() * np.sqrt(365.25 * BARS_PER_DAY)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    return {
        "return_pct": ret * 100,
        "monthly_return_pct": monthly * 100,
        "max_dd_pct": max_dd * 100,
        "sharpe": sharpe,
        "days": days,
    }


def trade_metrics(trades: list[pd.DataFrame]) -> dict:
    if not trades:
        return {
            "trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "expectancy_pct": 0.0,
        }

    trade_df = pd.concat([frame for frame in trades if not frame.empty], ignore_index=True)
    if trade_df.empty:
        return {
            "trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "expectancy_pct": 0.0,
        }

    pnl = trade_df.get("PnL", pd.Series(dtype=float)).astype(float)
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = pnl[pnl < 0].sum()
    profit_factor = gross_profit / abs(gross_loss) if gross_loss < 0 else (math.inf if gross_profit > 0 else 0.0)

    if "ReturnPct" in trade_df.columns:
        returns_pct = trade_df["ReturnPct"].astype(float) * 100
    else:
        returns_pct = pd.Series(dtype=float)

    return {
        "trades": int(len(trade_df)),
        "win_rate_pct": float((pnl > 0).mean() * 100),
        "profit_factor": float(profit_factor),
        "expectancy_pct": float(returns_pct.mean()) if len(returns_pct) else 0.0,
    }


def combine_segment_results(segment_results: list[dict]) -> dict:
    if not segment_results:
        return {
            "return_pct": 0.0,
            "monthly_return_pct": 0.0,
            "max_dd_pct": 0.0,
            "sharpe": 0.0,
            "days": 0.0,
            "trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "expectancy_pct": 0.0,
        }

    returns = np.array([item["return_pct"] / 100 for item in segment_results], dtype=float)
    total_return = np.prod(1 + returns) - 1
    total_days = sum(item["days"] for item in segment_results)
    monthly = (1 + total_return) ** (30.4375 / total_days) - 1 if total_days > 0 and total_return > -1 else -1
    trades = [item["trades_df"] for item in segment_results if not item["trades_df"].empty]
    trade_stats = trade_metrics(trades)
    return {
        "return_pct": total_return * 100,
        "monthly_return_pct": monthly * 100,
        "max_dd_pct": min(item["max_dd_pct"] for item in segment_results),
        "sharpe": float(np.nanmedian([item["sharpe"] for item in segment_results])),
        "days": total_days,
        **trade_stats,
    }


def evaluate_test_split(
    df: pd.DataFrame,
    split: CpcvSplit,
    params: dict,
    warmup_bars: int,
) -> dict:
    segment_results = []
    for start, end in split.test_ranges:
        warmup_start = max(0, start - warmup_bars)
        sample = df.iloc[warmup_start:end]
        activation_time = df.index[start]
        stats = run_bt(sample, params, activation_time=activation_time)
        equity = stats["_equity_curve"]["Equity"]
        test_equity = equity[equity.index >= activation_time]
        metrics = equity_metrics(test_equity)

        trades = stats["_trades"]
        if not trades.empty:
            exits = pd.to_datetime(trades["ExitTime"])
            trades = trades[exits >= activation_time].copy()
        metrics["trades_df"] = trades
        segment_results.append(metrics)

    return combine_segment_results(segment_results)


def evaluate_train_split(
    df: pd.DataFrame,
    split: CpcvSplit,
    params: dict,
    min_train_bars: int,
) -> dict:
    train_df = df.iloc[np.flatnonzero(split.train_mask)].copy()
    if len(train_df) < min_train_bars:
        return {
            "return_pct": 0.0,
            "monthly_return_pct": 0.0,
            "max_dd_pct": 0.0,
            "sharpe": 0.0,
            "days": 0.0,
            "trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "expectancy_pct": 0.0,
        }

    stats = run_bt(train_df, params)
    metrics = equity_metrics(stats["_equity_curve"]["Equity"])
    trades = stats["_trades"]
    metrics.update(trade_metrics([trades] if not trades.empty else []))
    return metrics


def score_metrics(metrics: dict, min_trades: int = 0) -> float:
    trades = metrics.get("trades", 0)
    ret = metrics.get("return_pct", 0.0)
    monthly = metrics.get("monthly_return_pct", 0.0)
    dd = abs(metrics.get("max_dd_pct", 0.0))
    pf = metrics.get("profit_factor", 0.0)
    expectancy = metrics.get("expectancy_pct", 0.0)
    sharpe = metrics.get("sharpe", 0.0)

    if not np.isfinite(pf):
        pf = 5.0
    if trades < min_trades:
        return -100_000 + trades
    if ret <= 0:
        return -50_000 + ret
    if expectancy <= 0:
        return -40_000 + expectancy
    if pf < 1:
        return -30_000 + pf

    return float(monthly + 0.25 * ret + 6 * expectancy + 0.5 * sharpe + 0.75 * pf - 0.5 * dd + min(trades, 80) * 0.02)


def objective_from_stats(stats: pd.Series) -> float:
    trades = stats.get("# Trades", 0)
    ret = stats.get("Return [%]", np.nan)
    dd = abs(stats.get("Max. Drawdown [%]", np.nan))
    pf = stats.get("Profit Factor", np.nan)
    expectancy = stats.get("Expectancy [%]", np.nan)
    sqn = stats.get("SQN", np.nan)

    if pd.isna(ret) or pd.isna(dd) or pd.isna(expectancy):
        return -1_000_000
    if trades < MIN_TRADES:
        return -100_000 + trades
    if ret <= 0:
        return -50_000 + ret
    if expectancy <= 0:
        return -40_000 + expectancy
    if not pd.isna(pf) and pf < 1:
        return -30_000 + pf
    if pd.isna(sqn):
        sqn = 0
    return float(ret + 10 * expectancy + 0.5 * sqn - 0.75 * dd + min(trades, 80) * 0.02)


def candidate_key(params: dict) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def valid_candidate(params: dict) -> bool:
    return params["ema_slow_period"] > params["ema_fast_period"]


def sample_candidates(max_tries: int, random_states: Iterable[int]) -> list[dict]:
    states = list(random_states)
    if not states:
        states = [42]
    keys = list(PARAM_GRID)
    all_size = math.prod(len(PARAM_GRID[key]) for key in keys)
    target_total = min(max_tries, all_size)
    target_per_seed = max(1, math.ceil(target_total / len(states)))
    selected: dict[str, dict] = {candidate_key(BEST_PARAMS): dict(BEST_PARAMS)}
    target_with_current = target_total + 1

    for seed in states:
        rng = np.random.default_rng(seed)
        attempts = 0
        seed_added = 0
        while seed_added < target_per_seed and len(selected) < target_with_current and attempts < target_per_seed * 120:
            params = {key: rng.choice(PARAM_GRID[key]).item() for key in keys}
            attempts += 1
            if not valid_candidate(params):
                continue
            key = candidate_key(params)
            if key not in selected:
                selected[key] = params
                seed_added += 1
    return list(selected.values())


def metrics_from_equity_ranges(equity: pd.Series, trades: pd.DataFrame, ranges: tuple[tuple[int, int], ...]) -> dict:
    segment_results = []
    for start, end in ranges:
        segment_equity = equity.iloc[start:end]
        metrics = equity_metrics(segment_equity)
        if not trades.empty:
            start_time = equity.index[start]
            end_time = equity.index[end - 1]
            exits = pd.to_datetime(trades["ExitTime"])
            segment_trades = trades[(exits >= start_time) & (exits <= end_time)].copy()
        else:
            segment_trades = trades
        metrics["trades_df"] = segment_trades
        segment_results.append(metrics)
    return combine_segment_results(segment_results)


def approximate_cpcv_rows_from_full_runs(
    df: pd.DataFrame,
    splits: list[CpcvSplit],
    prefiltered: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    all_rows = []
    params_by_label = {}
    for idx, row in prefiltered.iterrows():
        label = f"candidate_{idx + 1:02d}"
        params = json.loads(row["params_json"])
        params_by_label[label] = params
        stats = run_bt(df, params)
        equity = stats["_equity_curve"]["Equity"]
        trades = stats["_trades"]
        for split in splits:
            train_ranges = tuple(contiguous_ranges(split.train_mask))
            train_metrics = metrics_from_equity_ranges(equity, trades, train_ranges)
            test_metrics = metrics_from_equity_ranges(equity, trades, split.test_ranges)
            train_score = score_metrics(train_metrics, min_trades=MIN_TRADES)
            test_score = score_metrics(test_metrics, min_trades=0)
            train_monthly = train_metrics["monthly_return_pct"]
            test_monthly = test_metrics["monthly_return_pct"]
            train_test_gap_pct = max(0.0, (train_monthly - test_monthly) / max(abs(train_monthly), 1e-9) * 100)
            all_rows.append(
                {
                    "params_label": label,
                    "split_id": split.split_id,
                    "test_groups": ",".join(str(group) for group in split.test_groups),
                    "test_start": min(df.index[start] for start, _ in split.test_ranges),
                    "test_end": max(df.index[end - 1] for _, end in split.test_ranges),
                    "purged_train_bars": int(split.train_mask.sum()),
                    "candidate_cpcv_mode": "approx_full_backtest",
                    "train_score": train_score,
                    "test_score": test_score,
                    "train_return_pct": train_metrics["return_pct"],
                    "train_monthly_return_pct": train_monthly,
                    "train_max_dd_pct": train_metrics["max_dd_pct"],
                    "train_profit_factor": train_metrics["profit_factor"],
                    "train_trades": train_metrics["trades"],
                    "test_return_pct": test_metrics["return_pct"],
                    "test_monthly_return_pct": test_monthly,
                    "test_max_dd_pct": test_metrics["max_dd_pct"],
                    "test_profit_factor": test_metrics["profit_factor"],
                    "test_trades": test_metrics["trades"],
                    "test_win_rate_pct": test_metrics["win_rate_pct"],
                    "test_expectancy_pct": test_metrics["expectancy_pct"],
                    "train_test_gap_pct": train_test_gap_pct,
                }
            )
    return pd.DataFrame(all_rows), params_by_label


def prefilter_candidates(df: pd.DataFrame, candidates: list[dict], top_candidates: int) -> pd.DataFrame:
    rows = []
    for idx, params in enumerate(candidates, start=1):
        try:
            stats = run_bt(df, params)
            score = objective_from_stats(stats)
            rows.append(
                {
                    "candidate_id": idx,
                    "prefilter_score": score,
                    "prefilter_return_pct": float(stats["Return [%]"]),
                    "prefilter_max_dd_pct": float(stats["Max. Drawdown [%]"]),
                    "prefilter_profit_factor": float(stats["Profit Factor"]),
                    "prefilter_trades": int(stats["# Trades"]),
                    "params_json": candidate_key(params),
                    **params,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "candidate_id": idx,
                    "prefilter_score": -1_000_000,
                    "prefilter_error": str(exc),
                    "params_json": candidate_key(params),
                    **params,
                }
            )
    result = pd.DataFrame(rows).sort_values("prefilter_score", ascending=False)
    return result.head(top_candidates).reset_index(drop=True)


def cpcv_rows_for_params(
    df: pd.DataFrame,
    splits: list[CpcvSplit],
    params: dict,
    params_label: str,
    warmup_bars: int,
    min_train_bars: int,
) -> pd.DataFrame:
    rows = []
    for split in splits:
        train_metrics = evaluate_train_split(df, split, params, min_train_bars=min_train_bars)
        test_metrics = evaluate_test_split(df, split, params, warmup_bars=warmup_bars)
        train_score = score_metrics(train_metrics, min_trades=MIN_TRADES)
        test_score = score_metrics(test_metrics, min_trades=0)
        train_monthly = train_metrics["monthly_return_pct"]
        test_monthly = test_metrics["monthly_return_pct"]
        train_test_gap_pct = max(0.0, (train_monthly - test_monthly) / max(abs(train_monthly), 1e-9) * 100)
        rows.append(
            {
                "params_label": params_label,
                "split_id": split.split_id,
                "test_groups": ",".join(str(group) for group in split.test_groups),
                "test_start": min(df.index[start] for start, _ in split.test_ranges),
                "test_end": max(df.index[end - 1] for _, end in split.test_ranges),
                "purged_train_bars": int(split.train_mask.sum()),
                "train_score": train_score,
                "test_score": test_score,
                "train_return_pct": train_metrics["return_pct"],
                "train_monthly_return_pct": train_monthly,
                "train_max_dd_pct": train_metrics["max_dd_pct"],
                "train_profit_factor": train_metrics["profit_factor"],
                "train_trades": train_metrics["trades"],
                "test_return_pct": test_metrics["return_pct"],
                "test_monthly_return_pct": test_monthly,
                "test_max_dd_pct": test_metrics["max_dd_pct"],
                "test_profit_factor": test_metrics["profit_factor"],
                "test_trades": test_metrics["trades"],
                "test_win_rate_pct": test_metrics["win_rate_pct"],
                "test_expectancy_pct": test_metrics["expectancy_pct"],
                "train_test_gap_pct": train_test_gap_pct,
            }
        )
    return pd.DataFrame(rows)


def summarize_cpcv(rows: pd.DataFrame, label: str, pbo: float | None = None) -> dict:
    if rows.empty:
        return {
            "label": label,
            "splits": 0,
            "passed_gate": False,
            "pbo": pbo,
        }

    profitable_pct = (rows["test_return_pct"] > 0).mean() * 100
    median_monthly = rows["test_monthly_return_pct"].median()
    pf_series = rows["test_profit_factor"].replace([np.inf, -np.inf], np.nan)
    median_pf = pf_series.median()
    if pd.isna(median_pf) and np.isposinf(rows["test_profit_factor"]).any():
        median_pf = 5.0
    if pd.isna(median_pf):
        median_pf = 0.0
    median_dd = rows["test_max_dd_pct"].median()
    median_gap = rows["train_test_gap_pct"].median()
    p10_monthly = rows["test_monthly_return_pct"].quantile(0.10)
    total_test_trades = int(rows["test_trades"].sum())
    passed_gate = (
        median_monthly > 0
        and profitable_pct >= 65
        and median_pf >= 1.2
        and median_dd > -25
        and median_gap <= 40
    )
    if pbo is not None:
        passed_gate = passed_gate and pbo < 0.20

    return {
        "label": label,
        "splits": int(len(rows)),
        "median_test_monthly_return_pct": float(median_monthly),
        "p10_test_monthly_return_pct": float(p10_monthly),
        "median_test_return_pct": float(rows["test_return_pct"].median()),
        "profitable_splits_pct": float(profitable_pct),
        "median_test_profit_factor": float(min(median_pf, 5.0)) if np.isfinite(median_pf) else 5.0,
        "median_test_max_dd_pct": float(median_dd),
        "median_train_test_gap_pct": float(median_gap),
        "total_test_trades": total_test_trades,
        "pbo": pbo,
        "passed_gate": bool(passed_gate),
    }


def estimate_pbo(candidate_split_rows: pd.DataFrame) -> float:
    failures = 0
    evaluated = 0
    for _, split_rows in candidate_split_rows.groupby("split_id"):
        if split_rows["train_score"].nunique() < 2 or split_rows["test_score"].nunique() < 2:
            continue
        train_winner = split_rows.sort_values("train_score", ascending=False).iloc[0]
        ranked = split_rows["test_score"].rank(method="average", pct=True)
        winner_rank_pct = float(ranked.loc[train_winner.name])
        winner_rank_pct = min(max(winner_rank_pct, 1e-6), 1 - 1e-6)
        logit = math.log(winner_rank_pct / (1 - winner_rank_pct))
        failures += int(logit < 0)
        evaluated += 1
    return failures / evaluated if evaluated else math.nan


def robust_candidate_summary(candidate_rows: pd.DataFrame, pbo_by_label: dict[str, float]) -> pd.DataFrame:
    rows = []
    for label, group in candidate_rows.groupby("params_label"):
        summary = summarize_cpcv(group, label=label, pbo=pbo_by_label.get(label))
        penalty_gap = summary["median_train_test_gap_pct"] * 0.15
        penalty_dd = abs(min(summary["median_test_max_dd_pct"], 0)) * 0.20
        consistency_bonus = summary["profitable_splits_pct"] * 0.10
        robust_score = (
            summary["median_test_monthly_return_pct"] * 2.0
            + summary["median_test_return_pct"] * 0.30
            + summary["median_test_profit_factor"] * 1.5
            + consistency_bonus
            - penalty_gap
            - penalty_dd
        )
        summary["robust_score"] = float(robust_score)
        rows.append(summary)
    return pd.DataFrame(rows).sort_values("robust_score", ascending=False).reset_index(drop=True)


def run_fixed(
    df: pd.DataFrame,
    splits: list[CpcvSplit],
    output_dir: Path,
    warmup_bars: int,
    min_train_bars: int,
    params: dict,
    label: str,
) -> tuple[pd.DataFrame, dict]:
    rows = cpcv_rows_for_params(
        df,
        splits,
        params=params,
        params_label=label,
        warmup_bars=warmup_bars,
        min_train_bars=min_train_bars,
    )
    summary = summarize_cpcv(rows, label=label)
    rows.to_csv(output_dir / "cpcv_fixed_splits.csv", index=False)
    pd.DataFrame([summary]).to_csv(output_dir / "cpcv_fixed_summary.csv", index=False)
    return rows, summary


def run_reoptimization(
    df: pd.DataFrame,
    splits: list[CpcvSplit],
    output_dir: Path,
    warmup_bars: int,
    min_train_bars: int,
    max_tries: int,
    random_states: list[int],
    top_candidates: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    candidates = sample_candidates(max_tries=max_tries, random_states=random_states)
    prefiltered = prefilter_candidates(df, candidates, top_candidates=top_candidates)
    prefiltered.to_csv(output_dir / "cpcv_optimizer_prefilter.csv", index=False)

    candidate_rows, params_by_label = approximate_cpcv_rows_from_full_runs(df, splits, prefiltered)
    pbo = estimate_pbo(candidate_rows) if not candidate_rows.empty else math.nan
    summary = robust_candidate_summary(candidate_rows, {label: pbo for label in params_by_label})
    summary.to_csv(output_dir / "cpcv_optimizer_candidates.csv", index=False)
    candidate_rows.to_csv(output_dir / "cpcv_optimizer_split_scores.csv", index=False)

    if summary.empty:
        best_summary = {"label": "none", "passed_gate": False, "pbo": pbo}
        best_params = {}
    else:
        best_label = str(summary.iloc[0]["label"])
        best_params = params_by_label[best_label]
        best_rows = cpcv_rows_for_params(
            df,
            splits,
            params=best_params,
            params_label=best_label,
            warmup_bars=warmup_bars,
            min_train_bars=min_train_bars,
        )
        best_rows.to_csv(output_dir / "cpcv_reoptimized_splits.csv", index=False)
        best_summary = summarize_cpcv(best_rows, label=best_label, pbo=pbo)
        best_summary["robust_score"] = float(summary.iloc[0]["robust_score"])

    pd.DataFrame([best_summary]).to_csv(output_dir / "cpcv_reoptimized_summary.csv", index=False)
    (output_dir / "cpcv_reoptimized_best_params.json").write_text(
        json.dumps(best_params, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return candidate_rows, summary, best_summary, best_params


def walk_forward(df: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, pd.Series]:
    train_bars = TRAIN_DAYS * BARS_PER_DAY
    test_bars = TEST_DAYS * BARS_PER_DAY
    rows = []
    all_trade_returns = []

    fold = 1
    start = 0
    while start + train_bars + test_bars <= len(df):
        sample = df.iloc[start : start + train_bars + test_bars]
        activation_time = sample.index[train_bars]
        stats = run_bt(sample, params, activation_time=activation_time)
        equity = stats["_equity_curve"]["Equity"]
        test_equity = equity[equity.index >= activation_time]
        metrics = equity_metrics(test_equity)

        trades = stats["_trades"]
        if not trades.empty:
            exits = pd.to_datetime(trades["ExitTime"])
            test_trades = trades[exits >= activation_time].copy()
            if "ReturnPct" in test_trades.columns:
                all_trade_returns.extend(test_trades["ReturnPct"].astype(float).to_list())
        else:
            test_trades = trades

        rows.append(
            {
                "fold": fold,
                "test_start": test_equity.index[0] if len(test_equity) else activation_time,
                "test_end": test_equity.index[-1] if len(test_equity) else sample.index[-1],
                **{key: value for key, value in metrics.items() if key != "days"},
                "test_trades": int(len(test_trades)),
            }
        )
        start += test_bars
        fold += 1

    return pd.DataFrame(rows), pd.Series(all_trade_returns, dtype=float)


def monte_carlo(trade_returns: pd.Series) -> tuple[pd.DataFrame, dict]:
    returns = trade_returns.dropna().to_numpy(dtype=float)
    if len(returns) == 0:
        return pd.DataFrame(), {}

    rng = np.random.default_rng(42)
    rows = []
    for run in range(MONTE_CARLO_RUNS):
        sampled = rng.choice(returns, size=len(returns), replace=True)
        equity = np.insert(np.cumprod(1 + sampled), 0, 1.0)
        dd = equity / np.maximum.accumulate(equity) - 1
        rows.append(
            {
                "run": run + 1,
                "final_return_pct": (equity[-1] - 1) * 100,
                "max_dd_pct": dd.min() * 100,
            }
        )

    mc = pd.DataFrame(rows)
    summary = {
        "mc_runs": MONTE_CARLO_RUNS,
        "mc_trades": len(returns),
        "mc_final_return_p05": float(mc["final_return_pct"].quantile(0.05)),
        "mc_final_return_p50": float(mc["final_return_pct"].quantile(0.50)),
        "mc_final_return_p95": float(mc["final_return_pct"].quantile(0.95)),
        "mc_probability_loss_pct": float((mc["final_return_pct"] < 0).mean() * 100),
        "mc_max_dd_p05": float(mc["max_dd_pct"].quantile(0.05)),
        "mc_max_dd_p50": float(mc["max_dd_pct"].quantile(0.50)),
    }
    return mc, summary


def run_walk_forward_monte_carlo(df: pd.DataFrame, params: dict, output_dir: Path, prefix: str) -> dict:
    folds, trade_returns = walk_forward(df, params)
    mc, mc_summary = monte_carlo(trade_returns)

    folds.to_csv(output_dir / f"{prefix}_walkforward.csv", index=False)
    mc.to_csv(output_dir / f"{prefix}_monte_carlo.csv", index=False)

    wf_summary = {
        "wf_folds": int(len(folds)),
        "wf_total_return_pct_sum": float(folds["return_pct"].sum()) if not folds.empty else 0.0,
        "wf_median_fold_return_pct": float(folds["return_pct"].median()) if not folds.empty else 0.0,
        "wf_best_fold_return_pct": float(folds["return_pct"].max()) if not folds.empty else 0.0,
        "wf_worst_fold_return_pct": float(folds["return_pct"].min()) if not folds.empty else 0.0,
        "wf_profitable_folds_pct": float((folds["return_pct"] > 0).mean() * 100) if not folds.empty else 0.0,
        "wf_avg_monthly_return_pct": float(folds["monthly_return_pct"].mean()) if not folds.empty else 0.0,
        "wf_avg_max_dd_pct": float(folds["max_dd_pct"].mean()) if not folds.empty else 0.0,
        "wf_total_test_trades": int(folds["test_trades"].sum()) if not folds.empty else 0,
    }
    summary = {**wf_summary, **mc_summary}
    pd.DataFrame([summary]).to_csv(output_dir / f"{prefix}_wf_mc_summary.csv", index=False)
    return summary


def plot_distribution(
    output_dir: Path,
    fixed_rows: pd.DataFrame | None,
    reopt_rows: pd.DataFrame | None,
    best_label: str | None,
) -> None:
    plt.figure(figsize=(10, 6))
    if fixed_rows is not None and not fixed_rows.empty:
        plt.hist(
            fixed_rows["test_monthly_return_pct"],
            bins=14,
            alpha=0.65,
            label="Current params",
            color="#2F6BFF",
        )
    if reopt_rows is not None and best_label:
        best_rows = reopt_rows[reopt_rows["params_label"] == best_label]
        if not best_rows.empty:
            plt.hist(
                best_rows["test_monthly_return_pct"],
                bins=14,
                alpha=0.65,
                label="Reoptimized params",
                color="#F59E0B",
            )
    plt.axvline(0, color="#111111", linewidth=1)
    plt.title("CPCV monthly return distribution")
    plt.xlabel("Monthly return [%]")
    plt.ylabel("Split count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "cpcv_return_distribution.png", dpi=160)
    plt.close()


def write_report(
    output_dir: Path,
    settings: dict,
    fixed_summary: dict | None,
    reopt_summary: dict | None,
    best_params: dict | None,
    wf_mc_summary: dict | None = None,
) -> None:
    lines = [
        "CPCV anti-overfitting report",
        "",
        "Settings",
    ]
    for key, value in settings.items():
        lines.append(f"{key}: {value}")

    if fixed_summary:
        lines.extend(["", "Fixed-parameter CPCV"])
        for key, value in fixed_summary.items():
            lines.append(f"{key}: {value}")

    if reopt_summary:
        lines.extend(["", "Reoptimized-parameter CPCV"])
        for key, value in reopt_summary.items():
            lines.append(f"{key}: {value}")

    if wf_mc_summary:
        lines.extend(["", "Reoptimized walk-forward and Monte Carlo"])
        for key, value in wf_mc_summary.items():
            lines.append(f"{key}: {value}")

    if best_params:
        lines.extend(["", "Best reoptimized params", json.dumps(best_params, indent=2, sort_keys=True)])

    output_path = output_dir / "cpcv_report.txt"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPCV validation for the manual BTCUSDT strategy.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--mode", choices=["fixed", "reoptimize", "full"], default="full")
    parser.add_argument("--n-groups", type=int, default=8)
    parser.add_argument("--test-groups", type=int, default=2)
    parser.add_argument("--purge-bars", type=int, default=1200)
    parser.add_argument("--embargo-bars", type=int, default=336)
    parser.add_argument("--warmup-bars", type=int, default=1200)
    parser.add_argument("--max-tries", type=int, default=120)
    parser.add_argument("--random-states", nargs="+", type=int, default=[42, 1337, 2026])
    parser.add_argument("--top-candidates", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.csv)
    group_ranges, splits = build_cpcv_splits(
        len(df),
        n_groups=args.n_groups,
        test_groups=args.test_groups,
        purge_bars=args.purge_bars,
        embargo_bars=args.embargo_bars,
    )
    min_train_bars = max(args.warmup_bars + 200, 1500)

    settings = {
        "csv": resolve_path(args.csv),
        "rows": len(df),
        "start": df.index.min(),
        "end": df.index.max(),
        "mode": args.mode,
        "n_groups": args.n_groups,
        "test_groups": args.test_groups,
        "splits": len(splits),
        "purge_bars": args.purge_bars,
        "embargo_bars": args.embargo_bars,
        "warmup_bars": args.warmup_bars,
        "max_tries": args.max_tries,
        "random_states": args.random_states,
        "top_candidates": args.top_candidates,
        "candidate_cpcv_mode": "approx_full_backtest_for_ranking_then_strict_cpcv_for_selected_candidate",
        "group_ranges": group_ranges,
    }

    fixed_rows = None
    fixed_summary = None
    reopt_rows = None
    reopt_summary = None
    best_params = None
    wf_mc_summary = None

    if args.mode in {"fixed", "full"}:
        fixed_rows, fixed_summary = run_fixed(
            df,
            splits,
            output_dir=output_dir,
            warmup_bars=args.warmup_bars,
            min_train_bars=min_train_bars,
            params=BEST_PARAMS,
            label="current_best",
        )

    should_reoptimize = args.mode == "reoptimize" or (
        args.mode == "full" and fixed_summary is not None and fixed_summary["passed_gate"]
    )

    if should_reoptimize:
        reopt_rows, candidate_summary, reopt_summary, best_params = run_reoptimization(
            df,
            splits,
            output_dir=output_dir,
            warmup_bars=args.warmup_bars,
            min_train_bars=min_train_bars,
            max_tries=args.max_tries,
            random_states=args.random_states,
            top_candidates=args.top_candidates,
        )
        best_label = reopt_summary.get("label") if reopt_summary else None
        if best_params:
            wf_mc_summary = run_walk_forward_monte_carlo(
                df,
                best_params,
                output_dir=output_dir,
                prefix="cpcv_reoptimized",
            )
    else:
        best_label = None

    plot_distribution(output_dir, fixed_rows, reopt_rows, best_label)
    write_report(output_dir, settings, fixed_summary, reopt_summary, best_params, wf_mc_summary)

    if fixed_summary:
        print("Fixed CPCV summary")
        print(pd.DataFrame([fixed_summary]).to_string(index=False))
    if reopt_summary:
        print("\nReoptimized CPCV summary")
        print(pd.DataFrame([reopt_summary]).to_string(index=False))
    print(f"\nReport written to: {output_dir / 'cpcv_report.txt'}")


if __name__ == "__main__":
    main()
