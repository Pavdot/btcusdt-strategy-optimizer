import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

import strategy_factory_btc_search as sf


OUT_DIR = Path("optimization_runs")
OUT_DIR.mkdir(exist_ok=True)


def evaluate_candidate_pool(
    df: pd.DataFrame,
    funding: pd.Series,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    discovery = df.iloc[
        -(sf.DISCOVERY_DAYS + sf.HOLDOUT_DAYS) * sf.BARS_PER_DAY : -sf.HOLDOUT_DAYS * sf.BARS_PER_DAY
    ]
    holdout = df.iloc[-sf.HOLDOUT_DAYS * sf.BARS_PER_DAY :]
    samples = {
        "discovery": discovery,
        "holdout": holdout,
        "full": df,
    }
    caches = {name: sf.build_feature_cache(sample) for name, sample in samples.items()}
    fundings = {
        name: funding.reindex(sample.index).fillna(0.0).to_numpy(dtype=float)
        for name, sample in samples.items()
    }

    pool = pd.concat(
        [
            candidates.sort_values("score", ascending=False).head(50),
            candidates.sort_values("compound_monthly_return_pcent", ascending=False).head(50),
            candidates[candidates["trade_count"] >= 50]
            .sort_values("profit_factor", ascending=False)
            .head(50),
            candidates[candidates["max_dd_pcent"] >= -15]
            .sort_values("compound_monthly_return_pcent", ascending=False)
            .head(50),
        ],
        ignore_index=True,
    )
    pool = pool.drop_duplicates("strategy_id")

    rows = []
    for _, row in pool.iterrows():
        params = sf.params_from_row(row)
        for sample_name, sample in samples.items():
            result, trades = sf.run_backtest(sample, caches[sample_name], fundings[sample_name], params)
            metrics = sf.metrics_from_result(result, trades)
            out = asdict(params)
            out["sample"] = sample_name
            out.update(metrics)
            rows.append(out)
    return pd.DataFrame(rows)


def choose_candidate(evaluations: pd.DataFrame) -> pd.Series:
    holdout = evaluations[evaluations["sample"] == "holdout"].copy()
    if holdout.empty:
        raise RuntimeError("No holdout evaluations were generated.")

    holdout["selection_score"] = (
        holdout["compound_monthly_return_pcent"]
        + 3.0 * holdout["sharpe"]
        + 2.0 * holdout["profit_factor"].replace(np.inf, 5.0).clip(upper=5.0)
        + 0.35 * holdout["max_dd_pcent"]
        + np.log1p(holdout["trade_count"]) * 0.75
    )

    target_hits = holdout[
        (holdout["compound_monthly_return_pcent"] >= sf.TARGET_MONTHLY_RETURN * 100)
        & (holdout["trade_count"] >= 30)
        & (holdout["max_dd_pcent"] >= -25)
    ]
    if not target_hits.empty:
        return target_hits.sort_values("selection_score", ascending=False).iloc[0]

    active = holdout[holdout["trade_count"] >= 20]
    if not active.empty:
        return active.sort_values("selection_score", ascending=False).iloc[0]

    return holdout.sort_values("selection_score", ascending=False).iloc[0]


def run_round(interval: str, seed: int, candidate_count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    sf.configure(interval)
    sf.RANDOM_SEED = seed

    df = sf.load_klines(sf.KLINES_PATH)
    funding = sf.load_funding(sf.FUNDING_PATH, df.index)
    print(
        f"round interval={interval} seed={seed} bars={len(df)} candidates={candidate_count}",
        flush=True,
    )
    search = sf.run_search(df, funding, candidate_count=candidate_count)
    search["interval"] = interval
    search["seed"] = seed
    search_path = OUT_DIR / f"search_{interval}_seed_{seed}.csv"
    search.to_csv(search_path, index=False)

    evals = evaluate_candidate_pool(df, funding, search)
    evals["interval"] = interval
    evals["seed"] = seed
    eval_path = OUT_DIR / f"eval_{interval}_seed_{seed}.csv"
    evals.to_csv(eval_path, index=False)
    return search, evals


def robust_check(chosen: pd.Series) -> tuple[dict, dict, bool, list[str]]:
    interval = str(chosen["interval"])
    sf.configure(interval)
    df = sf.load_klines(sf.KLINES_PATH)
    funding = sf.load_funding(sf.FUNDING_PATH, df.index)
    params = sf.params_from_row(chosen)

    folds, wf_returns, wf_trades = sf.fixed_walk_forward(df, funding, params)
    wf_summary = sf.summarize_walk_forward(wf_returns, wf_trades, folds)
    _, mc_summary = sf.monte_carlo(
        wf_trades["net_return"] if not wf_trades.empty else pd.Series(dtype=float)
    )
    passed, failures = sf.passes_gates(chosen.to_dict(), wf_summary, mc_summary)
    sf.plot_candidate(df, funding, params)
    return wf_summary, mc_summary, passed, failures


def write_summary(
    all_search: pd.DataFrame,
    all_evals: pd.DataFrame,
    chosen: pd.Series,
    wf_summary: dict,
    mc_summary: dict,
    passed: bool,
    failures: list[str],
) -> None:
    search_path = OUT_DIR / "all_search_results.csv"
    eval_path = OUT_DIR / "all_candidate_evaluations.csv"
    summary_path = OUT_DIR / "optimization_summary.txt"
    all_search.to_csv(search_path, index=False)
    all_evals.to_csv(eval_path, index=False)

    lines = []
    lines.append("BTCUSDT strategy optimization")
    lines.append(f"search_rows: {len(all_search)}")
    lines.append(f"evaluated_rows: {len(all_evals)}")
    lines.append(f"target_monthly_return_pcent: {sf.TARGET_MONTHLY_RETURN * 100:.2f}")
    lines.append("")
    lines.append("Best discovery candidates by monthly return")
    cols = [
        "interval",
        "seed",
        "strategy_id",
        "template",
        "compound_monthly_return_pcent",
        "total_return_pcent",
        "max_dd_pcent",
        "profit_factor",
        "trade_count",
        "leverage",
    ]
    lines.append(all_search.sort_values("compound_monthly_return_pcent", ascending=False).head(12)[cols].to_string(index=False))
    lines.append("")
    lines.append("Chosen candidate from holdout")
    for key in [
        "interval",
        "seed",
        "strategy_id",
        "template",
        "compound_monthly_return_pcent",
        "total_return_pcent",
        "max_dd_pcent",
        "profit_factor",
        "trade_count",
        "win_rate_pcent",
        "sharpe",
        "sortino",
        "leverage",
        "sl_pcent",
        "tp_pcent",
        "max_hold_bars",
        "funding_filter",
    ]:
        value = chosen.get(key)
        if isinstance(value, float):
            lines.append(f"{key}: {value:.4f}")
        else:
            lines.append(f"{key}: {value}")
    lines.append("")
    lines.append("Walk-forward")
    for key, value in wf_summary.items():
        if isinstance(value, float):
            lines.append(f"{key}: {value:.4f}")
        else:
            lines.append(f"{key}: {value}")
    lines.append("")
    lines.append("Monte Carlo")
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
    summary_path.write_text(text, encoding="utf-8")
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intervals", nargs="+", default=["15m", "5m"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260611, 20260612])
    parser.add_argument("--candidates", type=int, default=300)
    args = parser.parse_args()

    searches = []
    evals = []
    for interval in args.intervals:
        for seed in args.seeds:
            search, evaluation = run_round(interval, seed, args.candidates)
            searches.append(search)
            evals.append(evaluation)

    all_search = pd.concat(searches, ignore_index=True)
    all_evals = pd.concat(evals, ignore_index=True)
    chosen = choose_candidate(all_evals)
    wf_summary, mc_summary, passed, failures = robust_check(chosen)
    write_summary(all_search, all_evals, chosen, wf_summary, mc_summary, passed, failures)


if __name__ == "__main__":
    main()
