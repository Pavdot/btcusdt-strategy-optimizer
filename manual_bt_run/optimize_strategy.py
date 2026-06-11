from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from backtesting.lib import FractionalBacktest

from my_strategy import MyStrategy


# =========================================================
# CONFIG
# =========================================================
CSV_PATH = Path("data/BTCUSDT_1h.csv")

INITIAL_CASH = 10_000
COMMISSION = 0.001

MIN_TRADES = 15


# =========================================================
# DATA
# =========================================================
def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV introuvable : {csv_path}\n"
            "Lance d'abord run_backtest.py pour télécharger/générer le dataset."
        )

    df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
    df = df.sort_index()

    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV : {missing}")

    return df[["Open", "High", "Low", "Close", "Volume"]]


# =========================================================
# OBJECTIVE FUNCTION
# =========================================================
def objective(stats: pd.Series) -> float:
    """
    Score custom pour éviter les configs qui font 1 ou 3 trades.

    Objectif :
    - Return positif
    - Profit Factor > 1
    - Expectancy positive
    - Drawdown contenu
    - nombre de trades suffisant
    """

    trades = stats.get("# Trades", 0)
    ret = stats.get("Return [%]", np.nan)
    dd = abs(stats.get("Max. Drawdown [%]", np.nan))
    pf = stats.get("Profit Factor", np.nan)
    expectancy = stats.get("Expectancy [%]", np.nan)
    sqn = stats.get("SQN", np.nan)

    if pd.isna(ret) or pd.isna(dd) or pd.isna(expectancy):
        return -1_000_000

    # Refuse les configs trop peu représentatives
    if trades < MIN_TRADES:
        return -100_000 + trades

    # Refuse les configs structurellement mauvaises
    if ret <= 0:
        return -50_000 + ret

    if expectancy <= 0:
        return -40_000 + expectancy

    if not pd.isna(pf) and pf < 1:
        return -30_000 + pf

    if pd.isna(sqn):
        sqn = 0

    # Score composite
    # Return et expectancy comptent, drawdown pénalise.
    score = (
        ret
        + 10 * expectancy
        + 0.5 * sqn
        - 0.75 * dd
        + min(trades, 80) * 0.02
    )

    return float(score)


# =========================================================
# OPTIMIZATION
# =========================================================
def run_optimization(df: pd.DataFrame, max_tries: int = 1200, random_state: int = 42):
    bt = FractionalBacktest(
        df,
        MyStrategy,
        cash=INITIAL_CASH,
        commission=COMMISSION,
        trade_on_close=False,
        exclusive_orders=True,
    )

    print("\n🚀 Optimisation mode continuation V1")
    print("=" * 70)
    print(f"Nombre de bougies : {len(df)}")
    print(f"Début            : {df.index.min()}")
    print(f"Fin              : {df.index.max()}")
    print(f"Minimum trades   : {MIN_TRADES}")
    print("=" * 70)

    stats, heatmap = bt.optimize(
        maximize=objective,
        method="grid",
        return_heatmap=True,
        max_tries=max_tries,
        random_state=random_state,

        constraint=lambda p: (
            p.ema_slow_period > p.ema_fast_period
        ),

        # =================================================
        # Structure, on garde assez fixe pour ne pas exploser
        # =================================================
        ema_fast_period=[200, 250, 300],
        ema_slow_period=[600, 750, 900],

        rsi_period=[14],
        atr_period=[14],

        vp_tactical_lookback=[252, 336],
        vp_macro_lookback=[750, 1000],
        vp_bins=[16, 24],

        breakout_lookback=[20],
        rsi_rank_lookback=[50],
        ema_slope_lookback=[20],

        # =================================================
        # Paramètres à desserrer / tester
        # =================================================
        min_body_ratio=[0.4, 0.6, 0.8, 1.0],
        min_vol_ratio=[0.5, 0.8, 1.0],
        min_break_dist_atr=[0.00, 0.02, 0.03, 0.05],

        min_node_percentile=[0.35, 0.50, 0.65],
        min_room_to_target_atr=[0.5, 0.75, 1.0, 1.25],

        min_rsi_percentile_long=[0.25, 0.35, 0.45],
        max_rsi_percentile_short=[0.55, 0.65, 0.75],

        min_close_position_long=[0.50, 0.55, 0.60],
        max_close_position_short=[0.40, 0.45, 0.50],

        stop_atr_buffer=[0.3, 0.5, 0.75],
        tp_fraction_to_next_level=[0.65, 0.85, 1.0],
    )

    return stats, heatmap


# =========================================================
# REPORTING
# =========================================================
def print_report(stats: pd.Series, heatmap: pd.Series):
    print("\n✅ MEILLEURE CONFIG TROUVÉE")
    print("=" * 70)
    print(stats["_strategy"])

    print("\n📊 STATS")
    print("=" * 70)
    print(stats)

    print("\n💹 RÉSUMÉ")
    print("=" * 70)
    print(f"Return [%]             : {stats['Return [%]']:.2f}%")
    print(f"Buy & Hold Return [%]  : {stats['Buy & Hold Return [%]']:.2f}%")
    print(f"Return Ann. [%]        : {stats['Return (Ann.) [%]']:.2f}%")
    print(f"Sharpe Ratio           : {stats['Sharpe Ratio']:.2f}")
    print(f"Sortino Ratio          : {stats['Sortino Ratio']:.2f}")
    print(f"Calmar Ratio           : {stats['Calmar Ratio']:.2f}")
    print(f"Max Drawdown [%]       : {stats['Max. Drawdown [%]']:.2f}%")
    print(f"# Trades               : {int(stats['# Trades'])}")
    print(f"Win Rate [%]           : {stats['Win Rate [%]']:.2f}%")
    print(f"Best Trade [%]         : {stats['Best Trade [%]']:.2f}%")
    print(f"Worst Trade [%]        : {stats['Worst Trade [%]']:.2f}%")
    print(f"Avg Trade [%]          : {stats['Avg. Trade [%]']:.4f}%")
    print(f"Profit Factor          : {stats['Profit Factor']:.2f}")
    print(f"Expectancy [%]         : {stats['Expectancy [%]']:.4f}%")
    print(f"SQN                    : {stats['SQN']:.2f}")
    print(f"Commissions [$]        : {stats['Commissions [$]']:.2f}")

    print("\n📈 OPTIMISATION")
    print("=" * 70)
    print(f"Combinaisons évaluées : {len(heatmap)}")

    heatmap_path = Path("optimization_heatmap.csv")
    heatmap.sort_values(ascending=False).to_csv(heatmap_path)
    print(f"Heatmap sauvegardée   : {heatmap_path}")


# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--max-tries", type=int, default=1200)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = load_data(args.csv)
    stats, heatmap = run_optimization(df, max_tries=args.max_tries, random_state=args.random_state)
    print_report(stats, heatmap)


if __name__ == "__main__":
    main()
