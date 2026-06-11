import json
from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path("btcusdt_1h_klines.json")
RESULTS_PATH = Path("backtest_results_btcusdt_1h_trend.csv")
SUMMARY_PATH = Path("backtest_summary_btcusdt_1h_trend.txt")


SHORT_WINDOW = 20
LONG_WINDOW = 50
SL_PCENT = 0.02
TP_PCENT = 0.05
SPREAD = 0.0002
FEES_PCENT = 0.00075


def load_binance_klines(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

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
    df = pd.DataFrame(raw, columns=columns)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["open", "high", "low", "close", "volume"]].dropna()


def generate_trend_signals(df: pd.DataFrame) -> pd.Series:
    short_ma = df["close"].rolling(SHORT_WINDOW, min_periods=SHORT_WINDOW).mean()
    long_ma = df["close"].rolling(LONG_WINDOW, min_periods=LONG_WINDOW).mean()
    in_uptrend = short_ma > long_ma
    was_uptrend = in_uptrend.shift(1, fill_value=False).astype(bool)
    entries = in_uptrend & ~was_uptrend
    exits = ~in_uptrend & was_uptrend

    raw_signal = pd.Series(0, index=df.index, dtype=int)
    raw_signal.loc[entries] = 1
    raw_signal.loc[exits] = -1

    # Crossovers are only known after the bar closes, so execute on the next bar.
    signal = raw_signal.shift(1).fillna(0).astype(int)
    return signal


def run_backtest(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["signal"] = generate_trend_signals(out)
    out["position"] = 0
    out["strategy_return"] = 0.0
    out["cost"] = 0.0
    out["entry_price"] = np.nan
    out["exit_price"] = np.nan
    out["exit_reason"] = ""
    out["trade_pnl_gross"] = np.nan

    position = 0
    entry_price = np.nan

    for i in range(1, len(out)):
        idx = out.index[i]
        prev_idx = out.index[i - 1]

        signal = int(out.at[idx, "signal"])
        current_open = float(out.at[idx, "open"])
        current_high = float(out.at[idx, "high"])
        current_low = float(out.at[idx, "low"])
        current_close = float(out.at[idx, "close"])
        prev_close = float(out.at[prev_idx, "close"])

        if position == 1:
            sl_level = entry_price * (1 - SL_PCENT)
            tp_level = entry_price * (1 + TP_PCENT)

            if current_low <= sl_level:
                out.at[idx, "strategy_return"] = (sl_level - prev_close) / prev_close
                out.at[idx, "position"] = 0
                out.at[idx, "cost"] += SPREAD + FEES_PCENT
                out.at[idx, "exit_price"] = sl_level
                out.at[idx, "exit_reason"] = "stop_loss"
                out.at[idx, "trade_pnl_gross"] = (sl_level - entry_price) / entry_price
                position = 0
                entry_price = np.nan
            elif current_high >= tp_level:
                out.at[idx, "strategy_return"] = (tp_level - prev_close) / prev_close
                out.at[idx, "position"] = 0
                out.at[idx, "cost"] += SPREAD + FEES_PCENT
                out.at[idx, "exit_price"] = tp_level
                out.at[idx, "exit_reason"] = "take_profit"
                out.at[idx, "trade_pnl_gross"] = (tp_level - entry_price) / entry_price
                position = 0
                entry_price = np.nan
            elif signal == -1:
                out.at[idx, "strategy_return"] = (current_open - prev_close) / prev_close
                out.at[idx, "position"] = 0
                out.at[idx, "cost"] += SPREAD + FEES_PCENT
                out.at[idx, "exit_price"] = current_open
                out.at[idx, "exit_reason"] = "trend_exit"
                out.at[idx, "trade_pnl_gross"] = (current_open - entry_price) / entry_price
                position = 0
                entry_price = np.nan
            else:
                out.at[idx, "strategy_return"] = (current_close - prev_close) / prev_close
                out.at[idx, "position"] = 1
                out.at[idx, "entry_price"] = entry_price

        if position == 0 and signal == 1:
            position = 1
            entry_price = current_open
            out.at[idx, "cost"] += SPREAD + FEES_PCENT
            out.at[idx, "position"] = 1
            out.at[idx, "entry_price"] = entry_price
            sl_level = entry_price * (1 - SL_PCENT)
            tp_level = entry_price * (1 + TP_PCENT)
            if current_low <= sl_level:
                out.at[idx, "strategy_return"] = (sl_level - entry_price) / entry_price
                out.at[idx, "position"] = 0
                out.at[idx, "cost"] += SPREAD + FEES_PCENT
                out.at[idx, "exit_price"] = sl_level
                out.at[idx, "exit_reason"] = "stop_loss"
                out.at[idx, "trade_pnl_gross"] = (sl_level - entry_price) / entry_price
                position = 0
                entry_price = np.nan
            elif current_high >= tp_level:
                out.at[idx, "strategy_return"] = (tp_level - entry_price) / entry_price
                out.at[idx, "position"] = 0
                out.at[idx, "cost"] += SPREAD + FEES_PCENT
                out.at[idx, "exit_price"] = tp_level
                out.at[idx, "exit_reason"] = "take_profit"
                out.at[idx, "trade_pnl_gross"] = (tp_level - entry_price) / entry_price
                position = 0
                entry_price = np.nan
            else:
                out.at[idx, "strategy_return"] = (current_close - entry_price) / entry_price

    out["trades"] = out["position"].diff().abs().fillna(0)
    out["strategy_return_net"] = out["strategy_return"] - out["cost"]
    out["equity_curve_net"] = (1 + out["strategy_return_net"]).cumprod()
    out["buy_hold_equity"] = out["close"] / out["close"].iloc[0]
    return out


def calculate_metrics(result: pd.DataFrame) -> dict:
    equity = result["equity_curve_net"]
    returns = equity.pct_change().dropna()
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    days = (equity.index[-1] - equity.index[0]).total_seconds() / 86400
    ann_factor = 365.25 / days if days > 0 else 0
    annual_return = (1 + total_return) ** ann_factor - 1 if days > 0 else 0
    annual_vol = returns.std() * np.sqrt(24 * 365.25)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
    drawdown = equity / equity.cummax() - 1
    max_drawdown = drawdown.min()
    orders = int(round(result["cost"].sum() / (SPREAD + FEES_PCENT)))
    entries = int(((result["cost"] > 0) & (result["entry_price"].notna())).sum())
    exits = result[result["exit_reason"] != ""]
    win_rate = float((exits["trade_pnl_gross"] > 0).mean()) if len(exits) else 0.0

    return {
        "rows": len(result),
        "start": result.index[0].isoformat(),
        "end": result.index[-1].isoformat(),
        "short_window": SHORT_WINDOW,
        "long_window": LONG_WINDOW,
        "sl_pcent": SL_PCENT,
        "tp_pcent": TP_PCENT,
        "cost_per_trade": SPREAD + FEES_PCENT,
        "orders": orders,
        "entries": entries,
        "closed_trades": int(len(exits)),
        "final_position": int(result["position"].iloc[-1]),
        "win_rate_pcent": win_rate * 100,
        "final_equity_net": equity.iloc[-1],
        "total_return_pcent": total_return * 100,
        "annualized_return_pcent": annual_return * 100,
        "annualized_volatility_pcent": annual_vol * 100,
        "sharpe_ratio": sharpe,
        "max_drawdown_pcent": max_drawdown * 100,
        "buy_hold_return_pcent": (result["buy_hold_equity"].iloc[-1] - 1) * 100,
    }


def main() -> None:
    df = load_binance_klines(DATA_PATH)
    result = run_backtest(df)
    metrics = calculate_metrics(result)

    result.to_csv(RESULTS_PATH)
    lines = [f"{key}: {value}" for key, value in metrics.items()]
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
