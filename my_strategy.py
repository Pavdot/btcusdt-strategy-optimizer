import numpy as np
import pandas as pd
from backtesting import Strategy


def _ema(values, period):
    return pd.Series(values).ewm(span=int(period), adjust=False, min_periods=int(period)).mean().to_numpy(copy=True)


def _rsi(values, period):
    close = pd.Series(values)
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_gain = up.ewm(alpha=1 / int(period), adjust=False, min_periods=int(period)).mean()
    avg_loss = down.ewm(alpha=1 / int(period), adjust=False, min_periods=int(period)).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).to_numpy(copy=True)


def _atr(high, low, close, period):
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(int(period), min_periods=int(period)).mean().to_numpy(copy=True)


def _rolling_rank(values, lookback):
    series = pd.Series(values)
    lookback = int(lookback)
    return series.rolling(lookback, min_periods=lookback).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1],
        raw=False,
    ).to_numpy(copy=True)


def _rolling_volume_profile_levels(high, low, close, volume, lookback, bins, node_percentile):
    lookback = int(lookback)
    close = np.asarray(close, dtype=float)
    node_percentile = float(node_percentile)
    series = pd.Series(close)

    # Fast proxy for high-volume nodes. A full rolling histogram per candidate
    # makes grid optimization painfully slow. These quantile bands approximate
    # nearby acceptance/rejection zones while preserving the optimizer knobs.
    band_width = max(0.12, min(0.42, node_percentile / 2))
    support_q = max(0.05, 0.50 - band_width)
    resistance_q = min(0.95, 0.50 + band_width)

    support = series.shift(1).rolling(lookback, min_periods=lookback).quantile(support_q)
    resistance = series.shift(1).rolling(lookback, min_periods=lookback).quantile(resistance_q)
    return support.to_numpy(copy=True), resistance.to_numpy(copy=True)


def _rolling_max(values, lookback):
    return pd.Series(values).shift(1).rolling(int(lookback), min_periods=int(lookback)).max().to_numpy(copy=True)


def _rolling_min(values, lookback):
    return pd.Series(values).shift(1).rolling(int(lookback), min_periods=int(lookback)).min().to_numpy(copy=True)


def _rolling_mean(values, lookback):
    return pd.Series(values).rolling(int(lookback), min_periods=int(lookback)).mean().to_numpy(copy=True)


class MyStrategy(Strategy):
    ema_fast_period = 250
    ema_slow_period = 750
    rsi_period = 14
    atr_period = 14

    vp_tactical_lookback = 336
    vp_macro_lookback = 1000
    vp_bins = 24

    breakout_lookback = 20
    rsi_rank_lookback = 50
    ema_slope_lookback = 20

    min_body_ratio = 0.6
    min_vol_ratio = 0.8
    min_break_dist_atr = 0.02

    min_node_percentile = 0.5
    min_room_to_target_atr = 0.75

    min_rsi_percentile_long = 0.35
    max_rsi_percentile_short = 0.65

    min_close_position_long = 0.55
    max_close_position_short = 0.45

    stop_atr_buffer = 0.5
    tp_fraction_to_next_level = 0.85

    risk_fraction = 0.95

    def init(self):
        high = np.asarray(self.data.High, dtype=float)
        low = np.asarray(self.data.Low, dtype=float)
        close = np.asarray(self.data.Close, dtype=float)
        volume = np.asarray(self.data.Volume, dtype=float)

        self.ema_fast = self.I(_ema, close, self.ema_fast_period, name="ema_fast")
        self.ema_slow = self.I(_ema, close, self.ema_slow_period, name="ema_slow")
        self.rsi = self.I(_rsi, close, self.rsi_period, name="rsi")
        self.atr = self.I(_atr, high, low, close, self.atr_period, name="atr")

        self.breakout_high = self.I(_rolling_max, high, self.breakout_lookback, name="breakout_high")
        self.breakout_low = self.I(_rolling_min, low, self.breakout_lookback, name="breakout_low")
        self.avg_volume = self.I(_rolling_mean, volume, 50, name="avg_volume")
        self.rsi_rank = self.I(_rolling_rank, self.rsi, self.rsi_rank_lookback, name="rsi_rank")

        tactical_support, tactical_resistance = _rolling_volume_profile_levels(
            high,
            low,
            close,
            volume,
            self.vp_tactical_lookback,
            self.vp_bins,
            self.min_node_percentile,
        )
        macro_support, macro_resistance = _rolling_volume_profile_levels(
            high,
            low,
            close,
            volume,
            self.vp_macro_lookback,
            self.vp_bins,
            self.min_node_percentile,
        )
        self.tactical_support = self.I(lambda: tactical_support.copy(), name="tactical_support")
        self.tactical_resistance = self.I(lambda: tactical_resistance.copy(), name="tactical_resistance")
        self.macro_support = self.I(lambda: macro_support.copy(), name="macro_support")
        self.macro_resistance = self.I(lambda: macro_resistance.copy(), name="macro_resistance")

    def _ready(self):
        values = [
            self.ema_fast[-1],
            self.ema_slow[-1],
            self.rsi[-1],
            self.atr[-1],
            self.breakout_high[-1],
            self.breakout_low[-1],
            self.avg_volume[-1],
            self.rsi_rank[-1],
        ]
        return all(np.isfinite(v) for v in values) and self.atr[-1] > 0

    def _candle_quality(self):
        high = self.data.High[-1]
        low = self.data.Low[-1]
        open_ = self.data.Open[-1]
        close = self.data.Close[-1]
        candle_range = high - low
        if candle_range <= 0:
            return 0.0, 0.5
        body_ratio = abs(close - open_) / candle_range
        close_position = (close - low) / candle_range
        return body_ratio, close_position

    def _nearest_levels(self, side):
        close = self.data.Close[-1]
        atr = self.atr[-1]
        if side == 1:
            support_candidates = [self.tactical_support[-1], self.macro_support[-1]]
            target_candidates = [self.tactical_resistance[-1], self.macro_resistance[-1]]
            supports = [x for x in support_candidates if np.isfinite(x) and x < close]
            targets = [x for x in target_candidates if np.isfinite(x) and x > close]
            support = max(supports) if supports else close - 1.5 * atr
            target = min(targets) if targets else close + 2.0 * atr
        else:
            resistance_candidates = [self.tactical_resistance[-1], self.macro_resistance[-1]]
            target_candidates = [self.tactical_support[-1], self.macro_support[-1]]
            resistances = [x for x in resistance_candidates if np.isfinite(x) and x > close]
            targets = [x for x in target_candidates if np.isfinite(x) and x < close]
            support = min(resistances) if resistances else close + 1.5 * atr
            target = max(targets) if targets else close - 2.0 * atr
        return support, target

    def next(self):
        if not self._ready():
            return

        close = self.data.Close[-1]
        volume = self.data.Volume[-1]
        atr = self.atr[-1]
        body_ratio, close_position = self._candle_quality()

        if body_ratio < self.min_body_ratio:
            return
        if volume < self.avg_volume[-1] * self.min_vol_ratio:
            return

        trend_up = self.ema_fast[-1] > self.ema_slow[-1]
        trend_down = self.ema_fast[-1] < self.ema_slow[-1]
        slope = self.ema_fast[-1] - self.ema_fast[-1 - int(self.ema_slope_lookback)]
        break_dist = self.min_break_dist_atr * atr

        long_breakout = close > self.breakout_high[-1] + break_dist
        short_breakout = close < self.breakout_low[-1] - break_dist

        if self.position:
            if self.position.is_long and (not trend_up or short_breakout):
                self.position.close()
            elif self.position.is_short and (not trend_down or long_breakout):
                self.position.close()
            return

        if (
            trend_up
            and slope > 0
            and long_breakout
            and self.rsi_rank[-1] >= self.min_rsi_percentile_long
            and close_position >= self.min_close_position_long
        ):
            support, target = self._nearest_levels(side=1)
            room = (target - close) / atr
            if room < self.min_room_to_target_atr:
                return
            sl = min(support - self.stop_atr_buffer * atr, close - 0.5 * atr)
            tp = close + self.tp_fraction_to_next_level * (target - close)
            if sl < close < tp:
                self.buy(size=self.risk_fraction, sl=sl, tp=tp)

        elif (
            trend_down
            and slope < 0
            and short_breakout
            and self.rsi_rank[-1] <= self.max_rsi_percentile_short
            and close_position <= self.max_close_position_short
        ):
            resistance, target = self._nearest_levels(side=-1)
            room = (close - target) / atr
            if room < self.min_room_to_target_atr:
                return
            sl = max(resistance + self.stop_atr_buffer * atr, close + 0.5 * atr)
            tp = close - self.tp_fraction_to_next_level * (close - target)
            if tp < close < sl:
                self.sell(size=self.risk_fraction, sl=sl, tp=tp)
