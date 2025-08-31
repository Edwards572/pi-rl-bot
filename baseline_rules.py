import os
from dataclasses import dataclass
from typing import List, Optional
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ENV-configurable params (reasonable defaults)
BUFFER_PIPS = float(os.getenv("BRK_BUFFER_PIPS", "1.0"))      # buffer above/below range
SL_PIPS = float(os.getenv("BRK_SL_PIPS", "8.0"))              # initial stop loss
TP_PIPS = float(os.getenv("BRK_TP_PIPS", "12.0"))             # take profit
SESSION_START = os.getenv("SESSION_START_HHMM", "0700")       # e.g., 0700 (UTC)
SESSION_END = os.getenv("SESSION_END_HHMM", "1700")           # e.g., 1700 (UTC)
FIRST_WINDOW_MIN = int(os.getenv("FIRST_WINDOW_MIN", "30"))   # minutes for initial range
INSTRUMENT = os.getenv("INSTRUMENT_BASELINE", os.getenv("INSTRUMENTS", "EUR_USD").split(",")[0].strip())

# Breakeven controls
BE_TRIGGER_PIPS = float(os.getenv("BE_TRIGGER_PIPS", "6.0"))   # move SL to BE after this many pips in profit
LOCK_PROFIT_PIPS = float(os.getenv("LOCK_PROFIT_PIPS", "0.0")) # lock +X pips beyond BE

def pips_to_price(pips: float, instrument: str) -> float:
    if instrument.endswith("_JPY"):
        return pips * 0.01
    return pips * 0.0001

@dataclass
class Trade:
    time: pd.Timestamp
    side: str               # 'long' or 'short'
    entry: float
    sl: float
    tp: float
    exit: Optional[float] = None
    exit_time: Optional[pd.Timestamp] = None
    exit_reason: Optional[str] = None

def _hhmm_to_minutes(hhmm: str) -> int:
    return int(hhmm[:2]) * 60 + int(hhmm[2:])

def _in_session(ts: pd.Timestamp) -> bool:
    tsu = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
    minutes = tsu.hour * 60 + tsu.minute
    return _hhmm_to_minutes(SESSION_START) <= minutes <= _hhmm_to_minutes(SESSION_END)

def _session_day_key(ts: pd.Timestamp) -> str:
    tsu = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
    return tsu.strftime("%Y-%m-%d")

def _first_window_mask(df: pd.DataFrame) -> pd.Series:
    keys = df.index.map(_session_day_key)
    first_mask = pd.Series(False, index=df.index)
    for day in pd.unique(keys):
        day_idx = (keys == day) & df.index.map(_in_session)
        day_times = df.index[day_idx]
        if len(day_times) == 0:
            continue
        start_time = day_times[0]
        cutoff = start_time + pd.Timedelta(minutes=FIRST_WINDOW_MIN)
        first_mask |= (df.index >= start_time) & (df.index < cutoff)
    return first_mask

def make_signals(df: pd.DataFrame, instrument: str = INSTRUMENT) -> List[Trade]:
    """Video #1 baseline with buffer + SL/TP + breakeven (never loosens)."""
    df = df.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    in_sess = df.index.map(_in_session)
    df = df[in_sess]
    if df.empty:
        return []

    first_mask = _first_window_mask(df)

    trades: List[Trade] = []
    day_keys = df.index.map(_session_day_key)

    for day in pd.unique(day_keys):
        day_df = df[day_keys == day]
        if day_df.empty:
            continue
        fw = day_df[first_mask.loc[day_df.index]]
        if fw.empty:
            continue
        rng_high = fw["high"].max()
        rng_low = fw["low"].min()
        buf = pips_to_price(BUFFER_PIPS, instrument)
        long_trig = rng_high + buf
        short_trig = rng_low - buf

        entered = False
        moved_be = False
        high_water = None
        low_water = None

        for t, row in day_df.iterrows():
            if not entered:
                if row["close"] > long_trig:
                    entry = row["close"]
                    sl = entry - pips_to_price(SL_PIPS, instrument)
                    tp = entry + pips_to_price(TP_PIPS, instrument)
                    trade = Trade(time=t, side="long", entry=entry, sl=sl, tp=tp)
                    entered = True
                    high_water = row["high"]
                elif row["close"] < short_trig:
                    entry = row["close"]
                    sl = entry + pips_to_price(SL_PIPS, instrument)
                    tp = entry - pips_to_price(TP_PIPS, instrument)
                    trade = Trade(time=t, side="short", entry=entry, sl=sl, tp=tp)
                    entered = True
                    low_water = row["low"]
                else:
                    continue
            else:
                if trade.side == "long":
                    high_water = max(high_water, row["high"]) if high_water is not None else row["high"]
                    be_trigger_price = trade.entry + pips_to_price(BE_TRIGGER_PIPS, instrument)
                    if not moved_be and high_water >= be_trigger_price:
                        new_sl = trade.entry + pips_to_price(LOCK_PROFIT_PIPS, instrument)
                        trade.sl = max(trade.sl, new_sl)
                        moved_be = True
                    if row["low"] <= trade.sl:
                        trade.exit = trade.sl
                        trade.exit_time = t
                        trade.exit_reason = "SL"
                        trades.append(trade)
                        break
                    if row["high"] >= trade.tp:
                        trade.exit = trade.tp
                        trade.exit_time = t
                        trade.exit_reason = "TP"
                        trades.append(trade)
                        break
                else:  # short
                    low_water = min(low_water, row["low"]) if low_water is not None else row["low"]
                    be_trigger_price = trade.entry - pips_to_price(BE_TRIGGER_PIPS, instrument)
                    if not moved_be and low_water <= be_trigger_price:
                        new_sl = trade.entry - pips_to_price(LOCK_PROFIT_PIPS, instrument)
                        trade.sl = min(trade.sl, new_sl)
                        moved_be = True
                    if row["high"] >= trade.sl:
                        trade.exit = trade.sl
                        trade.exit_time = t
                        trade.exit_reason = "SL"
                        trades.append(trade)
                        break
                    if row["low"] <= trade.tp:
                        trade.exit = trade.tp
                        trade.exit_time = t
                        trade.exit_reason = "TP"
                        trades.append(trade)
                        break

        if entered and trade.exit is None:
            last_t = day_df.index[-1]
            trade.exit = day_df["close"].iloc[-1]
            trade.exit_time = last_t
            trade.exit_reason = "EOD"
            trades.append(trade)

    return trades
