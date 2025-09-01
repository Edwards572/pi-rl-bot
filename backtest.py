import os
import argparse
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from data_feed import fetch_history
from baseline_rules import make_signals, pips_to_price, INSTRUMENT

load_dotenv()

LOOKBACK = int(os.getenv("LOOKBACK_CANDLES", "1000"))
SPREAD_PIPS = float(os.getenv("SPREAD_PIPS", "0.8"))
SLIPPAGE_PIPS = float(os.getenv("SLIPPAGE_PIPS", "0.2"))

def apply_costs(price: float, side: str, instrument: str) -> float:
    """Apply spread + slippage to a fill price (very simple cost model)."""
    spread = pips_to_price(SPREAD_PIPS, instrument)
    slip   = pips_to_price(SLIPPAGE_PIPS, instrument)
    if side == "long":
        return price + spread + slip
    return price - spread - slip

def baseline_bt(inst: str) -> pd.DataFrame:
    """Run the baseline breakout strategy over recent history and return trade rows."""
    df = fetch_history(inst, lookback=LOOKBACK)
    if df.empty:
        raise SystemExit("No data fetched.")
    trades = make_signals(df, instrument=inst)

    rows = []
    for tr in trades:
        entry_costed = apply_costs(tr.entry, tr.side, inst)
        # exiting a long is a short fill (and vice versa) for the simple cost model
        exit_side = "short" if tr.side == "long" else "long"
        exit_costed = apply_costs(tr.exit, exit_side, inst)
        pnl = (exit_costed - entry_costed) if tr.side == "long" else (entry_costed - exit_costed)
        rows.append({
            "time": tr.time,
            "exit_time": tr.exit_time,
            "side": tr.side,
            "entry": tr.entry,
            "exit": tr.exit,
            "exit_reason": tr.exit_reason,
            "pnl": pnl,
        })
    return pd.DataFrame(rows)

def metrics(df_tr: pd.DataFrame) -> dict:
    if df_tr.empty:
        return {"trades": 0, "pnl": 0.0, "sharpe": 0.0, "max_dd": 0.0, "hit": 0.0, "pf": 0.0}
    pnl_curve = df_tr["pnl"].cumsum()
    ret = df_tr["pnl"].fillna(0.0)
    sharpe = (ret.mean() / (ret.std() + 1e-9)) * np.sqrt(252)
    peak = pnl_curve.cummax()
    dd = (pnl_curve - peak)
    max_dd = float(dd.min())
    wins = int((df_tr["pnl"] > 0).sum())
    losses = int((df_tr["pnl"] < 0).sum())
    pf = (df_tr[df_tr["pnl"] > 0]["pnl"].sum() /
          abs(df_tr[df_tr["pnl"] < 0]["pnl"].sum())) if losses > 0 else float("inf")
    return {
        "trades": int(len(df_tr)),
        "pnl": float(df_tr["pnl"].sum()),
        "sharpe": float(sharpe),
        "max_dd": max_dd,
        "hit": float(wins / max(1, (wins + losses))),
        "pf": float(pf),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instrument", default=INSTRUMENT)
    args = ap.parse_args()

    df_tr = baseline_bt(args.instrument)
    print(df_tr.tail(5))
    m = metrics(df_tr)
    print("\nRESULTS:")
    for k, v in m.items():
        print(f"{k}: {v}")

    out = f"backtest_BASELINE_{args.instrument}.csv"
    df_tr.to_csv(out, index=False)
    print(f"Saved trades -> {out}")

if __name__ == "__main__":
    main()
