import os
import time
import pandas as pd
from dotenv import load_dotenv
import oandapyV20
from oandapyV20.endpoints.instruments import InstrumentsCandles

load_dotenv()

API_KEY     = os.getenv("OANDA_API_KEY", "")
ENV         = os.getenv("OANDA_ENV", "practice")  # practice or live (use practice)
INSTRUMENTS = [s.strip() for s in os.getenv("INSTRUMENTS", "EUR_USD").split(",")]
GRANULARITY = os.getenv("GRANULARITY", "M5")
LOOKBACK    = int(os.getenv("LOOKBACK_CANDLES", "300"))

api = None
if API_KEY:
    api = oandapyV20.API(access_token=API_KEY, environment=ENV)

def fetch_candles(instrument: str, count: int = 100, granularity: str = GRANULARITY) -> pd.DataFrame:
    """Fetch up to 500 recent, completed candles for an instrument."""
    if api is None:
        raise RuntimeError("OANDA_API_KEY missing. Set it in .env")
    params = {"granularity": granularity, "count": min(count, 500), "price": "M"}
    r = InstrumentsCandles(instrument=instrument, params=params)
    api.request(r)
    rows = []
    for c in r.response["candles"]:
        if not c["complete"]:
            continue
        mid = c["mid"]
        rows.append([c["time"], float(mid["o"]), float(mid["h"]), float(mid["l"]), float(mid["c"])])
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close"]).dropna()
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    return df

def fetch_history(instrument: str, lookback: int = LOOKBACK, granularity: str = GRANULARITY) -> pd.DataFrame:
    """Fetch up to `lookback` candles, in 500-sized batches."""
    out = []
    remaining = lookback
    while remaining > 0:
        cnt = min(remaining, 500)
        df = fetch_candles(instrument, count=cnt, granularity=granularity)
        if df.empty:
            break
        out.append(df)
        remaining -= len(df)
        time.sleep(0.1)  # be gentle
        if len(df) < cnt:
            break
    if not out:
        return pd.DataFrame()
    hist = pd.concat(out).sort_index()
    return hist.tail(lookback)

if __name__ == "__main__":
    inst = INSTRUMENTS[0]
    df = fetch_history(inst, lookback=100)
    print(inst, df.tail())
