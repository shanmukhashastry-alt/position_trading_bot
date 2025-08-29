import ccxt
import pandas as pd
import logging
import time

# Hardcoded top-volume USDT pairs
TOP_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT",
    "SOL/USDT", "DOGE/USDT", "MATIC/USDT", "DOT/USDT", "LTC/USDT"
]

TIMEFRAME = "1h"   # can be "15m", "4h", "1d"
LIMIT = 100

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# ✅ Binance MAINNET (not testnet)
exchange = ccxt.binance({
    "enableRateLimit": True,
    "options": {"defaultType": "spot"},
})

def fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=LIMIT):
    """Fetch OHLCV data safely"""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","volume"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        return df
    except Exception as e:
        logging.error(f"Failed to fetch {symbol}: {e}")
        return None

def check_signal(df):
    """Simple signal: close > moving average"""
    if df is None or df.empty:
        return None
    df["ma20"] = df["close"].rolling(20).mean()
    if df["close"].iloc[-1] > df["ma20"].iloc[-1]:
        return "BUY"
    elif df["close"].iloc[-1] < df["ma20"].iloc[-1]:
        return "SELL"
    return "HOLD"

def run_bot():
    logging.info("Bot started ✅")
    for symbol in TOP_SYMBOLS:
        df = fetch_ohlcv(symbol)
        signal = check_signal(df)
        logging.info(f"{symbol} → {signal}")

if __name__ == "__main__":
    while True:
        run_bot()
        time.sleep(60 * 60)  # run every hour
