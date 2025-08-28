# position_trading_bot.py
"""
Position trading bot — single-run version using CCXT.
Run this once (e.g., via GitHub Actions every 30 min). It fetches top-20 USDT pairs,
evaluates 1H signals, sends Telegram alerts, logs to CSV, and exits.
"""

import os
import time
import math
import requests
import logging
from datetime import datetime

import ccxt
import pandas as pd
import numpy as np

# -------------------------
# Configuration
# -------------------------
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")  # string or int ok
INTERVAL_1H = "1h"
INTERVAL_4H = "4h"
LOOKBACK_CANDLES = 200
LOG_FILE = "hourly_position_signals.csv"

# Strategy parameters (same as your earlier config)
ATR_MULTIPLIER_SL = 2.0
ATR_MULTIPLIER_TP = 3.5
MOMENTUM_THRESHOLD = 0.01
MIN_TREND_STRENGTH = 0.5

TOP_N = 20  # top N USDT pairs by quoteVolume

# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# -------------------------
# Helpers
# -------------------------
def safe_dataframe(data):
    """Return a pandas DataFrame safely from dict or list-of-dicts."""
    if isinstance(data, dict):
        return pd.DataFrame([data])
    return pd.DataFrame(data)

def send_telegram_message(msg: str):
    if not BOT_TOKEN or not CHAT_ID:
        logging.warning("BOT_TOKEN or CHAT_ID not set — skipping Telegram send.")
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code != 200:
            logging.error("Telegram error: %s", r.text)
        else:
            logging.info("Sent Telegram message.")
    except Exception as e:
        logging.exception("Telegram send failed: %s", e)

# -------------------------
# CCXT Exchange init
# -------------------------
import ccxt

exchange = ccxt.binance({
    "enableRateLimit": True,
    "options": {"defaultType": "spot"},
    "urls": {"api": "https://testnet.binance.vision"}
})
# If you face specific regional blocks, consider adding proxy config to exchange.options['proxies']

# -------------------------
# Market Data functions (ccxt)
# -------------------------
def get_top_n_usdt_symbols(n=TOP_N):
    """Return top-n USDT symbols by quoteVolume using ccxt.fetch_tickers."""
    try:
        tickers = exchange.fetch_tickers()  # may be heavy but returns all tickers
    except Exception as e:
        logging.exception("fetch_tickers failed: %s", e)
        return []

    # Build list of dicts with symbol and quoteVolume
    records = []
    for sym, info in tickers.items():
        try:
            # ccxt symbol format: "BTC/USDT"
            if not sym.endswith("/USDT"):
                continue
            qv = info.get("quoteVolume", None)
            # sometimes ccxt returns nested dicts; convert safely
            qv = float(qv) if qv not in (None, "") else 0.0
            records.append({"symbol": sym, "quoteVolume": qv})
        except Exception:
            continue

    df = safe_dataframe(records)
    if df.empty:
        return []

    df = df.sort_values("quoteVolume", ascending=False).head(n)
    symbols = df["symbol"].tolist()
    logging.info("Top symbols: %s", symbols)
    return symbols

def get_ohlcv_df(symbol_ccxt, timeframe="1h", limit=LOOKBACK_CANDLES):
    """
    Fetch OHLCV via ccxt for symbol like 'BTC/USDT'.
    Returns DataFrame with cols: time,o,h,l,c,v (time as datetime)
    """
    try:
        ohlcv = exchange.fetch_ohlcv(symbol_ccxt, timeframe=timeframe, limit=limit)
    except Exception as e:
        logging.exception("fetch_ohlcv failed for %s: %s", symbol_ccxt, e)
        return pd.DataFrame()

    if not ohlcv:
        return pd.DataFrame()

    df = pd.DataFrame(ohlcv, columns=["time", "o", "h", "l", "c", "v"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    for col in ["o", "h", "l", "c", "v"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["c"])
    return df

# -------------------------
# Indicators (same logic as your original)
# -------------------------
def ema(series, period): return series.ewm(span=period, adjust=False).mean()
def sma(series, period): return series.rolling(window=period).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def adx(df, period=14):
    high, low, close = df['h'], df['l'], df['c']
    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    tr_sum = tr.rolling(period).sum().replace(0, np.nan)
    plus_di = 100 * (plus_dm.rolling(period).sum() / tr_sum)
    minus_di = 100 * (minus_dm.rolling(period).sum() / tr_sum)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = dx.rolling(period).mean()
    return adx_val.iloc[-1], plus_di.iloc[-1], minus_di.iloc[-1]

def calculate_atr(df, period=14):
    hl = df['h'] - df['l']
    hc = (df['h'] - df['c'].shift()).abs()
    lc = (df['l'] - df['c'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean().iloc[-1]

def momentum_oscillator(df, period=10):
    close = df['c']
    momentum = (close / close.shift(period) - 1) * 100
    return momentum.iloc[-1]

def volume_weighted_momentum(df, period=20):
    close = df['c']; volume = df['v']
    vwap = (close * volume).rolling(period).sum() / volume.rolling(period).sum()
    return (close.iloc[-1] / vwap.iloc[-1] - 1) * 100

def trend_strength_4h(df):
    close = df['c']
    ema_20, ema_50, ema_100 = ema(close, 20), ema(close, 50), ema(close, 100)
    if ema_20.iloc[-1] > ema_50.iloc[-1] > ema_100.iloc[-1]:
        return "strong_up", 1.0
    elif ema_20.iloc[-1] > ema_50.iloc[-1]:
        return "up", 0.7
    elif ema_20.iloc[-1] < ema_50.iloc[-1] < ema_100.iloc[-1]:
        return "strong_down", 1.0
    elif ema_20.iloc[-1] < ema_50.iloc[-1]:
        return "down", 0.7
    else:
        return "sideways", 0.3

def dynamic_position_sizing(volatility, base_size=1.0):
    if volatility > 0.05: return base_size * 0.5
    elif volatility > 0.03: return base_size * 0.75
    else: return base_size

# -------------------------
# Check signal (adapted from your original)
# -------------------------
last_alert_time = {}  # in-memory dedupe per run

def check_hourly_signal(symbol_ccxt):
    """
    symbol_ccxt is like 'BTC/USDT'
    returns message string (Markdown) or None
    """
    # get data
    df1h = get_ohlcv_df(symbol_ccxt, timeframe="1h", limit=LOOKBACK_CANDLES)
    df4h = get_ohlcv_df(symbol_ccxt, timeframe="4h", limit=LOOKBACK_CANDLES)
    if df1h.empty or df4h.empty:
        logging.info("Insufficient data for %s", symbol_ccxt)
        return None

    close = df1h['c']
    latest_time = df1h['time'].iloc[-1]

    # indicators
    ema_21, ema_50, sma_200 = ema(close, 21), ema(close, 50), sma(close, 200)
    rsi_val = rsi(close, 14)
    macd_line, signal_line, _ = macd(close)
    atr_val = calculate_atr(df1h)
    try:
        adx_val, plus_di, minus_di = adx(df1h)
    except Exception:
        adx_val, plus_di, minus_di = 0.0, 0.0, 0.0
    momentum = momentum_oscillator(df1h)
    vw_momentum = volume_weighted_momentum(df1h)
    trend_4h, trend_strength = trend_strength_4h(df4h)

    close_last = close.iloc[-1]
    ema_21_last = ema_21.iloc[-1]
    ema_50_last = ema_50.iloc[-1]
    sma_200_last = sma_200.iloc[-1]
    rsi_last = rsi_val.iloc[-1]
    macd_last = macd_line.iloc[-1]
    signal_last = signal_line.iloc[-1]

    confidence = 0.0
    direction = None

    returns = close.pct_change().dropna()
    if len(returns) < 20:
        logging.info("Not enough returns for %s", symbol_ccxt)
        return None
    volatility = returns.rolling(20).std().iloc[-1]
    position_multiplier = dynamic_position_sizing(volatility)

    # BUY conditions
    if (ema_21_last > ema_50_last and close_last > sma_200_last and 40 < rsi_last < 75 and adx_val > 25):
        confidence += 1
        if macd_last > signal_last and macd_last > 0:
            confidence += 1
        if momentum > MOMENTUM_THRESHOLD:
            confidence += 1
        if vw_momentum > 0:
            confidence += 1
        if trend_4h in ["up", "strong_up"] and trend_strength > MIN_TREND_STRENGTH:
            confidence += 1
        if plus_di > minus_di and adx_val > 30:
            confidence += 0.5
        direction = "BUY"

    # SELL conditions
    elif (ema_21_last < ema_50_last and close_last < sma_200_last and 25 < rsi_last < 60 and adx_val > 25):
        confidence += 1
        if macd_last < signal_last and macd_last < 0:
            confidence += 1
        if momentum < -MOMENTUM_THRESHOLD:
            confidence += 1
        if vw_momentum < 0:
            confidence += 1
        if trend_4h in ["down", "strong_down"] and trend_strength > MIN_TREND_STRENGTH:
            confidence += 1
        if minus_di > plus_di and adx_val > 30:
            confidence += 0.5
        direction = "SELL"

    if direction and confidence >= 4:
        # dedupe per-run
        key = f"{symbol_ccxt}-{latest_time}"
        if last_alert_time.get(key):
            logging.info("Duplicate alert for %s skipped", key)
            return None
        last_alert_time[key] = True

        if direction == "BUY":
            sl = round(close_last - (atr_val * ATR_MULTIPLIER_SL), 6)
            tp = round(close_last + (atr_val * ATR_MULTIPLIER_TP), 6)
        else:
            sl = round(close_last + (atr_val * ATR_MULTIPLIER_SL), 6)
            tp = round(close_last - (atr_val * ATR_MULTIPLIER_TP), 6)

        conf_level = "Very High" if confidence >= 5.5 else "High" if confidence >= 4.5 else "Medium"
        risk_reward = None
        try:
            if direction == "BUY":
                risk_reward = abs((tp - close_last) / (close_last - sl))
            else:
                risk_reward = abs((close_last - tp) / (sl - close_last))
        except Exception:
            risk_reward = float("nan")

        # Prepare message (replace '/' so Telegram displays nicely, keep symbol as e.g., BTC/USDT)
        msg = (
            f"*{direction} POSITION* — `{symbol_ccxt}` @ {close_last}\n"
            f"SL: `{sl}` | TP: `{tp}`\n"
            f"Risk:Reward = 1:{(risk_reward if not (risk_reward is None or math.isnan(risk_reward)) else 0):.2f}\n"
            f"Position Size: {position_multiplier:.2f}x\n"
            f"Confidence: *{conf_level}* ({confidence:.1f}/6)\n"
            f"Trend 4H: *{trend_4h}* | ADX: {adx_val:.1f}\n"
            f"Timeframe: 1H | {latest_time.strftime('%Y-%m-%d %H:%M')}"
        )

        # Log to CSV (safe)
        try:
            entry = {
                "time": latest_time,
                "symbol": symbol_ccxt,
                "direction": direction,
                "price": close_last,
                "stop_loss": sl,
                "take_profit": tp,
                "confidence": conf_level,
                "confidence_score": confidence,
                "trend_4h": trend_4h,
                "risk_reward": risk_reward,
                "adx": adx_val,
                "position_size": position_multiplier,
            }
            df_entry = safe_dataframe(entry)
            header = not os.path.exists(LOG_FILE)
            df_entry.to_csv(LOG_FILE, mode="a", header=header, index=False)
        except Exception as e:
            logging.exception("Failed to log CSV for %s: %s", symbol_ccxt, e)

        return msg

    return None

# -------------------------
# Main single-run
# -------------------------
def main():
    logging.info("Position trading bot: single-run start")
    try:
        top_symbols = get_top_n_usdt_symbols(TOP_N)
        if not top_symbols:
            logging.error("No top symbols retrieved — aborting run.")
            send_telegram_message("🔴 Bot Error: No top symbols retrieved.")
            return

        found_any = False
        for sym in top_symbols:
            try:
                msg = check_hourly_signal(sym)
                if msg:
                    send_telegram_message(msg)
                    found_any = True
                    logging.info("Alert sent for %s", sym)
                else:
                    logging.debug("No signal for %s", sym)
            except Exception as e:
                logging.exception("Error processing %s: %s", sym, e)
                # continue to next symbol

        if not found_any:
            logging.info("No 1H position signals found in this run.")
    except Exception as e:
        logging.exception("Top-level bot error: %s", e)
        send_telegram_message(f"🔴 1H Bot Error: {str(e)[:200]}")
    finally:
        logging.info("Bot run finished")

if __name__ == "__main__":
    main()
