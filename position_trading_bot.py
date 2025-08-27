import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime

# ==== CONFIG ====
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")  # keep as string; Telegram accepts string/int
INTERVAL_1H = "1h"
INTERVAL_4H = "4h"
LOOKBACK_CANDLES = 100
LOG_FILE = "hourly_position_signals.csv"
last_alert_time = {}

# Strategy Parameters
ATR_MULTIPLIER_SL = 2.0
ATR_MULTIPLIER_TP = 3.5
MOMENTUM_THRESHOLD = 0.01
MIN_TREND_STRENGTH = 0.5

BINANCE_BASE = "https://api.binance.com"

# ==== UTILITIES ====
def _send_telegram_message_safe(msg: str):
    """Send a Telegram message but never crash the run if Telegram fails/missing env."""
    if not BOT_TOKEN or not CHAT_ID:
        print("⚠️ BOT_TOKEN/CHAT_ID missing; skipping Telegram send.")
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code != 200:
            print("[ERROR] Telegram send error:", r.text)
        else:
            print("✅ Telegram message sent")
    except Exception as e:
        print("[ERROR] Telegram send exception:", e)

def _get_json_with_retries(url: str, params=None, expect_type=list, max_retries=4, base_sleep=1.0):
    """
    GET JSON with retries + validation.
    - expect_type: `list` for Binance endpoints that return arrays
    """
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=20)
            # Raise on non-2xx
            resp.raise_for_status()
            data = resp.json()

            # Binance sometimes returns {"code": ..., "msg": "..."} on errors/rate limits.
            if isinstance(data, dict) and "code" in data and "msg" in data:
                raise ValueError(f"Binance error: {data}")

            if expect_type is not None and not isinstance(data, expect_type):
                raise TypeError(f"Unexpected JSON type. Expected {expect_type.__name__}, got {type(data).__name__}: {str(data)[:200]}")

            return data

        except Exception as e:
            last_err = e
            sleep_s = base_sleep * (2 ** (attempt - 1))  # exponential backoff
            print(f"[WARN] GET {url} failed (attempt {attempt}/{max_retries}): {e} -> retrying in {sleep_s:.1f}s")
            time.sleep(sleep_s)

    # If we ran out of retries, raise a concise error
    raise RuntimeError(f"Failed to fetch {url} after {max_retries} attempts: {last_err}")

# ==== TELEGRAM ====
def send_telegram_message(msg):
    _send_telegram_message_safe(msg)

# ==== GET SYMBOLS ====
def get_top_20_symbols():
    url = f"{BINANCE_BASE}/api/v3/ticker/24hr"
    data = _get_json_with_retries(url, expect_type=list)
    df = pd.DataFrame(data)

    # Ensure required columns exist
    if "symbol" not in df.columns or "quoteVolume" not in df.columns:
        raise ValueError("Unexpected /ticker/24hr schema from Binance.")

    # Filter + sort
    df = df[df["symbol"].str.endswith("USDT")].copy()
    # Some rows may have non-numeric quoteVolume during incidents; coerce safely
    df["quoteVolume"] = pd.to_numeric(df["quoteVolume"], errors="coerce")
    df = df.dropna(subset=["quoteVolume"]).sort_values("quoteVolume", ascending=False).head(20)
    symbols = df["symbol"].tolist()

    if not symbols:
        raise ValueError("No USDT symbols obtained from Binance 24hr ticker.")
    return symbols

# ==== GET CANDLES ====
def get_klines(symbol, interval, limit):
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": int(limit)}
    data = _get_json_with_retries(url, params=params, expect_type=list)

    if not data or not isinstance(data, list) or not isinstance(data[0], list):
        raise ValueError(f"Unexpected klines response for {symbol} {interval}: {str(data)[:200]}")

    cols = ['time','o','h','l','c','v','ct','qv','n','tbbav','tbqv','ignore']
    df = pd.DataFrame(data, columns=cols[:len(data[0])])  # be len-safe

    # Cast numeric cols cautiously
    for col in ['o', 'h', 'l', 'c', 'v']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], unit='ms', errors="coerce")

    # Basic sanity checks
    df = df.dropna(subset=['c'])
    if len(df) < 50:
        raise ValueError(f"Not enough candles for {symbol} {interval} (got {len(df)}).")
    return df

# ==== INDICATORS ====
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

# ==== LOG TRADES ====
def log_hourly_trade(symbol, direction, price, sl, tp, conf_level, confidence, trend, timestamp, risk_reward, adx, position_size):
    entry = {
        "time": timestamp, "symbol": symbol, "direction": direction,
        "price": price, "stop_loss": sl, "take_profit": tp,
        "confidence": conf_level, "confidence_score": confidence,
        "trend_4h": trend, "risk_reward": risk_reward, "adx": adx,
        "position_size": position_size
    }
    df = pd.DataFrame([entry])  # <-- list of dicts avoids scalar DataFrame error
    file_exists = os.path.exists(LOG_FILE)
    df.to_csv(LOG_FILE, mode='a', header=not file_exists, index=False)
    print(f"[DEBUG] Logged trade for {symbol}")

# ==== CHECK SIGNAL ====
def check_hourly_signal(symbol):
    df1h = get_klines(symbol, INTERVAL_1H, LOOKBACK_CANDLES)
    df4h = get_klines(symbol, INTERVAL_4H, LOOKBACK_CANDLES)

    close = df1h['c']
    latest_time = df1h['time'].iloc[-1]

    ema_21, ema_50, sma_200 = ema(close, 21), ema(close, 50), sma(close, 200)
    rsi_val = rsi(close, 14)
    macd_line, signal_line, _ = macd(close)
    atr_val = calculate_atr(df1h)
    adx_val, plus_di, minus_di = adx(df1h)
    momentum = momentum_oscillator(df1h)
    vw_momentum = volume_weighted_momentum(df1h)
    trend_4h, trend_strength = trend_strength_4h(df4h)

    close_last, ema_21_last, ema_50_last, sma_200_last = close.iloc[-1], ema_21.iloc[-1], ema_50.iloc[-1], sma_200.iloc[-1]
    rsi_last, macd_last, signal_last = rsi_val.iloc[-1], macd_line.iloc[-1], signal_line.iloc[-1]

    confidence, direction = 0.0, None
    returns = close.pct_change().dropna()
    if len(returns) < 20:
        print(f"[INFO] Not enough returns history for {symbol}")
        return None
    volatility = returns.rolling(20).std().iloc[-1]
    position_multiplier = dynamic_position_sizing(volatility)

    # BUY
    if (ema_21_last > ema_50_last and close_last > sma_200_last and 40 < rsi_last < 75 and adx_val > 25):
        confidence += 1
        if macd_last > signal_last and macd_last > 0: confidence += 1
        if momentum > MOMENTUM_THRESHOLD: confidence += 1
        if vw_momentum > 0: confidence += 1
        if trend_4h in ["up", "strong_up"] and trend_strength > MIN_TREND_STRENGTH: confidence += 1
        if plus_di > minus_di and adx_val > 30: confidence += 0.5
        direction = "BUY"

    # SELL
    elif (ema_21_last < ema_50_last and close_last < sma_200_last and 25 < rsi_last < 60 and adx_val > 25):
        confidence += 1
        if macd_last < signal_last and macd_last < 0: confidence += 1
        if momentum < -MOMENTUM_THRESHOLD: confidence += 1
        if vw_momentum < 0: confidence += 1
        if trend_4h in ["down", "strong_down"] and trend_strength > MIN_TREND_STRENGTH: confidence += 1
        if minus_di > plus_di and adx_val > 30: confidence += 0.5
        direction = "SELL"

    if direction and confidence >= 4:
        if last_alert_time.get(symbol) == latest_time:
            print(f"[INFO] Duplicate alert skipped for {symbol}")
            return None
        last_alert_time[symbol] = latest_time

        if direction == "BUY":
            sl = round(close_last - (atr_val * ATR_MULTIPLIER_SL), 6)
            tp = round(close_last + (atr_val * ATR_MULTIPLIER_TP), 6)
        else:
            sl = round(close_last + (atr_val * ATR_MULTIPLIER_SL), 6)
            tp = round(close_last - (atr_val * ATR_MULTIPLIER_TP), 6)

        conf_level = "Very High" if confidence >= 5.5 else "High" if confidence >= 4.5 else "Medium"
        risk_reward = abs((tp - close_last) / (close_last - sl)) if direction == "BUY" else abs((close_last - tp) / (sl - close_last))

        # Log trade to CSV (safe)
        try:
            log_hourly_trade(
                symbol, direction, close_last, sl, tp, conf_level, confidence,
                trend_4h, latest_time, risk_reward, adx_val, position_multiplier
            )
        except Exception as e:
            print("[WARN] Failed to log CSV:", e)

        msg = (f"*{direction} POSITION* — `{symbol}` @ {close_last}\n"
               f"SL: `{sl}` | TP: `{tp}`\n"
               f"Risk:Reward = 1:{risk_reward:.2f}\n"
               f"Position Size: {position_multiplier:.2f}x\n"
               f"Confidence: *{conf_level}* ({confidence:.1f}/6)\n"
               f"Trend 4H: *{trend_4h}* | ADX: {adx_val:.1f}\n"
               f"Timeframe: 1H | {latest_time.strftime('%Y-%m-%d %H:%M')}")
        return msg

    print(f"[INFO] No valid 1H signal for {symbol}")
    return None

# ==== MAIN (single run) ====
if __name__ == "__main__":
    print("🚀 Bot Run Started")
    try:
        symbols = get_top_20_symbols()
        found_signal = False
        for sym in symbols:
            try:
                signal = check_hourly_signal(sym)
            except Exception as e:
                # Per-symbol failure should not kill the whole run
                print(f"[WARN] Symbol {sym} failed: {e}")
                continue

            if signal:
                send_telegram_message(signal)
                print(f"[ALERT] {signal}")
                found_signal = True

        if not found_signal:
            print("[INFO] No signals found in this run.")

    except Exception as e:
        err_msg = f"🔴 1H Bot Error: {str(e)[:180]}"
        print("[ERROR]", err_msg)
        _send_telegram_message_safe(err_msg)

    print("✅ Bot run finished")
