import requests
import pandas as pd
import numpy as np
import os
from datetime import datetime

# ==== CONFIG ====
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
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


# ==== TELEGRAM ====
def send_telegram_message(msg):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code != 200:
            print("[ERROR] Telegram send error:", r.text)
        else:
            print("✅ Telegram message sent")
    except Exception as e:
        print("[ERROR] Telegram send exception:", e)


# ==== GET SYMBOLS ====
def get_top_20_symbols():
    url = "https://api.binance.com/api/v3/ticker/24hr"
    data = requests.get(url).json()
    df = pd.DataFrame(data)
    df['quoteVolume'] = df['quoteVolume'].astype(float)
    df = df[df['symbol'].str.endswith('USDT')]
    df = df.sort_values('quoteVolume', ascending=False).head(20)
    return df['symbol'].tolist()


# ==== GET CANDLES ====
def get_klines(symbol, interval, limit):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    data = requests.get(url).json()
    df = pd.DataFrame(data, columns=[
        'time','o','h','l','c','v','ct','qv','n','tbbav','tbqv','ignore'
    ])
    for col in ['o','h','l','c','v']:
        df[col] = df[col].astype(float)
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    return df


# ==== INDICATORS ====
def ema(series, period): return series.ewm(span=period, adjust=False).mean()
def sma(series, period): return series.rolling(window=period).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta > 0, 0)).rolling(window=period).mean()
    rs = gain / loss
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
    minus_dm = low.diff() * -1
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    plus_di = 100 * (plus_dm.rolling(period).sum() / tr.rolling(period).sum())
    minus_di = 100 * (minus_dm.rolling(period).sum() / tr.rolling(period).sum())
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(period).mean()
    return adx.iloc[-1], plus_di.iloc[-1], minus_di.iloc[-1]

def calculate_atr(df, period=14):
    hl = df['h'] - df['l']
    hc = np.abs(df['h'] - df['c'].shift())
    lc = np.abs(df['l'] - df['c'].shift())
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean().iloc[-1]

def momentum_oscillator(df, period=10):
    close = df['c']
    momentum = (close / close.shift(period) - 1) * 100
    return momentum.iloc[-1]

def volume_weighted_momentum(df, period=20):
    close = df['c']; volume = df['v']
    vwap = (close * volume).rolling(period).sum() / volume.rolling(period).sum()
    momentum = (close.iloc[-1] / vwap.iloc[-1] - 1) * 100
    return momentum

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


# ==== CHECK SIGNAL ====
def check_hourly_signal(symbol):
    df1h = get_klines(symbol, INTERVAL_1H, LOOKBACK_CANDLES)
    df4h = get_klines(symbol, INTERVAL_4H, LOOKBACK_CANDLES)
    close = df1h['c']; latest_time = df1h['time'].iloc[-1]

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

    confidence, direction = 0, None
    returns = close.pct_change().dropna()
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
        msg = (f"*{direction} POSITION* — `{symbol}` @ {close_last}\n"
               f"SL: `{sl}` | TP: `{tp}`\n"
               f"Risk:Reward = 1:{risk_reward:.2f}\n"
               f"Position Size: {position_multiplier:.2f}x\n"
               f"Confidence: *{conf_level}* ({confidence:.1f}/6)\n"
               f"Trend 4H: *{trend_4h}* | ADX: {adx_val:.1f}\n"
               f"Timeframe: 1H | {latest_time.strftime('%Y-%m-%d %H:%M')}")
        return msg
    return None


# ==== LOG TRADES ====
def log_hourly_trade(symbol, direction, price, sl, tp, conf_level, confidence, trend, timestamp, risk_reward, adx, position_size):
    entry = {
        "time": timestamp, "symbol": symbol, "direction": direction,
        "price": price, "stop_loss": sl, "take_profit": tp,
        "confidence": conf_level, "confidence_score": confidence,
        "trend_4h": trend, "risk_reward": risk_reward, "adx": adx,
        "position_size": position_size
    }
    df = pd.DataFrame([entry])
    file_exists = os.path.exists(LOG_FILE)
    df.to_csv(LOG_FILE, mode='a', header=not file_exists, index=False)


# ==== MAIN (single run) ====
if __name__ == "__main__":
    print("🚀 Bot Run Started")
    try:
        symbols = get_top_20_symbols()
        found_signal = False
        for sym in symbols:
            signal = check_hourly_signal(sym)
            if signal:
                send_telegram_message(signal)
                print(f"[ALERT] {signal}")
                found_signal = True
        if not found_signal:
            print("[INFO] No signals found in this run.")
    except Exception as e:
        print("[ERROR] Exception in run:", e)
        send_telegram_message(f"🔴 Bot Error: {str(e)[:100]}...")
    print("✅ Bot run finished")
