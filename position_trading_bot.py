import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime
import os

# ==== CONFIG ====
BOT_TOKEN = "7828854549:AAGo8Dx9RlIs13a6dZ9I73-2u6dDvkx7LvY"
CHAT_ID = -1002558399674
INTERVAL_1H = "1h"
INTERVAL_4H = "4h"
LOOKBACK_CANDLES = 100
CHECK_EVERY = 1800  # seconds (30 min checks for 1h strategy)
LOG_FILE = "hourly_position_signals.csv"
last_alert_time = {}

# 1-hour trading parameters
ATR_MULTIPLIER_SL = 2.0  # Wider stops for 1h timeframe
ATR_MULTIPLIER_TP = 3.5  # Better risk-reward for swing trades
MOMENTUM_THRESHOLD = 0.01  # 1% momentum threshold
MIN_TREND_STRENGTH = 0.5

# ==== TELEGRAM SEND FUNCTION ====
def send_telegram_message(msg):
    print("[DEBUG] Sending message to Telegram...")
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, json=payload)
        if r.status_code != 200:
            print("[ERROR] Telegram send error:", r.text)
    except Exception as e:
        print("[ERROR] Telegram send exception:", e)

# ==== GET COINS ====
def get_top_20_symbols():
    print("[DEBUG] Fetching top 20 USDT pairs by volume...")
    url = "https://api.binance.com/api/v3/ticker/24hr"
    data = requests.get(url).json()
    df = pd.DataFrame(data)
    df['quoteVolume'] = df['quoteVolume'].astype(float)
    df = df[df['symbol'].str.endswith('USDT')]
    df = df.sort_values('quoteVolume', ascending=False).head(20)
    symbols = df['symbol'].tolist()
    print(f"[INFO] Top 20 symbols: {symbols}")
    return symbols

# ==== GET CANDLE DATA ====
def get_klines(symbol, interval, limit):
    print(f"[DEBUG] Fetching {interval} klines for {symbol}...")
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    data = requests.get(url).json()
    df = pd.DataFrame(data, columns=[
        'time','o','h','l','c','v','ct','qv','n','tbbav','tbqv','ignore'
    ])
    for col in ['o','h','l','c','v']:
        df[col] = df[col].astype(float)
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    return df

# ==== ADVANCED TECHNICAL INDICATORS FOR 1H ====
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def sma(series, period):
    return series.rolling(window=period).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
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
    """Average Directional Index for trend strength"""
    high, low, close = df['h'], df['l'], df['c']
    
    # Calculate directional movement
    plus_dm = high.diff()
    minus_dm = low.diff() * -1
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Calculate true range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate directional indicators
    plus_di = 100 * (plus_dm.rolling(period).sum() / tr.rolling(period).sum())
    minus_di = 100 * (minus_dm.rolling(period).sum() / tr.rolling(period).sum())
    
    # Calculate ADX
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
    """Custom momentum oscillator for 1h timeframe"""
    close = df['c']
    momentum = (close / close.shift(period) - 1) * 100
    return momentum.iloc[-1]

def volume_weighted_momentum(df, period=20):
    """Volume-weighted price momentum"""
    close = df['c']
    volume = df['v']
    
    vwap = (close * volume).rolling(period).sum() / volume.rolling(period).sum()
    momentum = (close.iloc[-1] / vwap.iloc[-1] - 1) * 100
    return momentum

def trend_strength_4h(df):
    """Higher timeframe trend analysis"""
    close = df['c']
    ema_20 = ema(close, 20)
    ema_50 = ema(close, 50)
    ema_100 = ema(close, 100)
    
    # Calculate trend strength
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
    """Adjust position size based on volatility"""
    if volatility > 0.05:  # High volatility
        return base_size * 0.5
    elif volatility > 0.03:  # Medium volatility
        return base_size * 0.75
    else:  # Low volatility
        return base_size

# ==== ENHANCED 1H SIGNAL CHECK ====
def check_hourly_signal(symbol):
    print(f"[DEBUG] Checking 1H signal for {symbol}...")
    df1h = get_klines(symbol, INTERVAL_1H, LOOKBACK_CANDLES)
    df4h = get_klines(symbol, INTERVAL_4H, LOOKBACK_CANDLES)
    
    close = df1h['c']
    latest_time = df1h['time'].iloc[-1]
    
    # Technical indicators
    ema_21 = ema(close, 21)
    ema_50 = ema(close, 50)
    sma_200 = sma(close, 200)
    rsi_val = rsi(close, 14)
    macd_line, signal_line, histogram = macd(close)
    atr_val = calculate_atr(df1h)
    
    # Advanced indicators
    adx_val, plus_di, minus_di = adx(df1h)
    momentum = momentum_oscillator(df1h)
    vw_momentum = volume_weighted_momentum(df1h)
    trend_4h, trend_strength = trend_strength_4h(df4h)
    
    # Current values
    close_last = close.iloc[-1]
    ema_21_last = ema_21.iloc[-1]
    ema_50_last = ema_50.iloc[-1]
    sma_200_last = sma_200.iloc[-1]
    rsi_last = rsi_val.iloc[-1]
    macd_last = macd_line.iloc[-1]
    signal_last = signal_line.iloc[-1]
    
    confidence = 0
    direction = None
    
    # Calculate volatility for position sizing
    returns = close.pct_change().dropna()
    volatility = returns.rolling(20).std().iloc[-1]
    position_multiplier = dynamic_position_sizing(volatility)
    
    # BUY Setup for 1H timeframe
    if (ema_21_last > ema_50_last and close_last > sma_200_last and 
        rsi_last > 40 and rsi_last < 75 and adx_val > 25):
        
        confidence += 1  # Basic trend alignment
        
        # MACD confirmation
        if macd_last > signal_last and macd_last > 0:
            confidence += 1
        
        # Momentum confirmation
        if momentum > MOMENTUM_THRESHOLD:
            confidence += 1
        
        # Volume-weighted momentum
        if vw_momentum > 0:
            confidence += 1
        
        # Higher timeframe trend
        if trend_4h in ["up", "strong_up"] and trend_strength > MIN_TREND_STRENGTH:
            confidence += 1
        
        # ADX strength
        if plus_di > minus_di and adx_val > 30:
            confidence += 0.5
        
        direction = "BUY"
    
    # SELL Setup for 1H timeframe
    elif (ema_21_last < ema_50_last and close_last < sma_200_last and 
          rsi_last < 60 and rsi_last > 25 and adx_val > 25):
        
        confidence += 1  # Basic trend alignment
        
        # MACD confirmation
        if macd_last < signal_last and macd_last < 0:
            confidence += 1
        
        # Momentum confirmation
        if momentum < -MOMENTUM_THRESHOLD:
            confidence += 1
        
        # Volume-weighted momentum
        if vw_momentum < 0:
            confidence += 1
        
        # Higher timeframe trend
        if trend_4h in ["down", "strong_down"] and trend_strength > MIN_TREND_STRENGTH:
            confidence += 1
        
        # ADX strength
        if minus_di > plus_di and adx_val > 30:
            confidence += 0.5
        
        direction = "SELL"
    
    if direction and confidence >= 4:
        if last_alert_time.get(symbol) == latest_time:
            print(f"[INFO] Duplicate alert skipped for {symbol}")
            return None
        
        last_alert_time[symbol] = latest_time
        
        # Enhanced dynamic SL/TP for 1H timeframe
        if direction == "BUY":
            sl = round(close_last - (atr_val * ATR_MULTIPLIER_SL), 6)
            tp = round(close_last + (atr_val * ATR_MULTIPLIER_TP), 6)
        else:
            sl = round(close_last + (atr_val * ATR_MULTIPLIER_SL), 6)
            tp = round(close_last - (atr_val * ATR_MULTIPLIER_TP), 6)
        
        conf_level = "Very High" if confidence >= 5.5 else "High" if confidence >= 4.5 else "Medium"
        risk_reward = abs((tp - close_last) / (close_last - sl)) if direction == "BUY" else abs((close_last - tp) / (sl - close_last))
        
        log_hourly_trade(symbol, direction, close_last, sl, tp, conf_level, confidence, 
                        trend_4h, latest_time, risk_reward, adx_val, position_multiplier)
        
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

# ==== ENHANCED LOGGING ====
def log_hourly_trade(symbol, direction, price, sl, tp, conf_level, confidence, trend, timestamp, risk_reward, adx, position_size):
    entry = {
        "time": timestamp,
        "symbol": symbol,
        "direction": direction,
        "price": price,
        "stop_loss": sl,
        "take_profit": tp,
        "confidence": conf_level,
        "confidence_score": confidence,
        "trend_4h": trend,
        "risk_reward": risk_reward,
        "adx": adx,
        "position_size": position_size
    }
    
    df = pd.DataFrame([entry])
    file_exists = os.path.exists(LOG_FILE)
    df.to_csv(LOG_FILE, mode='a', header=not file_exists, index=False)
    print(f"[DEBUG] Logged 1H trade for {symbol}")

# ==== MAIN LOOP ====
if __name__ == "__main__":
    print("🚀 1-Hour Position Trading Bot Started...")
    while True:
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
                print("[INFO] No 1H position signals found in this check.")
                
        except Exception as e:
            print("[ERROR] Main loop exception:", e)
            send_telegram_message(f"🔴 1H Bot Error: {str(e)[:100]}...")
        
        print(f"[DEBUG] Waiting {CHECK_EVERY} seconds until next 1H scan...\n")
        time.sleep(CHECK_EVERY)
