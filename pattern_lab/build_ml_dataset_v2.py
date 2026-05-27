"""Строит ML-датасет со ВСЕМИ фичами из signal_scorer."""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from pattern_engine import analyze_all_figures
import requests
import json

data_dir = Path("data")
all_rows = []

window = 60
step = 15
future_bars = 5

# ============================================================
# Скачиваем BTC для regime detection
# ============================================================
print("Скачиваем BTC...")
r_btc = requests.get(
    "https://api.binance.com/api/v3/klines",
    params={"symbol": "BTCUSDT", "interval": "1h", "limit": 1000},
    timeout=30
)
df_btc = pd.DataFrame(r_btc.json(), columns=[
    "ts", "Open", "High", "Low", "Close", "Volume",
    "close_ts", "quote_vol", "trades", "tbb", "tbq", "ignore"
])
df_btc['ts'] = pd.to_datetime(df_btc['ts'], unit='ms')
df_btc = df_btc.set_index('ts')
for c in ["Open", "High", "Low", "Close", "Volume"]:
    df_btc[c] = df_btc[c].astype(float)


def get_btc_regime(btc_df, idx):
    """Определяет regime BTC на момент idx."""
    if idx < 50:
        return "unknown", 1.0
    chunk = btc_df.iloc[max(0, idx - 50):idx + 1]
    if len(chunk) < 30:
        return "unknown", 1.0
    c = chunk['Close']
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    trend_up = float(ema20.iloc[-1]) > float(ema50.iloc[-1])

    # ATR percentile
    tr = pd.concat([
        chunk['High'] - chunk['Low'],
        (chunk['High'] - chunk['Close'].shift()).abs(),
        (chunk['Low'] - chunk['Close'].shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().dropna()
    if len(atr) > 10:
        cur_atr = float(atr.iloc[-1])
        atr_rank = float(np.mean(atr.values <= cur_atr))
    else:
        atr_rank = 0.5

    h20 = float(chunk['High'].tail(20).max())
    l20 = float(chunk['Low'].tail(20).min())
    rng = (h20 - l20) / l20 * 100 if l20 > 0 else 0

    if atr_rank > 0.85:
        return "volatile", 0.9
    elif abs(float(ema20.iloc[-1]) - float(ema50.iloc[-1])) / float(ema50.iloc[-1]) * 100 > 0.5 and atr_rank < 0.7:
        return ("trend_up" if trend_up else "trend_down"), 1.3
    elif rng < 3.0:
        return "chop", 0.8
    return "normal", 1.0


# ============================================================
# Обработка файлов
# ============================================================
for f in sorted(data_dir.glob("*.csv")):
    symbol, tf = f.stem.split("_")
    print(f"Обрабатываю {symbol} {tf}...")

    df = pd.read_csv(f)
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df = df.set_index('ts')
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = df[c].astype(float)

    if len(df) < window + future_bars:
        continue

    # Предварительно считаем ATR и volume_sma для всего файла
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift()).abs(),
        (df['Low'] - df['Close'].shift()).abs(),
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['volume_sma'] = df['Volume'].rolling(20).mean()

    for i in range(0, len(df) - window - future_bars, step):
        chunk = df.iloc[i:i + window]
        future = df.iloc[i + window:i + window + future_bars]

        entry_price = float(chunk['Close'].iloc[-1])
        exit_price = float(future['Close'].iloc[-1])
        high_5 = float(future['High'].max())
        low_5 = float(future['Low'].min())

        target = 1 if exit_price > entry_price else 0

        # Фигура
        pattern = analyze_all_figures(chunk)

        close = chunk['Close'].values
        high = chunk['High'].values
        low = chunk['Low'].values
        volume = chunk['Volume'].values

        # === ТЕХНИЧЕСКИЕ ФИЧИ ===

        # RSI (14)
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else 0
        avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else 1e-8
        rs = avg_gain / avg_loss if avg_loss > 0 else 0
        rsi = 100 - (100 / (1 + rs))

        # ATR и ATR%
        atr_val = float(chunk['atr'].iloc[-1]) if not pd.isna(chunk['atr'].iloc[-1]) else 0
        atr_pct = atr_val / entry_price * 100 if entry_price > 0 else 0

        # ATR percentile
        atr_vals = chunk['atr'].dropna().values
        if len(atr_vals) > 10:
            atr_percentile = np.mean(atr_vals <= atr_val) if atr_val > 0 else 0.5
        else:
            atr_percentile = 0.5

        # Volume ratio
        vol_sma = np.mean(volume[-20:]) if len(volume) >= 20 else np.mean(volume)
        vol_ratio = volume[-1] / vol_sma if vol_sma > 0 else 1

        # Volume trend
        if len(volume) >= 4:
            vs = volume[-4:]
            vol_trend = 1.0 if all(vs[i] > vs[i - 1] for i in range(1, 4)) else (0.5 if vs[-1] > vs[0] else 0.1)
        else:
            vol_trend = 0.5

        # Candle efficiency
        r = chunk.tail(5)
        body = (r['Close'] - r['Open']).abs()
        range_ = (r['High'] - r['Low']).replace(0, np.nan)
        candle_eff = float((body / range_).dropna().mean()) if not (body / range_).dropna().empty else 0.5

        # Momentum
        momentum_4 = (close[-1] - close[-5]) / close[-5] * 100 if len(close) >= 5 else 0
        momentum_12 = (close[-1] - close[-13]) / close[-13] * 100 if len(close) >= 13 else 0

        # Compression
        range_start = high[-20] - low[-20] if len(high) >= 20 else 0
        range_end = high[-1] - low[-1]
        compression = (range_start - range_end) / range_start * 100 if range_start > 0 else 0

        # === РЫНОЧНЫЕ ФИЧИ ===

        # BTC regime
        btc_idx = min(i + window, len(df_btc) - 1)
        btc_regime, btc_rmult = get_btc_regime(df_btc, btc_idx)

        # === ВРЕМЯ ===
        dt = chunk.index[-1]
        hour = dt.hour
        dow = dt.dayofweek

        # === SESSION ===
        SESSION_GREAT = [3, 4, 6, 8, 11, 14, 16, 18, 23]
        SESSION_GOOD = [0, 5, 7, 10, 12, 13, 21]
        SESSION_BAD = [1, 2, 9, 15, 17, 19, 20, 22]

        if hour in SESSION_GREAT:
            session_sc = 1.0
        elif hour in SESSION_GOOD:
            session_sc = 0.6
        elif hour in SESSION_BAD:
            session_sc = 0.1
        else:
            session_sc = 0.4

        # === СОБИРАЕМ ===
        row = {
            # Идентификация
            'symbol': symbol, 'tf': tf, 'date': str(dt),
            # Target
            'target': target, 'entry_price': entry_price, 'exit_price': exit_price,
            'high_5': high_5, 'low_5': low_5,
            'pnl_pct': round((exit_price - entry_price) / entry_price * 100, 2),
            # Pattern
            'pattern_found': 1 if pattern else 0,
            'pattern_name': pattern['pattern'] if pattern else 'none',
            'pattern_en': pattern['pattern_en'] if pattern else 'none',
            'pattern_weight': pattern['weight'] if pattern else 0,
            'pattern_direction': pattern['direction'] if pattern else 'none',
            # Technical
            'rsi': round(rsi, 1),
            'atr_pct': round(atr_pct, 2),
            'atr_percentile': round(atr_percentile, 2),
            'volume_ratio': round(vol_ratio, 2),
            'volume_trend': round(vol_trend, 2),
            'candle_efficiency': round(candle_eff, 2),
            'momentum_4': round(momentum_4, 2),
            'momentum_12': round(momentum_12, 2),
            'compression': round(compression, 1),
            # Market
            'btc_regime': btc_regime,
            'btc_rmult': btc_rmult,
            'session_score': session_sc,
            # Time
            'hour': hour, 'dow': dow,
            'hour_sin': round(np.sin(2 * np.pi * hour / 24), 4),
            'hour_cos': round(np.cos(2 * np.pi * hour / 24), 4),
            'dow_sin': round(np.sin(2 * np.pi * dow / 7), 4),
            'dow_cos': round(np.cos(2 * np.pi * dow / 7), 4),
            'is_weekend': 1 if dow >= 5 else 0,
        }
        all_rows.append(row)

# Сохраняем
df_ml = pd.DataFrame(all_rows)
df_ml.to_csv('ml_dataset_v2.csv', index=False)
print(f"\nГотово! {len(df_ml)} строк в ml_dataset_v2.csv")
print(f"Колонок: {len(df_ml.columns)}")
print(f"Target distribution: {df_ml['target'].mean():.1%} up")
print(f"Колонки: {list(df_ml.columns)}")