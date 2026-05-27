"""Строит ML-датасет из скачанных свечей."""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from pattern_engine import analyze_all_figures
import json

data_dir = Path("data")
all_rows = []

window = 60
step = 15
future_bars = 5

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

    for i in range(0, len(df) - window - future_bars, step):
        chunk = df.iloc[i:i + window]
        future = df.iloc[i + window:i + window + future_bars]

        # Цена входа и выхода
        entry_price = float(chunk['Close'].iloc[-1])
        exit_price = float(future['Close'].iloc[-1])

        # Target: 1 = цена пошла вверх, 0 = вниз
        target = 1 if exit_price > entry_price else 0

        # Фигура из pattern_engine
        pattern = analyze_all_figures(chunk)

        # Технические фичи из чанка
        close = chunk['Close'].values
        high = chunk['High'].values
        low = chunk['Low'].values
        volume = chunk['Volume'].values

        # RSI
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else 0
        avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else 1e-8
        rs = avg_gain / avg_loss if avg_loss > 0 else 0
        rsi = 100 - (100 / (1 + rs))

        # ATR
        tr = np.maximum(high[-14:] - low[-14:],
                        np.abs(high[-14:] - np.roll(close[-15:-1], 1)))
        atr = np.mean(tr)
        atr_pct = atr / entry_price * 100 if entry_price > 0 else 0

        # Volume ratio
        vol_sma = np.mean(volume[-20:]) if len(volume) >= 20 else np.mean(volume)
        vol_ratio = volume[-1] / vol_sma if vol_sma > 0 else 1

        # Momentum
        momentum_4 = (close[-1] - close[-5]) / close[-5] * 100 if len(close) >= 5 else 0
        momentum_12 = (close[-1] - close[-13]) / close[-13] * 100 if len(close) >= 13 else 0

        # Compression (диапазон)
        range_start = high[-20] - low[-20] if len(high) >= 20 else 0
        range_end = high[-1] - low[-1]
        compression = (range_start - range_end) / range_start * 100 if range_start > 0 else 0

        # Час и день
        dt = chunk.index[-1]
        hour = dt.hour
        dow = dt.dayofweek

        # Собираем строку
        row = {
            'symbol': symbol,
            'tf': tf,
            'date': str(dt),
            'target': target,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_pct': round((exit_price - entry_price) / entry_price * 100, 2),
            # Pattern
            'pattern_found': 1 if pattern else 0,
            'pattern_name': pattern['pattern'] if pattern else 'none',
            'pattern_en': pattern['pattern_en'] if pattern else 'none',
            'pattern_weight': pattern['weight'] if pattern else 0,
            'pattern_direction': pattern['direction'] if pattern else 'none',
            # Технические
            'rsi': round(rsi, 1),
            'atr_pct': round(atr_pct, 2),
            'volume_ratio': round(vol_ratio, 2),
            'momentum_4': round(momentum_4, 2),
            'momentum_12': round(momentum_12, 2),
            'compression': round(compression, 1),
            # Время
            'hour': hour,
            'dow': dow,
            'hour_sin': round(np.sin(2 * np.pi * hour / 24), 4),
            'hour_cos': round(np.cos(2 * np.pi * hour / 24), 4),
            'dow_sin': round(np.sin(2 * np.pi * dow / 7), 4),
            'dow_cos': round(np.cos(2 * np.pi * dow / 7), 4),
            'is_weekend': 1 if dow >= 5 else 0,
        }
        all_rows.append(row)

# Сохраняем
df_ml = pd.DataFrame(all_rows)
df_ml.to_csv('ml_dataset.csv', index=False)
print(f"\nГотово! {len(df_ml)} строк сохранено в ml_dataset.csv")
print(f"Target distribution: {df_ml['target'].mean():.1%} up")

# Статистика по паттернам
if 'pattern_name' in df_ml.columns:
    print(f"\nПаттерны:")
    for pat in df_ml['pattern_name'].value_counts().head(10).index:
        subset = df_ml[df_ml['pattern_name'] == pat]
        if len(subset) > 10:
            print(f"  {pat}: n={len(subset)}, target_up={subset['target'].mean():.1%}")