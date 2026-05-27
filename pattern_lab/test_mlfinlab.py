"""Сравнение нашего pattern_engine с mlfinlab."""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# 1. Импортируем mlfinlab
# ============================================================
from mlfinlab.filters import filters
from mlfinlab.labeling import labeling
from mlfinlab.util import utils
from mlfinlab.structural_breaks import get_chow_type_stat_adf_values
from mlfinlab.features.fracdiff import frac_diff_ffd
from mlfinlab.data_generation import data_generation
from mlfinlab.sampling import sampling
from mlfinlab.bet_sizing import bet_sizing

print("mlfinlab загружен")
print(dir(filters)[:10])

# ============================================================
# 2. Загружаем наши данные
# ============================================================
data_dir = Path("data")
df = pd.read_csv(data_dir / "BTCUSDT_1h.csv")
df['ts'] = pd.to_datetime(df['ts'], unit='ms')
df = df.set_index('ts')
for c in ["Open", "High", "Low", "Close", "Volume"]:
    df[c] = df[c].astype(float)

close = df['Close'].values
print(f"Данные загружены: {len(df)} свечей BTC 1h")

# ============================================================
# 3. CUSUM фильтр (определяет смену режима)
# ============================================================
# Это не совсем фигуры, но мощный инструмент mlfinlab
try:
    cumsum_events = filters.cusum_filter(close, threshold=0.05)
    print(f"CUSUM нашёл {len(cumsum_events)} точек смены режима")
except Exception as e:
    print(f"CUSUM: {e}")

# ============================================================
# 4. Структурные разрывы (Chow-Type test)
# ============================================================
try:
    # Ищем структурные сдвиги в серии
    breaks = get_chow_type_stat_adf_values(close, max_lags=10)
    print(f"Структурные разрывы: {len(breaks)} точек")
except Exception as e:
    print(f"Chow test: {e}")

# ============================================================
# 5. Поиск пиков и впадин через mlfinlab
# ============================================================
from scipy.signal import argrelextrema

# mlfinlab не имеет прямых детекторов фигур, но имеет:
# - Фильтры (CUSUM, Z-score) для нахождения значимых движений
# - Структурные разрывы
# - Бет-сайзинг
# - Фрактальное дифференцирование

# Поэтому сравним: наш pattern_engine vs CUSUM точки входа
from pattern_engine import analyze_all_figures

# Находим фигуры нашим движком
our_figures = []
window = 60
for i in range(0, len(df) - window - 5, 20):
    chunk = df.iloc[i:i + window]
    result = analyze_all_figures(chunk)
    if result and result.get('weight', 0) >= 0.5:
        our_figures.append({
            'date': chunk.index[-1],
            'pattern': result['pattern'],
            'direction': result['direction'],
            'weight': result['weight'],
            'price': float(chunk['Close'].iloc[-1])
        })

print(f"\nНаш pattern_engine: {len(our_figures)} фигур")

# Сравниваем с CUSUM точками (если есть)
if len(cumsum_events) > 0:
    print(f"mlfinlab CUSUM: {len(cumsum_events)} точек входа")

    # Смотрим сколько наших фигур совпадает с CUSUM точками
    matches = 0
    for f in our_figures:
        for c_date in cumsum_events:
            if abs((f['date'] - c_date).days) <= 1:
                matches += 1
                break
    print(f"Совпадений: {matches}/{len(our_figures)}")

# ============================================================
# 6. ВЫВОД
# ============================================================
print("\n" + "=" * 60)
print("ВЫВОД: mlfinlab НЕ содержит детекторы фигур ТА")
print("=" * 60)
print("""
mlfinlab — это библиотека для:
  - Фильтрации значимых движений (CUSUM)
  - Структурных разрывов
  - Фрактального дифференцирования
  - Бет-сайзинга (расчёт размера ставки)
  - Сэмплирования данных

НО не для определения:
  - Голова-плечи
  - Двойное дно/вершина
  - Треугольники
  - Флаги/вымпелы
  - Клинья

Вывод: наш pattern_engine + калибровка — правильный путь.
""")