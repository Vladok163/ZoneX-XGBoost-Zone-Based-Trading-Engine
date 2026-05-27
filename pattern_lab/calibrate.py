"""Калибровка Pattern Engine — перебор параметров для всех фигур."""
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product
import json
from pattern_engine import (
    detect_double_top_bottom,
    detect_triangle,
    detect_flag_pennant,
    detect_wedge,
    detect_head_shoulders,
)

# ============================================================
# ЗАГРУЗКА ДАННЫХ
# ============================================================
data_dir = Path("data")
all_data = []

for f in data_dir.glob("*.csv"):
    symbol, tf = f.stem.split("_")
    df = pd.read_csv(f)
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df = df.set_index('ts')
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = df[c].astype(float)
    all_data.append({'symbol': symbol, 'tf': tf, 'df': df})

print(f"Загружено: {len(all_data)} файлов")


# ============================================================
# ФУНКЦИЯ ТЕСТИРОВАНИЯ
# ============================================================
def test_detector(detector_func, param_grid, data, window=60, step=15, future_bars=5):
    """
    Перебирает параметры, считает точность.
    Точность = % правильных предсказаний направления.
    """
    results = []

    keys = list(param_grid.keys())
    values = list(param_grid.values())

    for combo in product(*values):
        params = dict(zip(keys, combo))
        correct = 0
        total = 0

        for d in data:
            df = d['df']
            for i in range(0, len(df) - window - future_bars, step):
                chunk = df.iloc[i:i + window]
                future = df.iloc[i + window:i + window + future_bars]
                if len(future) < future_bars:
                    continue

                try:
                    result = detector_func(chunk, **params)
                except:
                    continue

                if result and result.get('weight', 0) >= 0.5:
                    total += 1
                    entry = float(chunk['Close'].iloc[-1])
                    exit_ = float(future['Close'].iloc[-1])

                    if result['direction'] == 'bull':
                        if exit_ > entry: correct += 1
                    else:
                        if exit_ < entry: correct += 1

        if total >= 5:
            accuracy = correct / total * 100
            results.append({**params, 'total': total, 'correct': correct, 'accuracy': round(accuracy, 1)})

    return sorted(results, key=lambda x: x['accuracy'], reverse=True)


# ============================================================
# 1. КАЛИБРОВКА ДВОЙНОГО ДНА/ВЕРШИНЫ
# ============================================================
print("\n" + "=" * 60)
print("1. Двойное дно / Двойная вершина")
print("=" * 60)

param_grid_dt = {
    'order': [3, 5, 7, 10],
    'tolerance': [0.02, 0.03, 0.05],
}
results_dt = test_detector(detect_double_top_bottom, param_grid_dt, all_data)

print(f"Топ-5 параметров:")
for r in results_dt[:5]:
    print(f"  order={r['order']}, tol={r['tolerance']:.2f}: {r['accuracy']}% ({r['correct']}/{r['total']})")

best_dt = results_dt[0] if results_dt else None

# ============================================================
# 2. КАЛИБРОВКА ТРЕУГОЛЬНИКОВ
# ============================================================
print("\n" + "=" * 60)
print("2. Треугольники")
print("=" * 60)

# Треугольник не принимает параметры, тестируем as-is
total_tri = 0
correct_tri = 0
for d in all_data:
    df = d['df']
    for i in range(0, len(df) - 65, 15):
        chunk = df.iloc[i:i + 60]
        future = df.iloc[i + 60:i + 65]
        if len(future) < 5: continue
        result = detect_triangle(chunk)
        if result and result.get('weight', 0) >= 0.5:
            total_tri += 1
            entry = float(chunk['Close'].iloc[-1])
            exit_ = float(future['Close'].iloc[-1])
            if result['direction'] == 'bull' and exit_ > entry:
                correct_tri += 1
            elif result['direction'] == 'bear' and exit_ < entry:
                correct_tri += 1

if total_tri > 0:
    print(f"  Точность: {correct_tri / total_tri * 100:.1f}% ({correct_tri}/{total_tri})")
else:
    print(f"  Фигур не найдено")

# ============================================================
# 3. КАЛИБРОВКА ФЛАГА/ВЫМПЕЛА
# ============================================================
print("\n" + "=" * 60)
print("3. Флаг / Вымпел")
print("=" * 60)

param_grid_flag = {
    # Нет параметров — тестируем as-is
}
total_flag = 0
correct_flag = 0
for d in all_data:
    df = d['df']
    for i in range(0, len(df) - 65, 15):
        chunk = df.iloc[i:i + 60]
        future = df.iloc[i + 60:i + 65]
        if len(future) < 5: continue
        result = detect_flag_pennant(chunk)
        if result and result.get('weight', 0) >= 0.5:
            total_flag += 1
            entry = float(chunk['Close'].iloc[-1])
            exit_ = float(future['Close'].iloc[-1])
            if result['direction'] == 'bull' and exit_ > entry:
                correct_flag += 1
            elif result['direction'] == 'bear' and exit_ < entry:
                correct_flag += 1

if total_flag > 0:
    print(f"  Точность: {correct_flag / total_flag * 100:.1f}% ({correct_flag}/{total_flag})")
else:
    print(f"  Фигур не найдено")

# ============================================================
# 4. КАЛИБРОВКА КЛИНА
# ============================================================
print("\n" + "=" * 60)
print("4. Клин")
print("=" * 60)

total_wedge = 0
correct_wedge = 0
for d in all_data:
    df = d['df']
    for i in range(0, len(df) - 65, 15):
        chunk = df.iloc[i:i + 60]
        future = df.iloc[i + 60:i + 65]
        if len(future) < 5: continue
        result = detect_wedge(chunk)
        if result and result.get('weight', 0) >= 0.5:
            total_wedge += 1
            entry = float(chunk['Close'].iloc[-1])
            exit_ = float(future['Close'].iloc[-1])
            if result['direction'] == 'bull' and exit_ > entry:
                correct_wedge += 1
            elif result['direction'] == 'bear' and exit_ < entry:
                correct_wedge += 1

if total_wedge > 0:
    print(f"  Точность: {correct_wedge / total_wedge * 100:.1f}% ({correct_wedge}/{total_wedge})")
else:
    print(f"  Фигур не найдено")

# ============================================================
# 5. КАЛИБРОВКА ГОЛОВА-ПЛЕЧИ
# ============================================================
print("\n" + "=" * 60)
print("5. Голова и плечи")
print("=" * 60)

total_hs = 0
correct_hs = 0
for d in all_data:
    df = d['df']
    for i in range(0, len(df) - 65, 15):
        chunk = df.iloc[i:i + 60]
        future = df.iloc[i + 60:i + 65]
        if len(future) < 5: continue
        result = detect_head_shoulders(chunk)
        if result and result.get('weight', 0) >= 0.5:
            total_hs += 1
            entry = float(chunk['Close'].iloc[-1])
            exit_ = float(future['Close'].iloc[-1])
            if result['direction'] == 'bull' and exit_ > entry:
                correct_hs += 1
            elif result['direction'] == 'bear' and exit_ < entry:
                correct_hs += 1

if total_hs > 0:
    print(f"  Точность: {correct_hs / total_hs * 100:.1f}% ({correct_hs}/{total_hs})")
else:
    print(f"  Фигур не найдено")

# ============================================================
# СВОДКА
# ============================================================
print("\n" + "=" * 60)
print("СВОДКА РЕЗУЛЬТАТОВ")
print("=" * 60)

summary = {}
if best_dt:
    print(f"Двойное дно/вершина: {best_dt['accuracy']}% (order={best_dt['order']}, tol={best_dt['tolerance']})")
    summary['double_top_bottom'] = best_dt
if total_tri > 0:
    print(f"Треугольники:        {correct_tri / total_tri * 100:.1f}% ({correct_tri}/{total_tri})")
    summary['triangle'] = {'accuracy': round(correct_tri / total_tri * 100, 1), 'total': total_tri}
if total_flag > 0:
    print(f"Флаг/Вымпел:         {correct_flag / total_flag * 100:.1f}% ({correct_flag}/{total_flag})")
    summary['flag'] = {'accuracy': round(correct_flag / total_flag * 100, 1), 'total': total_flag}
if total_wedge > 0:
    print(f"Клин:                {correct_wedge / total_wedge * 100:.1f}% ({correct_wedge}/{total_wedge})")
    summary['wedge'] = {'accuracy': round(correct_wedge / total_wedge * 100, 1), 'total': total_wedge}
if total_hs > 0:
    print(f"Голова и плечи:      {correct_hs / total_hs * 100:.1f}% ({correct_hs}/{total_hs})")
    summary['head_shoulders'] = {'accuracy': round(correct_hs / total_hs * 100, 1), 'total': total_hs}

with open('results/calibration.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("\nСохранено: results/calibration.json")