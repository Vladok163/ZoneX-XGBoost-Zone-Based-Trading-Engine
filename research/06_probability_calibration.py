"""Исследование #6: Probability Calibration — насколько prediction соответствует реальности."""
import pandas as pd
import numpy as np
import json
from pathlib import Path

df = pd.read_csv('ml_dataset_v2.csv')
results_dir = Path('results')

# Используем упрощённый скорер для получения probability
import math


def calc_probability(row):
    """Быстрый скорер для calibration."""
    score = 0
    rsi = row['rsi']
    vol = row['volume_ratio']
    imp = abs(row['momentum_4'])
    comp = row['compression']

    # RSI
    if 25 <= rsi <= 35:
        score += 1.5
    elif 45 <= rsi <= 55:
        score += 1.0
    elif 35 <= rsi < 45:
        score += 0.5
    elif rsi < 25:
        score += 1.0
    else:
        score += 0

    # Volume
    if 0.5 <= vol <= 1.0:
        score += 1.0
    elif 0.3 <= vol < 0.5:
        score += 0.5
    elif 1.0 < vol <= 1.5:
        score -= 0.5
    else:
        score += 0

    # Impulse
    if 5 <= imp <= 10:
        score += 1.0
    elif 3 <= imp < 5:
        score += 0.5
    elif imp > 15:
        score -= 0.5
    else:
        score += 0

    # Compression
    if 70 <= comp <= 90:
        score += 1.0
    elif 50 <= comp < 70:
        score += 0
    elif 30 <= comp < 50:
        score += 0.5
    else:
        score += 0.5

    # Session
    good_hours = [3, 4, 6, 8, 11, 14, 16, 18, 23]
    bad_hours = [1, 2, 9, 15, 17, 19, 20, 22]
    if row['hour'] in good_hours:
        score += 1.0
    elif row['hour'] in bad_hours:
        score -= 0.5

    # Day
    if row['dow'] in [2, 5]:
        score += 0.5
    elif row['dow'] == 1:
        score -= 0.5

    # ATR
    if 0.5 <= row['atr_pct'] <= 3.0:
        score += 0.5
    elif row['atr_pct'] > 3.0:
        score -= 0.5

    # Sigmoid
    prob = 1 / (1 + math.exp(-(score - 3.0)))
    return round(prob, 3)


df['probability'] = df.apply(calc_probability, axis=1)

print('=' * 80)
print('ИССЛЕДОВАНИЕ #6: PROBABILITY CALIBRATION')
print('Насколько prediction соответствует реальности?')
print('=' * 80)

# Распределение вероятностей
print(f'\nРаспределение вероятностей:')
for p_min, p_max, label in [(0, 0.45, '<0.45'), (0.45, 0.55, '0.45-0.55'), (0.55, 0.65, '0.55-0.65'),
                            (0.65, 0.75, '0.65-0.75'), (0.75, 0.85, '0.75-0.85'), (0.85, 1.0, '0.85+')]:
    s = df[(df['probability'] >= p_min) & (df['probability'] < p_max)]
    if len(s) >= 10:
        wr = (s['target'] == 1).mean() * 100
        print(f'  {label:<12s}: n={len(s):>6d}, predicted_mid={(p_min + p_max) / 2:.2f}, actual_WR={wr:.1f}%')

# Детальная калибровка
print(f'\n{"=" * 60}')
print('КАЛИБРОВОЧНАЯ ТАБЛИЦА')
print(f'{"=" * 60}')
print(f'{"Predicted":>10s} {"N":>6s} {"Actual WR":>10s} {"Bias":>8s} {"Calibration"}')
print('-' * 55)

calibration = []
for threshold in np.arange(0.30, 0.90, 0.05):
    s = df[df['probability'] >= threshold]
    if len(s) >= 20:
        predicted = round(threshold + (1 - threshold) / 2, 2)
        actual = (s['target'] == 1).mean()
        bias = actual - predicted
        status = '✅ GOOD' if abs(bias) < 0.05 else ('⚠️ OVER' if bias > 0.05 else '❌ UNDER')

        calibration.append({
            'threshold': round(threshold, 2),
            'predicted_p': predicted,
            'actual_wr': round(actual, 4),
            'n': len(s),
            'bias': round(bias, 4),
            'status': status
        })

        print(f'{predicted:>10.2f} {len(s):>6d} {actual:>9.1%} {bias:>+8.3f} {status}')

# Лучший порог
print(f'\n{"=" * 60}')
print('ПОИСК ОПТИМАЛЬНОГО ПОРОГА')
print(f'{"=" * 60}')
print(f'{"Threshold":>10s} {"N":>6s} {"WR":>7s} {"avgPnL":>8s} {"PF":>7s} {"sumR":>8s}')
print('-' * 55)

best_thresholds = []
for threshold in np.arange(0.30, 0.85, 0.05):
    s = df[df['probability'] >= threshold]
    if len(s) >= 30:
        wr = (s['target'] == 1).mean() * 100
        avg_pnl = s['pnl_pct'].mean()
        pos = s[s['pnl_pct'] > 0]['pnl_pct'].sum()
        neg = abs(s[s['pnl_pct'] < 0]['pnl_pct'].sum())
        pf = pos / neg if neg > 0 else 99
        sum_r = s['pnl_pct'].sum()

        best_thresholds.append({
            'threshold': round(threshold, 2),
            'n': len(s), 'wr': round(wr, 1),
            'avg_pnl': round(avg_pnl, 2), 'pf': round(pf, 2), 'sum_r': round(sum_r, 1)
        })

        mark = ' ✅ BEST' if pf >= 1.2 and wr >= 50 else ''
        print(f'{threshold:>10.2f} {len(s):>6d} {wr:>6.1f}% {avg_pnl:>+7.2f}% {pf:>6.2f} {sum_r:>+7.1f}{mark}')

# Сохраняем
with open(results_dir / '06_probability_calibration.json', 'w', encoding='utf-8') as f:
    json.dump({
        'calibration': calibration,
        'best_thresholds': best_thresholds,
        'optimal_threshold': max(best_thresholds, key=lambda x: x['pf']) if best_thresholds else None
    }, f, indent=2, ensure_ascii=False)
print(f'\n✅ Сохранено: results/06_probability_calibration.json')