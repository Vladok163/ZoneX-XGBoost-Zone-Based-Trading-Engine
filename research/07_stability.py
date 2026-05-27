"""Исследование #7: Stability — устойчивость результатов по месяцам."""
import pandas as pd
import numpy as np
import json
from pathlib import Path

df = pd.read_csv('ml_dataset_v2.csv')
results_dir = Path('results')

df['month'] = pd.to_datetime(df['date']).dt.month
df['month_name'] = df['month'].map({1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                                    7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'})

months = sorted(df['month'].unique())

print('=' * 80)
print('ИССЛЕДОВАНИЕ #7: STABILITY')
print('Устойчивость результатов по месяцам')
print('=' * 80)

# По месяцам
print(f'\n{"=" * 60}')
print('БАЗОВЫЕ МЕТРИКИ ПО МЕСЯЦАМ')
print(f'{"=" * 60}')
print(f'{"Месяц":<8s} {"N":>6s} {"WR":>7s} {"avgPnL":>8s} {"PF":>7s} {"sumR":>8s}')
print('-' * 50)

monthly_stats = []
for m in months:
    subset = df[df['month'] == m]
    if len(subset) < 30:
        continue
    n = len(subset)
    wr = (subset['target'] == 1).mean() * 100
    avg_pnl = subset['pnl_pct'].mean()
    pos = subset[subset['pnl_pct'] > 0]['pnl_pct'].sum()
    neg = abs(subset[subset['pnl_pct'] < 0]['pnl_pct'].sum())
    pf = pos / neg if neg > 0 else 99

    name = subset['month_name'].iloc[0]
    monthly_stats.append({
        'month': name, 'n': n, 'wr': round(wr, 1),
        'avg_pnl': round(avg_pnl, 2), 'pf': round(pf, 2),
        'sum_r': round(subset['pnl_pct'].sum(), 1)
    })
    bar = '█' * int(wr / 5)
    print(f'{name:<8s} {n:>6d} {wr:>6.1f}% {avg_pnl:>+7.2f}% {pf:>6.2f} {subset["pnl_pct"].sum():>+7.1f} {bar}')

# Стабильность зон по месяцам
print(f'\n{"=" * 60}')
print('СТАБИЛЬНОСТЬ ЗОН (RSI < 35)')
print(f'{"=" * 60}')

rsi_zone = df[(df['rsi'] >= 25) & (df['rsi'] < 35)]
for m in months:
    subset = rsi_zone[rsi_zone['month'] == m]
    if len(subset) >= 10:
        wr = (subset['target'] == 1).mean() * 100
        pos = subset[subset['pnl_pct'] > 0]['pnl_pct'].sum()
        neg = abs(subset[subset['pnl_pct'] < 0]['pnl_pct'].sum())
        pf = pos / neg if neg > 0 else 99
        name = subset['month_name'].iloc[0]
        print(f'  {name}: WR={wr:.1f}% PF={pf:.2f} n={len(subset)}')

# Стабильность лучших часов
print(f'\n{"=" * 60}')
print('СТАБИЛЬНОСТЬ ЛУЧШИХ ЧАСОВ')
print(f'{"=" * 60}')

for h in [3, 4, 6, 8, 11, 14, 16, 18, 23]:
    hour_data = df[df['hour'] == h]
    monthly_wr = []
    for m in months:
        subset = hour_data[hour_data['month'] == m]
        if len(subset) >= 5:
            monthly_wr.append((subset['target'] == 1).mean())

    if len(monthly_wr) >= 2:
        avg_wr = np.mean(monthly_wr) * 100
        std_wr = np.std(monthly_wr) * 100
        stability = '✅ STABLE' if std_wr < 10 else ('⚠️ VARIABLE' if std_wr < 20 else '❌ UNSTABLE')
        print(f'  {h:02d}:00: avgWR={avg_wr:.1f}% std={std_wr:.1f}% n_months={len(monthly_wr)} {stability}')

# Итог
print(f'\n{"=" * 80}')
print('ИТОГ ПО СТАБИЛЬНОСТИ')
print('=' * 80)

wr_values = [s['wr'] for s in monthly_stats]
if len(wr_values) >= 2:
    avg_wr = np.mean(wr_values)
    std_wr = np.std(wr_values)
    print(f'  Средний WR по месяцам: {avg_wr:.1f}%')
    print(f'  Стандартное отклонение: {std_wr:.1f}%')
    print(f'  {"✅ СТАБИЛЬНО" if std_wr < 5 else "⚠️ УМЕРЕННО" if std_wr < 10 else "❌ НЕСТАБИЛЬНО"}')

with open(results_dir / '07_stability.json', 'w', encoding='utf-8') as f:
    json.dump(monthly_stats, f, indent=2)
print(f'\n✅ Сохранено: results/07_stability.json')