"""Исследование #3: Regime Dependence — какие фичи работают в каком режиме."""
import pandas as pd
import numpy as np
import json
from pathlib import Path

df = pd.read_csv('ml_dataset_v2.csv')
results_dir = Path('results')

print('=' * 80)
print('ИССЛЕДОВАНИЕ #3: REGIME DEPENDENCE')
print('Какие признаки важны в каждом режиме BTC?')
print('=' * 80)

regimes = df['btc_regime'].dropna().unique()

for regime in sorted(regimes):
    subset = df[df['btc_regime'] == regime]
    if len(subset) < 50:
        continue

    print(f'\n{"=" * 60}')
    print(f'  REGIME: {regime} (n={len(subset)})')
    print(f'  Base WR: {(subset["target"] == 1).mean() * 100:.1f}%')
    print(f'{"=" * 60}')

    # RSI зоны
    print(f'\n  RSI зоны:')
    for rsi_lo, rsi_hi, label in [(0, 35, '<35'), (35, 45, '35-45'), (45, 55, '45-55'), (55, 65, '55-65'),
                                  (65, 100, '65+')]:
        s = subset[(subset['rsi'] >= rsi_lo) & (subset['rsi'] < rsi_hi)]
        if len(s) >= 10:
            wr = (s['target'] == 1).mean() * 100
            pos = s[s['pnl_pct'] > 0]['pnl_pct'].sum()
            neg = abs(s[s['pnl_pct'] < 0]['pnl_pct'].sum())
            pf = pos / neg if neg > 0 else 99
            mark = ' ✅' if wr >= 52 and pf >= 1.2 else ''
            print(f'    {label:<8s}: WR={wr:.1f}% PF={pf:.2f} n={len(s)}{mark}')

    # Volume зоны
    print(f'\n  Volume зоны:')
    for v_lo, v_hi, label in [(0, 0.5, '<0.5'), (0.5, 1.0, '0.5-1.0'), (1.0, 2.0, '1.0-2.0'), (2.0, 10, '2.0+')]:
        s = subset[(subset['volume_ratio'] >= v_lo) & (subset['volume_ratio'] < v_hi)]
        if len(s) >= 10:
            wr = (s['target'] == 1).mean() * 100
            mark = ' ✅' if wr >= 52 else ''
            print(f'    {label:<8s}: WR={wr:.1f}% n={len(s)}{mark}')

    # Impulse зоны
    print(f'\n  Impulse зоны:')
    for i_lo, i_hi, label in [(0, 5, '<5%'), (5, 10, '5-10%'), (10, 15, '10-15%'), (15, 100, '15%+')]:
        s = subset[(subset['momentum_4'].abs() >= i_lo) & (subset['momentum_4'].abs() < i_hi)]
        if len(s) >= 10:
            wr = (s['target'] == 1).mean() * 100
            mark = ' ✅' if wr >= 52 else ''
            print(f'    {label:<8s}: WR={wr:.1f}% n={len(s)}{mark}')

    # Compression зоны
    print(f'\n  Compression зоны:')
    for c_lo, c_hi, label in [(0, 40, '<40%'), (40, 60, '40-60%'), (60, 80, '60-80%'), (80, 100, '80%+')]:
        s = subset[(subset['compression'] >= c_lo) & (subset['compression'] < c_hi)]
        if len(s) >= 10:
            wr = (s['target'] == 1).mean() * 100
            mark = ' ✅' if wr >= 52 else ''
            print(f'    {label:<8s}: WR={wr:.1f}% n={len(s)}{mark}')

    # Паттерны
    print(f'\n  Паттерны:')
    for pat in subset['pattern_name'].value_counts().head(5).index:
        s = subset[subset['pattern_name'] == pat]
        if len(s) >= 10:
            wr = (s['target'] == 1).mean() * 100
            pos = s[s['pnl_pct'] > 0]['pnl_pct'].sum()
            neg = abs(s[s['pnl_pct'] < 0]['pnl_pct'].sum())
            pf = pos / neg if neg > 0 else 99
            mark = ' ✅' if wr >= 52 and pf >= 1.2 else ''
            print(f'    {str(pat)[:25]:<25s}: WR={wr:.1f}% PF={pf:.2f} n={len(s)}{mark}')

    # Часы
    print(f'\n  Лучшие часы:')
    for h in range(24):
        s = subset[subset['hour'] == h]
        if len(s) >= 10:
            wr = (s['target'] == 1).mean() * 100
            if wr >= 55:
                print(f'    {h:02d}:00: WR={wr:.1f}% n={len(s)} ✅')

# Сводка: что работает в каждом режиме
print(f'\n{"=" * 80}')
print('СВОДКА: ЧТО РАБОТАЕТ В КАЖДОМ РЕЖИМЕ')
print('=' * 80)

summary = {}
for regime in sorted(regimes):
    subset = df[df['btc_regime'] == regime]
    if len(subset) < 50:
        continue

    # Ищем лучшие зоны
    best = []
    # RSI
    for rsi_lo, rsi_hi, label in [(0, 35, 'RSI<35'), (35, 45, 'RSI35-45'), (45, 55, 'RSI45-55'), (55, 65, 'RSI55-65'),
                                  (65, 100, 'RSI65+')]:
        s = subset[(subset['rsi'] >= rsi_lo) & (subset['rsi'] < rsi_hi)]
        if len(s) >= 15:
            wr = (s['target'] == 1).mean() * 100
            if wr >= 52:
                best.append(f'{label} WR={wr:.0f}%')

    print(f'\n{regime}: {", ".join(best) if best else "нет устойчивых зон"}')
    summary[regime] = {'n': len(subset), 'base_wr': round((subset['target'] == 1).mean() * 100, 1), 'best_zones': best}

with open(results_dir / '03_regime_dependence.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f'\n✅ Сохранено: results/03_regime_dependence.json')