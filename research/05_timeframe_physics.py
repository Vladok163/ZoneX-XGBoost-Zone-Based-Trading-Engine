"""Исследование #5: Timeframe Physics — рынок на разных ТФ."""
import pandas as pd
import numpy as np
import json
from pathlib import Path

df = pd.read_csv('ml_dataset_v2.csv')
results_dir = Path('results')

print('=' * 80)
print('ИССЛЕДОВАНИЕ #5: TIMEFRAME PHYSICS')
print('Как рынок ведёт себя на разных таймфреймах')
print('=' * 80)

timeframes = sorted(df['tf'].unique())

for tf in timeframes:
    subset = df[df['tf'] == tf]
    if len(subset) < 100:
        continue

    print(f'\n{"=" * 60}')
    print(f'  TIMEFRAME: {tf} (n={len(subset)})')
    print(f'  Base WR: {(subset["target"] == 1).mean() * 100:.1f}%')
    print(f'  Base avgPnL: {subset["pnl_pct"].mean():+.2f}%')
    print(f'{"=" * 60}')

    # RSI зоны
    print(f'\n  RSI зоны:')
    for rsi_lo, rsi_hi, label in [(0, 35, '<35'), (35, 45, '35-45'), (45, 55, '45-55'), (55, 65, '55-65'),
                                  (65, 100, '65+')]:
        s = subset[(subset['rsi'] >= rsi_lo) & (subset['rsi'] < rsi_hi)]
        if len(s) >= 10:
            wr = (s['target'] == 1).mean() * 100
            mark = ' ✅' if wr >= 52 else ''
            print(f'    {label:<8s}: WR={wr:.1f}% n={len(s)}{mark}')

    # Volume
    print(f'\n  Volume зоны:')
    for v_lo, v_hi, label in [(0, 0.5, '<0.5'), (0.5, 1.0, '0.5-1.0'), (1.0, 2.0, '1.0-2.0'), (2.0, 10, '2.0+')]:
        s = subset[(subset['volume_ratio'] >= v_lo) & (subset['volume_ratio'] < v_hi)]
        if len(s) >= 10:
            wr = (s['target'] == 1).mean() * 100
            mark = ' ✅' if wr >= 52 else ''
            print(f'    {label:<8s}: WR={wr:.1f}% n={len(s)}{mark}')

    # Импульс
    print(f'\n  Impulse зоны:')
    for i_lo, i_hi, label in [(0, 5, '<5%'), (5, 10, '5-10%'), (10, 15, '10-15%'), (15, 100, '15%+')]:
        s = subset[(subset['momentum_4'].abs() >= i_lo) & (subset['momentum_4'].abs() < i_hi)]
        if len(s) >= 10:
            wr = (s['target'] == 1).mean() * 100
            mark = ' ✅' if wr >= 52 else ''
            print(f'    {label:<8s}: WR={wr:.1f}% n={len(s)}{mark}')

    # Compression
    print(f'\n  Compression:')
    for c_lo, c_hi, label in [(0, 40, '<40%'), (40, 60, '40-60%'), (60, 80, '60-80%'), (80, 100, '80%+')]:
        s = subset[(subset['compression'] >= c_lo) & (subset['compression'] < c_hi)]
        if len(s) >= 10:
            wr = (s['target'] == 1).mean() * 100
            mark = ' ✅' if wr >= 52 else ''
            print(f'    {label:<8s}: WR={wr:.1f}% n={len(s)}{mark}')

    # ATR
    print(f'\n  ATR зоны:')
    for a_lo, a_hi, label in [(0, 1, '<1%'), (1, 2, '1-2%'), (2, 3, '2-3%'), (3, 10, '3%+')]:
        s = subset[(subset['atr_pct'] >= a_lo) & (subset['atr_pct'] < a_hi)]
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
    print(f'\n  Лучшие часы (WR>=55%):')
    for h in range(24):
        s = subset[subset['hour'] == h]
        if len(s) >= 10:
            wr = (s['target'] == 1).mean() * 100
            if wr >= 55:
                print(f'    {h:02d}:00: WR={wr:.1f}% n={len(s)} ✅')

# Сводка
print(f'\n{"=" * 80}')
print('СВОДКА ПО ТАЙМФРЕЙМАМ')
print('=' * 80)

for tf in timeframes:
    subset = df[df['tf'] == tf]
    if len(subset) < 50:
        continue
    wr = (subset['target'] == 1).mean() * 100
    avg_pnl = subset['pnl_pct'].mean()
    print(f'\n{tf}: WR={wr:.1f}% avgPnL={avg_pnl:+.2f}% n={len(subset)}')

    # Лучшие зоны для этого ТФ
    best = []
    for rsi_lo, rsi_hi, label in [(0, 35, 'RSI<35'), (35, 45, 'RSI35-45'), (45, 55, 'RSI45-55')]:
        s = subset[(subset['rsi'] >= rsi_lo) & (subset['rsi'] < rsi_hi)]
        if len(s) >= 15:
            w = (s['target'] == 1).mean() * 100
            if w >= 52:
                best.append(f'{label}:{w:.0f}%')

    for v_lo, v_hi, label in [(0.5, 1.0, 'VOL0.5-1'), (0, 0.5, 'VOL<0.5')]:
        s = subset[(subset['volume_ratio'] >= v_lo) & (subset['volume_ratio'] < v_hi)]
        if len(s) >= 15:
            w = (s['target'] == 1).mean() * 100
            if w >= 52:
                best.append(f'{label}:{w:.0f}%')

    print(f'  Лучшие зоны: {", ".join(best) if best else "нет устойчивых"}')

with open(results_dir / '05_timeframe_physics.json', 'w') as f:
    json.dump({'timeframes': list(timeframes)}, f)
print(f'\n✅ Сохранено: results/05_timeframe_physics.json')