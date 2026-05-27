"""Симуляция v5 — без Флага, только лучшие часы/дни."""
import pandas as pd
import numpy as np

df = pd.read_csv('ml_dataset.csv')

IDEAL = {
    'sl_pct': 2.0,
    'tp_pct': 4.5,
    'min_volume': 0.3,
    'max_volume': 2.5,
    'rsi_min': 30,
    'rsi_max': 70,
    'atr_min': 0.3,
    'atr_max': 5.0,
    'momentum_min': -15,
    'momentum_max': 15,
    'good_hours': [3, 8, 12, 16, 18],
    'bad_hours': [7, 10, 13, 14, 23],
    'good_days': [0, 2, 3],  # Пн, Ср, Чт
    'bad_days': [1, 4, 5, 6], # Вт, Пт, Сб, Вс
    'good_patterns': ['Голова и плечи', 'Двойная вершина'],
}

def passes_filters(row):
    if row['volume_ratio'] < IDEAL['min_volume'] or row['volume_ratio'] > IDEAL['max_volume']:
        return False
    if not (IDEAL['rsi_min'] <= row['rsi'] <= IDEAL['rsi_max']):
        return False
    if row['atr_pct'] < IDEAL['atr_min'] or row['atr_pct'] > IDEAL['atr_max']:
        return False
    if row['momentum_4'] < IDEAL['momentum_min'] or row['momentum_4'] > IDEAL['momentum_max']:
        return False
    if row['hour'] in IDEAL['bad_hours']:
        return False
    if row['dow'] in IDEAL['bad_days']:
        return False
    if row['pattern_name'] not in IDEAL['good_patterns']:
        return False
    if row['hour'] not in IDEAL['good_hours']:
        return False
    return True

results = []
passed = 0
total = 0

for _, row in df.iterrows():
    if row['pattern_found'] == 0:
        continue
    total += 1
    if passes_filters(row):
        passed += 1
        pnl = row['pnl_pct']
        win = pnl > 0
        results.append({
            'pnl': pnl, 'win': win,
            'pattern': row['pattern_name'],
            'hour': row['hour'], 'dow': row['dow']
        })

if results:
    df_r = pd.DataFrame(results)
    wr = df_r['win'].mean() * 100
    avg_pnl = df_r['pnl'].mean()
    total_r = df_r['pnl'].sum()
    pos = df_r[df_r['pnl']>0]['pnl'].sum()
    neg = abs(df_r[df_r['pnl']<0]['pnl'].sum())
    pf = pos/neg if neg > 0 else 99

    print(f'Сигналов: {total}')
    print(f'Прошли: {passed} ({passed/total*100:.0f}%)')
    print(f'Сделок/день: {len(df_r)/90:.1f}')
    print(f'Винрейт: {wr:.1f}%')
    print(f'Средний PnL: {avg_pnl:+.2f}%')
    print(f'Суммарный R: {total_r:+.1f}')
    print(f'Profit Factor: {pf:.2f}')

    print(f'\nПо паттернам:')
    for pat in df_r['pattern'].unique():
        sub = df_r[df_r['pattern'] == pat]
        print(f'  {pat}: n={len(sub)}, WR={sub["win"].mean()*100:.1f}%')

    print(f'\nПо часам:')
    for h in sorted(df_r['hour'].unique()):
        sub = df_r[df_r['hour'] == h]
        print(f'  {h:02d}:00: n={len(sub)}, WR={sub["win"].mean()*100:.1f}%')

    print(f'\nПо дням:')
    days = {0:'Пн',1:'Вт',2:'Ср',3:'Чт',4:'Пт',5:'Сб',6:'Вс'}
    for d in sorted(df_r['dow'].unique()):
        sub = df_r[df_r['dow'] == d]
        print(f'  {days[d]}: n={len(sub)}, WR={sub["win"].mean()*100:.1f}%')
else:
    print('Нет сделок')