"""Исследование #2: Interactions — как признаки меняют значение друг друга."""
import pandas as pd
import numpy as np
import json
from pathlib import Path

df = pd.read_csv('ml_dataset_v2.csv')
results_dir = Path('results')
results_dir.mkdir(exist_ok=True)

print('=' * 80)
print('ИССЛЕДОВАНИЕ #2: INTERACTIONS')
print('Условные зависимости между признаками')
print('=' * 80)

# ============================================================
# 1. COMPRESSION × VOLUME
# ============================================================
print(f'\n{"="*60}')
print('  1. COMPRESSION × VOLUME')
print(f'{"="*60}')

comp_bins = [0, 40, 60, 80, 100]
comp_labels = ['<40%', '40-60%', '60-80%', '80%+']
vol_bins = [0, 0.5, 1.0, 2.0, 10]
vol_labels = ['<0.5', '0.5-1.0', '1.0-2.0', '2.0+']

df['comp_z'] = pd.cut(df['compression'], bins=comp_bins, labels=comp_labels)
df['vol_z'] = pd.cut(df['volume_ratio'], bins=vol_bins, labels=vol_labels)

header = "Comp\\Vol"
print(f'{header:<12s}', end='')
for vl in vol_labels:
    print(f'{vl:>12s}', end='')
print(f' {"":>6s}')
print('-' * 65)

interactions_comp_vol = []
for cl in comp_labels:
    print(f'{cl:<12s}', end='')
    for vl in vol_labels:
        subset = df[(df['comp_z'] == cl) & (df['vol_z'] == vl)]
        if len(subset) >= 15:
            wr = (subset['target']==1).mean()*100
            print(f'  WR={wr:>5.1f}%', end='')
            interactions_comp_vol.append({
                'compression': cl, 'volume': vl,
                'n': len(subset), 'wr': round(wr,1),
                'avg_pnl': round(subset['pnl_pct'].mean(),2)
            })
        else:
            print(f'  {"n<15":>12s}', end='')
    print()

# ============================================================
# 2. RSI × REGIME
# ============================================================
print(f'\n{"="*60}')
print('  2. RSI × BTC REGIME')
print(f'{"="*60}')

rsi_bins = [0, 35, 50, 65, 100]
rsi_labels = ['<35', '35-50', '50-65', '65+']
df['rsi_z'] = pd.cut(df['rsi'], bins=rsi_bins, labels=rsi_labels)

regimes = df['btc_regime'].dropna().unique()
header = "RSI\\Regime"
print(f'{header:<12s}', end='')
for reg in sorted(regimes):
    print(f'{str(reg):>12s}', end='')
print()
print('-' * 50)

for rl in rsi_labels:
    print(f'{rl:<12s}', end='')
    for reg in sorted(regimes):
        subset = df[(df['rsi_z'] == rl) & (df['btc_regime'] == reg)]
        if len(subset) >= 10:
            wr = (subset['target']==1).mean()*100
            print(f'  WR={wr:>5.1f}%', end='')
        else:
            print(f'  {"n<10":>12s}', end='')
    print()

# ============================================================
# 3. IMPULSE × PATTERN
# ============================================================
print(f'\n{"="*60}')
print('  3. IMPULSE × PATTERN')
print(f'{"="*60}')

imp_bins = [0, 5, 10, 15, 100]
imp_labels = ['<5%', '5-10%', '10-15%', '15%+']
df['imp_z'] = pd.cut(df['momentum_4'].abs(), bins=imp_bins, labels=imp_labels)

top_patterns = df['pattern_name'].value_counts().head(5).index

header = "Imp\\Pattern"
print(f'{header:<12s}', end='')
for pat in top_patterns:
    print(f'{str(pat)[:12]:>12s}', end='')
print()
print('-' * 65)

for il in imp_labels:
    print(f'{il:<12s}', end='')
    for pat in top_patterns:
        subset = df[(df['imp_z'] == il) & (df['pattern_name'] == pat)]
        if len(subset) >= 10:
            wr = (subset['target']==1).mean()*100
            print(f'  WR={wr:>5.1f}%', end='')
        else:
            print(f'  {"n<10":>12s}', end='')
    print()

# ============================================================
# 4. ATR × TIMEFRAME
# ============================================================
print(f'\n{"="*60}')
print('  4. ATR × TIMEFRAME')
print(f'{"="*60}')

atr_bins = [0, 1, 2, 3, 10]
atr_labels = ['<1%', '1-2%', '2-3%', '3%+']
df['atr_z'] = pd.cut(df['atr_pct'], bins=atr_bins, labels=atr_labels)

for tf in sorted(df['tf'].unique()):
    print(f'\n  TF={tf}:')
    for al in atr_labels:
        subset = df[(df['atr_z'] == al) & (df['tf'] == tf)]
        if len(subset) >= 10:
            wr = (subset['target']==1).mean()*100
            print(f'    ATR {al:<8s}: WR={wr:.1f}% (n={len(subset)})')

# ============================================================
# 5. SESSION × DAY OF WEEK
# ============================================================
print(f'\n{"="*60}')
print('  5. HOUR × DAY OF WEEK')
print(f'{"="*60}')

days = {0:'Пн',1:'Вт',2:'Ср',3:'Чт',4:'Пт',5:'Сб',6:'Вс'}
best_interactions = []

for d in range(7):
    day_data = df[df['dow'] == d]
    for h in range(24):
        subset = day_data[day_data['hour'] == h]
        if len(subset) >= 10:
            wr = (subset['target']==1).mean()*100
            pos = subset[subset['pnl_pct']>0]['pnl_pct'].sum()
            neg = abs(subset[subset['pnl_pct']<0]['pnl_pct'].sum())
            pf = pos/neg if neg > 0 else 99
            if wr >= 55 and pf >= 1.3:
                best_interactions.append({
                    'day': days[d], 'hour': h,
                    'n': len(subset), 'wr': round(wr,1), 'pf': round(pf,2)
                })
                print(f'  {days[d]} {h:02d}:00: WR={wr:.1f}% PF={pf:.2f} n={len(subset)}')

# ============================================================
# СВОДКА
# ============================================================
print(f'\n{"="*80}')
print('ТОП-10 INTERACTIONS (WR >= 55%, PF >= 1.3)')
print('=' * 80)
for x in sorted(best_interactions, key=lambda x: x['wr'], reverse=True)[:10]:
    print(f"  {x['day']} {x['hour']:02d}:00: WR={x['wr']:.1f}% PF={x['pf']:.2f} n={x['n']}")

# Сохраняем
output = {
    'title': 'Feature Interactions Research',
    'compression_volume': interactions_comp_vol,
    'best_hour_day': best_interactions,
}
with open(results_dir / '02_interactions.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f'\n✅ Сохранено: results/02_interactions.json')