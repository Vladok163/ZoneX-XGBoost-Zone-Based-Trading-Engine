"""Исследование #1: Nonlinear Zones — в каких зонах признаков положительное expectancy."""
import pandas as pd
import numpy as np

df = pd.read_csv('ml_dataset_v2.csv')

FEATURES = {
    'rsi': {
        'bins': [0, 25, 35, 45, 55, 65, 75, 100],
        'labels': ['<25', '25-35', '35-45', '45-55', '55-65', '65-75', '75+'],
    },
    'volume_ratio': {
        'bins': [0, 0.3, 0.5, 0.6, 0.9, 1.2, 1.5, 2.0, 10],
        'labels': ['<0.3', '0.3-0.5', '0.5-0.6', '0.6-0.9', '0.9-1.2', '1.2-1.5', '1.5-2.0', '2.0+'],
    },
    'momentum_4': {
        'bins': [-100, -15, -5, 0, 5, 10, 15, 100],
        'labels': ['<-15', '-15..-5', '-5..0', '0..5', '5..10', '10..15', '15+'],
    },
    'compression': {
        'bins': [0, 20, 40, 50, 60, 70, 80, 100],
        'labels': ['<20', '20-40', '40-50', '50-60', '60-70', '70-80', '80+'],
    },
    'atr_pct': {
        'bins': [0, 0.5, 1.0, 2.0, 3.0, 5.0, 10],
        'labels': ['<0.5', '0.5-1', '1-2', '2-3', '3-5', '5+'],
    },
    'candle_efficiency': {
        'bins': [0, 0.2, 0.4, 0.6, 0.8, 1.0],
        'labels': ['<0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0'],
    },
}

print('=' * 80)
print('ИССЛЕДОВАНИЕ #1: NONLINEAR ZONES')
print('Где expectancy положительное?')
print('=' * 80)

for feature, cfg in FEATURES.items():
    print(f'\n{"=" * 60}')
    print(f'  {feature.upper()}')
    print(f'{"=" * 60}')
    print(f'{"Зона":<15s} {"N":>6s} {"WR":>7s} {"avgPnL":>8s} {"PF":>7s} {"sumR":>8s} {"Стабильность"}')
    print('-' * 65)

    df['zone'] = pd.cut(df[feature], bins=cfg['bins'], labels=cfg['labels'])

    for zone in cfg['labels']:
        subset = df[df['zone'] == zone]
        if len(subset) < 15:
            continue

        n = len(subset)
        wr = (subset['target'] == 1).mean() * 100
        avg_pnl = subset['pnl_pct'].mean()
        pos = subset[subset['pnl_pct'] > 0]['pnl_pct'].sum()
        neg = abs(subset[subset['pnl_pct'] < 0]['pnl_pct'].sum())
        pf = pos / neg if neg > 0 else 99

        # Стабильность: смотрим по месяцам
        stability = '?'
        if 'date' in df.columns:
            subset_copy = subset.copy()
            subset_copy['month'] = pd.to_datetime(subset_copy['date']).dt.month
            months = subset_copy['month'].unique()
            monthly_wr = []
            for m in months:
                ms = subset_copy[subset_copy['month'] == m]
                if len(ms) >= 5:
                    monthly_wr.append((ms['target'] == 1).mean())
            if len(monthly_wr) >= 2 and np.std(monthly_wr) < 0.15:
                stability = '✅ stable'
            elif len(monthly_wr) >= 1:
                stability = '⚠️'

        bar = '█' * int(wr / 5)
        marker = ' ← ЛУЧШИЙ' if wr >= 52 and pf >= 1.2 else ''
        print(
            f'{zone:<15s} {n:>6d} {wr:>6.1f}% {avg_pnl:>+7.2f}% {pf:>6.2f} {subset["pnl_pct"].sum():>+7.1f}  {stability}{marker} {bar}')
# Итоговая таблица лучших зон
print(f'\n{"=" * 80}')
print('СВОДКА: ЛУЧШИЕ ЗОНЫ (WR >= 52%, PF >= 1.2, n >= 20)')
print('=' * 80)

best_zones = []
for feature, cfg in FEATURES.items():
    df['zone'] = pd.cut(df[feature], bins=cfg['bins'], labels=cfg['labels'])
    for zone in cfg['labels']:
        subset = df[df['zone'] == zone]
        if len(subset) >= 20:
            wr = (subset['target'] == 1).mean() * 100
            pos = subset[subset['pnl_pct'] > 0]['pnl_pct'].sum()
            neg = abs(subset[subset['pnl_pct'] < 0]['pnl_pct'].sum())
            pf = pos / neg if neg > 0 else 99
            if wr >= 52 and pf >= 1.2:
                best_zones.append({
                    'feature': feature, 'zone': zone,
                    'n': len(subset), 'wr': round(wr, 1),
                    'pf': round(pf, 2), 'avg_pnl': round(subset['pnl_pct'].mean(), 2)
                })

for z in sorted(best_zones, key=lambda x: x['wr'], reverse=True):
    print(
        f"  {z['feature']:<20s} {z['zone']:<12s} WR={z['wr']:.1f}% PF={z['pf']:.2f} n={z['n']} avgPnL={z['avg_pnl']:+.2f}%")
# Добавь в конец файла 01_nonlinear_zones.py:

import json
from pathlib import Path

results_dir = Path('results')
results_dir.mkdir(exist_ok=True)

# Сохраняем все результаты
output = {
    'title': 'Nonlinear Zones Research',
    'description': 'Устойчивые зоны с положительным expectancy',
    'best_zones': best_zones,
    'detailed': {}
}

for feature, cfg in FEATURES.items():
    df['zone'] = pd.cut(df[feature], bins=cfg['bins'], labels=cfg['labels'])
    feature_data = []
    for zone in cfg['labels']:
        subset = df[df['zone'] == zone]
        if len(subset) >= 15:
            pos = subset[subset['pnl_pct']>0]['pnl_pct'].sum()
            neg = abs(subset[subset['pnl_pct']<0]['pnl_pct'].sum())
            feature_data.append({
                'zone': zone,
                'n': len(subset),
                'wr': round((subset['target']==1).mean()*100, 1),
                'avg_pnl': round(subset['pnl_pct'].mean(), 2),
                'pf': round(pos/neg, 2) if neg > 0 else 99,
                'sum_r': round(subset['pnl_pct'].sum(), 1),
                'stable': True  # упрощённо
            })
    output['detailed'][feature] = feature_data

with open(results_dir / '01_nonlinear_zones.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f'\n✅ Результаты сохранены: results/01_nonlinear_zones.json')