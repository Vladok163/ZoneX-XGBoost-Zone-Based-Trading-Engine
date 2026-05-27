"""Исследование #4: Archetypes — прибыльность рыночных архетипов."""
import pandas as pd
import numpy as np
import json
from pathlib import Path

df = pd.read_csv('ml_dataset_v2.csv')
results_dir = Path('results')

print('=' * 80)
print('ИССЛЕДОВАНИЕ #4: ARCHETYPES')
print('Прибыльность повторяющихся рыночных состояний')
print('=' * 80)


# Определяем архетипы по зонам (упрощённо, без pattern_engine)
def classify_archetype(row):
    rsi = row['rsi']
    vol = row['volume_ratio']
    imp = abs(row['momentum_4'])
    comp = row['compression']
    atr = row['atr_pct']

    # Quiet Coil: compression 70-100%, vol 0.5-1.2, imp 3-10%, RSI 35-55
    if comp >= 70 and 0.5 <= vol <= 1.2 and 3 <= imp <= 10 and 35 <= rsi <= 55:
        return 'quiet_coil'

    # Exhaustion Spike: imp > 20%, vol > 2.0, RSI > 65 or < 35
    if imp > 20 and vol > 2.0 and (rsi > 65 or rsi < 35):
        return 'exhaustion_spike'

    # Trend Continuation: imp 5-15%, vol 0.7-1.5, comp 30-60%
    if 5 <= imp <= 15 and 0.7 <= vol <= 1.5 and 30 <= comp <= 60:
        return 'trend_continuation'

    # Failed Breakout: comp 60-95%, vol 0.3-0.7, imp 2-8%
    if comp >= 60 and 0.3 <= vol <= 0.7 and 2 <= imp <= 8:
        return 'failed_breakout'

    # Liquidity Sweep: imp 10-30%, vol 1.5-5.0, comp 10-40%
    if 10 <= imp <= 30 and 1.5 <= vol <= 5.0 and 10 <= comp <= 40:
        return 'liquidity_sweep'

    return 'none'


df['archetype'] = df.apply(classify_archetype, axis=1)

# Статистика по архетипам
archetypes = df['archetype'].value_counts()
print(f'\nРаспределение архетипов:')
for arch, n in archetypes.items():
    print(f'  {arch}: {n}')

print(f'\n{"=" * 60}')
print(f'{"Архетип":<25s} {"N":>6s} {"WR":>7s} {"avgPnL":>8s} {"PF":>7s} {"sumR":>8s} {"avgDur":>7s}')
print('-' * 75)

results = []
for arch in archetypes.index:
    subset = df[df['archetype'] == arch]
    if len(subset) < 15:
        continue

    n = len(subset)
    wr = (subset['target'] == 1).mean() * 100
    avg_pnl = subset['pnl_pct'].mean()
    pos = subset[subset['pnl_pct'] > 0]['pnl_pct'].sum()
    neg = abs(subset[subset['pnl_pct'] < 0]['pnl_pct'].sum())
    pf = pos / neg if neg > 0 else 99
    sum_r = subset['pnl_pct'].sum()

    # По режимам
    regime_stats = {}
    for reg in subset['btc_regime'].dropna().unique():
        rs = subset[subset['btc_regime'] == reg]
        if len(rs) >= 10:
            regime_stats[reg] = {
                'n': len(rs),
                'wr': round((rs['target'] == 1).mean() * 100, 1),
                'pf': round(rs[rs['pnl_pct'] > 0]['pnl_pct'].sum() /
                            abs(rs[rs['pnl_pct'] < 0]['pnl_pct'].sum()), 2) if rs[rs['pnl_pct'] < 0][
                                                                                   'pnl_pct'].sum() != 0 else 99
            }

    # По таймфреймам
    tf_stats = {}
    for tf in subset['tf'].unique():
        ts = subset[subset['tf'] == tf]
        if len(ts) >= 10:
            tf_stats[tf] = {
                'n': len(ts),
                'wr': round((ts['target'] == 1).mean() * 100, 1)
            }

    bar = '█' * int(wr / 5)
    mark = ' ✅' if wr >= 52 and pf >= 1.2 else ''
    print(f'{arch:<25s} {n:>6d} {wr:>6.1f}% {avg_pnl:>+7.2f}% {pf:>6.2f} {sum_r:>+7.1f} {"N/A":>7s}{mark} {bar}')

    if regime_stats:
        print(f'  По режимам:')
        for reg, rs in regime_stats.items():
            print(f'    {str(reg):<15s}: WR={rs["wr"]:.1f}% PF={rs["pf"]:.2f} n={rs["n"]}')

    if tf_stats:
        print(f'  По ТФ:')
        for tf, ts in tf_stats.items():
            print(f'    {tf:<8s}: WR={ts["wr"]:.1f}% n={ts["n"]}')

    results.append({
        'archetype': arch, 'n': n, 'wr': round(wr, 1),
        'avg_pnl': round(avg_pnl, 2), 'pf': round(pf, 2),
        'sum_r': round(sum_r, 1),
        'regime_stats': regime_stats,
        'tf_stats': tf_stats,
    })

# Сохраняем
with open(results_dir / '04_archetypes.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f'\n✅ Сохранено: results/04_archetypes.json')