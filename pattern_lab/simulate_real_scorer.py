"""Grid Search v2 — все фичи + XGBoost важность."""
import pandas as pd
import numpy as np
import math
from itertools import product
import json

df = pd.read_csv('ml_dataset_v2.csv')
print(f'Загружено: {len(df)} строк, {len(df.columns)} колонок')

# ============================================================
# СЕТКА ПАРАМЕТРОВ
# ============================================================
# В grid_search.py замени GRID на:
GRID = {
    'rsi_long_lo': [40, 45],
    'rsi_long_hi': [55, 60],
    'vol_lo': [0.5, 0.6],
    'vol_hi': [0.9, 1.2],
    'imp_lo': [5],
    'imp_hi': [12],
    'comp_lo': [50, 70],
    'comp_hi': [90],
    'gw_technical': [0.25, 0.30],
    'gw_pattern': [0.15, 0.20],
    'gw_market': [0.15, 0.20],
    'prob_threshold': [0.65, 0.70],
    'sigmoid_shift': [6.0, 6.5],
    'good_patterns': [('Голова и плечи', 'Двойная вершина')],
    # Остальное — дефолты
    'rsi_short_lo': [35],
    'rsi_short_hi': [50],
    'mom12_lo': [-5],
    'mom12_hi': [15],
    'atr_lo': [0.3],
    'atr_hi': [5.0],
    'candle_eff_min': [0.3],
    'vol_trend_min': [0.5],
    'rm_chop': [0.90],
    'rm_trend_down': [0.7],
    'use_bad_hours': [True],
    'use_weekend_filter': [True],
}

SESSION_GREAT = [3, 4, 6, 8, 11, 14, 16, 18, 23]
SESSION_GOOD  = [0, 5, 7, 10, 12, 13, 21]
SESSION_BAD   = [1, 2, 9, 15, 17, 19, 20, 22]

PATTERN_SCORES = {
    "flag": 0.8, "double_top": 0.45, "double_bottom": 0.45,
    "head_shoulders": 0.55, "inv_head_shoulders": 0.40,
}

def evaluate_params(params):
    results = []

    for _, row in df.iterrows():
        if row['pattern_found'] == 0: continue
        if row['pattern_name'] not in params['good_patterns']: continue

        # === ХАРД-ФИЛЬТРЫ ===
        if abs(row['momentum_4']) > 15: continue
        if row['volume_ratio'] > 4: continue
        if row['rsi'] > 85 or row['rsi'] < 15: continue
        if params['use_bad_hours'] and row['hour'] in SESSION_BAD: continue
        if params['use_weekend_filter'] and row['is_weekend']: continue

        # === ФИЧИ ===
        if row['candle_efficiency'] < params['candle_eff_min']: continue
        if row['volume_trend'] < params['vol_trend_min']: continue
        if not (params['mom12_lo'] <= row['momentum_12'] <= params['mom12_hi']): continue
        if not (params['atr_lo'] <= row['atr_pct'] <= params['atr_hi']): continue

        direction = row['pattern_direction'] if row['pattern_direction'] != 'none' else 'bull'

        # RSI
        rsi = row['rsi']
        if direction in ('bull', 'long'):
            rsi_sc = 0.9 if params['rsi_long_lo'] <= rsi <= params['rsi_long_hi'] else (0.5 if rsi < params['rsi_long_lo'] else 0.3)
        else:
            rsi_sc = 0.9 if params['rsi_short_lo'] <= rsi <= params['rsi_short_hi'] else (0.5 if rsi < params['rsi_short_lo'] else 0.3)

        # Volume
        vr = row['volume_ratio']
        vol_sc = 0.9 if params['vol_lo'] <= vr <= params['vol_hi'] else (0.2 if 1.0 < vr <= 1.5 else 0.5)

        # Impulse
        imp = abs(row['momentum_4'])
        imp_sc = 0.9 if params['imp_lo'] <= imp <= params['imp_hi'] else (0.2 if imp > params['imp_hi'] else 0.5)

        # Compression
        comp = row['compression']
        comp_sc = 0.9 if params['comp_lo'] <= comp <= params['comp_hi'] else (0.5 if comp > params['comp_hi'] else 0.3)

        # ATR
        atr = row['atr_pct']
        atr_sc = 0.9 if params['atr_lo'] <= atr <= params['atr_hi'] else 0.4

        # Session
        hour = row['hour']
        ses_sc = 1.0 if hour in SESSION_GREAT else (0.6 if hour in SESSION_GOOD else 0.1)

        # Day
        dow = row['dow']
        day_sc = 1.0 if dow in [2, 5] else (0.3 if dow == 1 else 0.6)

        # BTC regime
        regime = str(row['btc_regime'])
        btc_sc = 0.8 if regime in ('trend_up', 'normal') else (0.6 if regime == 'chop' else 0.3)

        # Pattern
        pat = PATTERN_SCORES.get(row['pattern_en'], 0.3)
        pat_w = row['pattern_weight'] if not pd.isna(row['pattern_weight']) else 0
        if pat_w >= 0.7: pat *= 1.3

        # === ГРУППЫ ===
        tech = (rsi_sc * 0.30 + vol_sc * 0.25 + atr_sc * 0.15 + imp_sc * 0.15 +
                row['candle_efficiency'] * 0.10 + row['volume_trend'] * 0.05) * 10
        pat_grp = (pat * 0.35 + imp_sc * 0.30 + comp_sc * 0.35) * 10
        mkt = (ses_sc * 0.40 + btc_sc * 0.35 + day_sc * 0.15 + row['atr_percentile'] * 0.10) * 10

        gw_sum = params['gw_technical'] + params['gw_pattern'] + params['gw_market'] + 0.10 + 0.25
        raw = (tech * params['gw_technical'] + pat_grp * params['gw_pattern'] +
               mkt * params['gw_market'] + 5.0 * 0.10 + 5.0 * 0.25) / gw_sum

        # Regime multiplier
        if regime == 'trend_up': rmult = 1.1
        elif regime == 'trend_down': rmult = params['rm_trend_down']
        elif regime == 'chop': rmult = params['rm_chop']
        else: rmult = 1.0
        raw *= rmult

        # Probability
        prob = 1 / (1 + math.exp(-(raw - params['sigmoid_shift'])))

        if prob >= params['prob_threshold']:
            results.append({'pnl': row['pnl_pct'], 'win': row['pnl_pct'] > 0})

    if len(results) < 20:
        return None

    df_r = pd.DataFrame(results)
    wr = df_r['win'].mean()
    pos = df_r[df_r['pnl']>0]['pnl'].sum()
    neg = abs(df_r[df_r['pnl']<0]['pnl'].sum())
    pf = pos/neg if neg > 0 else 99

    return {
        'n': len(results), 'wr': round(wr * 100, 1),
        'avg_pnl': round(df_r['pnl'].mean(), 2),
        'pf': round(pf, 2), 'tpd': round(len(results)/90, 1),
    }

# ============================================================
# ПЕРЕБОР
# ============================================================
keys = list(GRID.keys())
values = list(GRID.values())
total = 1
for v in values: total *= len(v)
print(f'Комбинаций: {total}')

best = None
all_results = []
count = 0

for combo in product(*values):
    params = dict(zip(keys, combo))
    result = evaluate_params(params)
    count += 1

    if result:
        all_results.append({**result, **params})
        if best is None or result['wr'] > best['wr']:
            best = {**result, **params}
            print(f'#{count}: WR={best["wr"]:.1f}% PF={best["pf"]:.2f} n={best["n"]} tpd={best["tpd"]}/д')

all_results.sort(key=lambda x: (x['wr'], x['pf']), reverse=True)

print(f'\n{"="*80}')
print(f'ТОП-15 КОМБИНАЦИЙ')
print(f'{"="*80}')
print(f'{"WR":>6s} {"PF":>6s} {"N":>5s} {"TPD":>5s} {"RSI_L":>8s} {"VOL":>12s} {"IMP":>12s} {"COMP":>15s} {"THR":>5s} {"GW":>12s}')
print('-'*95)

for r in all_results[:15]:
    print(f'{r["wr"]:>5.1f}% {r["pf"]:>5.2f} {r["n"]:>5d} {r["tpd"]:>4.1f}/д '
          f'RSI={r["rsi_long_lo"]}-{r["rsi_long_hi"]} '
          f'V={r["vol_lo"]}-{r["vol_hi"]} '
          f'I={r["imp_lo"]}-{r["imp_hi"]} '
          f'C={r["comp_lo"]}-{r["comp_hi"]} '
          f'THR={r["prob_threshold"]} '
          f'T={r["gw_technical"]} P={r["gw_pattern"]} M={r["gw_market"]}')

# Сохраняем
with open('results/grid_search_results.json', 'w') as f:
    json.dump(all_results[:50], f, indent=2, default=str)

if best:
    print(f'\nЛУЧШАЯ КОМБИНАЦИЯ:')
    for k, v in best.items():
        if k not in ['good_patterns']:
            print(f'  {k}: {v}')
    print(f'  patterns: {best.get("good_patterns")}')