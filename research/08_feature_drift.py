"""Исследование #8: Feature Importance Drift — как меняется важность фичей."""
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import json
from pathlib import Path

df = pd.read_csv('ml_dataset_v2.csv')
results_dir = Path('results')

df['month'] = pd.to_datetime(df['date']).dt.month
df['month_name'] = df['month'].map({1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                                    7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'})

feature_cols = [
    'rsi', 'atr_pct', 'atr_percentile', 'volume_ratio', 'volume_trend',
    'candle_efficiency', 'momentum_4', 'momentum_12', 'compression',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend',
    'pattern_weight', 'session_score'
]

# Добавляем dummy для паттернов
pattern_dummies = pd.get_dummies(df['pattern_name'], prefix='pat')
pattern_cols = [c for c in pattern_dummies.columns if c != 'pat_none']
X_full = pd.concat([df[feature_cols], pattern_dummies[pattern_cols]], axis=1).fillna(0)
y = df['target']

print('=' * 80)
print('ИССЛЕДОВАНИЕ #8: FEATURE IMPORTANCE DRIFT')
print('Как меняется важность признаков по месяцам')
print('=' * 80)

months = sorted(df['month'].unique())
feature_names = list(X_full.columns)
drift_matrix = {}

for m in months:
    mask = df['month'] == m
    if mask.sum() < 100:
        continue

    X_m = X_full[mask]
    y_m = y[mask]

    # Разделяем на train/val в пределах месяца
    n = len(X_m)
    split = int(n * 0.7)
    X_train, X_val = X_m.iloc[:split], X_m.iloc[split:]
    y_train, y_val = y_m.iloc[:split], y_m.iloc[split:]

    if len(X_val) < 30:
        continue

    model = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)

    auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    importance = dict(zip(feature_names, model.feature_importances_))

    month_name = df[df['month'] == m]['month_name'].iloc[0]
    drift_matrix[month_name] = {
        'auc': round(auc, 3),
        'n': len(X_m),
        'importance': {k: round(v, 4) for k, v in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]}
    }

    print(f'\n{month_name} (n={len(X_m)}, AUC={auc:.3f}):')
    for feat, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f'  {feat:<25s}: {imp:.4f}')

# Сравнение важности между месяцами
print(f'\n{"=" * 80}')
print('ДРЕЙФ ВАЖНОСТИ (топ-5 фичей)')
print('=' * 80)

# Заголовок
month_names = [m for m in drift_matrix.keys()]
header = '|'.join(f'{m:^12s}' for m in month_names)
print(f'{"Feature":<25s} | {header}')
print('-' * (25 + 14 * len(month_names)))

# Топ фичи
top_features = set()
for m in drift_matrix:
    for f in drift_matrix[m]['importance']:
        top_features.add(f)
top_features = sorted(top_features, key=lambda f: sum(drift_matrix[m]['importance'].get(f, 0) for m in drift_matrix),
                      reverse=True)[:10]

for feat in top_features:
    values = []
    for m in month_names:
        val = drift_matrix[m]['importance'].get(feat, 0)
        values.append(f'{val:>10.4f}')
    print(f'{feat:<25s}  ' + '  '.join(values))

# Стабильность фичей
print(f'\n{"=" * 80}')
print('СТАБИЛЬНОСТЬ ФИЧЕЙ (по AUC)')
print('=' * 80)

auc_values = [drift_matrix[m]['auc'] for m in month_names]
print(f'Месяцы: {month_names}')
print(f'AUC:    {[drift_matrix[m]["auc"] for m in month_names]}')
if len(auc_values) >= 2:
    print(f'Средний AUC: {np.mean(auc_values):.3f} ± {np.std(auc_values):.3f}')

with open(results_dir / '08_feature_drift.json', 'w', encoding='utf-8') as f:
    json.dump(drift_matrix, f, indent=2)
print(f'\n✅ Сохранено: results/08_feature_drift.json')