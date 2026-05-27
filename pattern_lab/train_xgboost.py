"""XGBoost на ML-датасете."""
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

df = pd.read_csv('ml_dataset.csv')

# Фичи
feature_cols = [
    'pattern_found', 'pattern_weight',
    'rsi', 'atr_pct', 'volume_ratio',
    'momentum_4', 'momentum_12', 'compression',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend',
]

# Кодируем pattern_name
pattern_dummies = pd.get_dummies(df['pattern_name'], prefix='pat')
pattern_dummies = pattern_dummies.drop(columns=['pat_none'], errors='ignore')

X = pd.concat([df[feature_cols], pattern_dummies], axis=1)
X = X.fillna(0)
y = df['target']

# Сплит по времени (последние 20% — тест)
n = len(X)
X_train, X_test = X.iloc[:int(n*0.8)], X.iloc[int(n*0.8):]
y_train, y_test = y.iloc[:int(n*0.8)], y.iloc[int(n*0.8):]

# Обучаем
model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

# Оценка
y_pred = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred)
print(f'Test AUC: {auc:.3f}')

# Feature importance
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print('\nТОП-15 ФИЧЕЙ:')
for _, row in importance.head(15).iterrows():
    bar = '#' * int(row['importance'] * 100)
    print(f'  {row["feature"]:<25s}: {row["importance"]:.4f} {bar}')

model.save_model('pattern_xgb_model.json')
print('\nМодель сохранена: pattern_xgb_model.json')