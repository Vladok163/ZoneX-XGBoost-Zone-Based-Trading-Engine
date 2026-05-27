"""Сохраняет ВСЕ найденные фигуры в папки для ручной проверки."""
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from pathlib import Path
from pattern_engine import analyze_all_figures
import shutil

# Очищаем старые результаты
for d in Path("results").iterdir():
    if d.is_dir():
        shutil.rmtree(d)

# Загружаем данные
data_dir = Path("data")
results_dir = Path("results")

all_figures = []

for f in sorted(data_dir.glob("*.csv")):
    symbol, tf = f.stem.split("_")
    df = pd.read_csv(f)
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df = df.set_index('ts')
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = df[c].astype(float)

    window = 60
    step = 30

    for i in range(0, len(df) - window - 5, step):
        chunk = df.iloc[i:i + window]
        result = analyze_all_figures(chunk)
        if result and result.get('weight', 0) >= 0.4:
            pat_name = result['pattern'].replace(' ', '_').replace('(', '').replace(')', '')
            pat_dir = results_dir / pat_name
            pat_dir.mkdir(exist_ok=True)

            # Сохраняем график
            title = f"{symbol} {tf} - {result['pattern']} w={result['weight']:.2f}"
            try:
                fig, axs = mpf.plot(chunk, type='candle', volume=True,
                                    title=title, style='charles', returnfig=True, figsize=(14, 8))
                fname = pat_dir / f"{symbol}_{tf}_{i}.png"
                fig.savefig(fname, dpi=100)
                plt.close(fig)
                all_figures.append({'pattern': result['pattern'], 'file': str(fname)})
            except:
                pass

# Статистика
from collections import Counter

stats = Counter(f['pattern'] for f in all_figures)
print(f"Всего сохранено графиков: {len(all_figures)}")
for pat, n in stats.most_common():
    print(f"  {pat}: {n}")
print(f"\nСмотри папки в results/")