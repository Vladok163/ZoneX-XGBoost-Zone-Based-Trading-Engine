"""Визуализация фигур на графике."""
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from pathlib import Path
from pattern_engine import analyze_all_figures

# Загружаем данные
df = pd.read_csv("data/BTCUSDT_1h.csv")
df['ts'] = pd.to_datetime(df['ts'], unit='ms')
df = df.set_index('ts')
for c in ["Open","High","Low","Close","Volume"]:
    df[c] = df[c].astype(float)

# Ищем фигуры во всех окнах по 60 свечей
window = 60
step = 20
found = []

for i in range(0, len(df) - window, step):
    chunk = df.iloc[i:i+window]
    result = analyze_all_figures(chunk)
    if result and result.get("weight", 0) >= 0.5:
        found.append({
            "start": chunk.index[0],
            "end": chunk.index[-1],
            "pattern": result["pattern"],
            "direction": result["direction"],
            "weight": result["weight"],
            "chunk": chunk
        })

print(f"Найдено фигур: {len(found)}")

# Рисуем первые 5
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

for i, f in enumerate(found[:5]):
    title = f"{f['pattern']} ({f['direction']}) w={f['weight']:.2f}"
    fig, axs = mpf.plot(
        f['chunk'], type='candle', volume=True,
        title=title,
        style='charles',
        returnfig=True,
        figsize=(14, 8)
    )
    fig.savefig(results_dir / f"figure_{i+1}_{f['pattern']}.png", dpi=150)
    plt.close(fig)
    print(f"  Сохранён: figure_{i+1}_{f['pattern']}.png")

print("Готово! Смотри папку results/")