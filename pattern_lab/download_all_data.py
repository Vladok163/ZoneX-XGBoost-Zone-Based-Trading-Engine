# download_all_data.py — Все монеты с Binance Futures
import pandas as pd
import requests
from datetime import datetime, timedelta
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Получаем ВСЕ USDT пары с Binance Futures
print("Загружаем список всех USDT пар...")
r = requests.get("https://fapi.binance.com/fapi/v1/exchangeInfo")
data = r.json()

symbols = [s['symbol'] for s in data['symbols']
           if s['symbol'].endswith('USDT')
           and s['status'] == 'TRADING'
           and s['contractType'] == 'PERPETUAL']

print(f"Найдено {len(symbols)} USDT пар")

TIMEFRAMES = ["15m", "1h", "4h"]
DAYS = 90  # 3 месяца

total = len(symbols) * len(TIMEFRAMES)
done = 0

for sym in symbols:
    for tf in TIMEFRAMES:
        fname = DATA_DIR / f"{sym}_{tf}.csv"
        if fname.exists():
            done += 1
            continue

        try:
            r = requests.get(
                "https://fapi.binance.com/fapi/v1/klines",
                params={
                    "symbol": sym, "interval": tf, "limit": 1000,
                    "startTime": int((datetime.now() - timedelta(days=DAYS)).timestamp() * 1000),
                    "endTime": int(datetime.now().timestamp() * 1000)
                }, timeout=30)

            if r.status_code == 200 and len(r.json()) > 100:
                df = pd.DataFrame(r.json(), columns=[
                    "ts", "Open", "High", "Low", "Close", "Volume",
                    "close_ts", "quote_vol", "trades", "tbb", "tbq", "ignore"
                ])
                df.to_csv(fname, index=False)
                done += 1
                print(f"[{done}/{total}] {sym} {tf}: {len(df)} свечей")
            else:
                done += 1
        except Exception as e:
            done += 1
            print(f"[{done}/{total}] {sym} {tf}: ОШИБКА - {e}")

print(f"\nГотово! Скачано {done} файлов в {DATA_DIR}")