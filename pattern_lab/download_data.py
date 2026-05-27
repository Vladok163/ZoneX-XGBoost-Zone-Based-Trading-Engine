import pandas as pd
import requests
from datetime import datetime, timedelta
from pathlib import Path

SYMBOLS = [
    # Топ-20 по объёму
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
    "MATICUSDT", "UNIUSDT", "ATOMUSDT", "LTCUSDT", "ETCUSDT",
    "OPUSDT", "ARBUSDT", "SUIUSDT", "APTUSDT", "NEARUSDT",
    # Средние
    "FILUSDT", "INJUSDT", "TIAUSDT", "SEIUSDT", "RUNEUSDT",
    "HBARUSDT", "STXUSDT", "IMXUSDT", "GRTUSDT", "ALGOUSDT",
    "FTMUSDT", "SANDUSDT", "MANAUSDT", "AXSUSDT", "THETAUSDT",
    # Ещё
    "FLOWUSDT", "QNTUSDT", "EGLDUSDT", "KAVAUSDT", "CFXUSDT",
    "MASKUSDT", "SNXUSDT", "COMPUSDT", "ZRXUSDT", "BATUSDT",
    "SUSHIUSDT", "CRVUSDT", "1INCHUSDT", "DYDXUSDT", "CAKEUSDT",
    # Новые популярные
    "PENDLEUSDT", "TNSRUSDT", "ENAUSDT", "ETHFIUSDT", "REZUSDT",
    "OMNIUSDT", "TAOUSDT", "WUSDT", "ZROUSDT", "ZKUSDT",
    # Мемкоины
    "PEPEUSDT", "WIFUSDT", "BONKUSDT", "FLOKIUSDT", "SHIBUSDT",
    # Ещё
    "COREUSDT", "BEAMUSDT", "BLASTUSDT", "XAIUSDT", "PRIMEUSDT",
    "ACEUSDT", "NFPUSDT", "AIUSDT", "XAIUSDT", "PORTALUSDT",
    "PIXELUSDT", "STRKUSDT", "MAVIAUSDT", "ZETAUSDT", "DYMUSDT",
    "MANTAUSDT", "ALTUSDT", "JUPUSDT", "PYTHUSDT", "JTOUSDT",
]
TIMEFRAMES = ["1h",'15m', "4h"]
DAYS = 90
DATA_DIR = Path("data")

for sym in SYMBOLS:
    for tf in TIMEFRAMES:
        print(f"Скачиваю {sym} {tf}...")
        r = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={
                "symbol": sym, "interval": tf, "limit": 1000,
                "startTime": int((datetime.now() - timedelta(days=DAYS)).timestamp() * 1000),
                "endTime": int(datetime.now().timestamp() * 1000)
            }, timeout=30)
        if r.status_code == 200:
            df = pd.DataFrame(r.json(), columns=[
                "ts","Open","High","Low","Close","Volume",
                "close_ts","quote_vol","trades","tbb","tbq","ignore"
            ])
            df.to_csv(DATA_DIR / f"{sym}_{tf}.csv", index=False)
            print(f"  OK: {len(df)} свечей")
        else:
            print(f"  Ошибка: {r.status_code}")

print("\nГотово!")