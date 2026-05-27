
"""Technical Analysis Figure Engine — структурные фигуры (не свечные)."""
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from scipy.signal import argrelextrema
from scipy.stats import linregress


def find_peaks_valleys(series: np.ndarray, order: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Находит пики и впадины в серии."""
    peaks = argrelextrema(series, np.greater, order=order)[0]
    valleys = argrelextrema(series, np.less, order=order)[0]
    return peaks, valleys


def detect_double_top_bottom(df: pd.DataFrame, order: int = 5, tolerance: float = 0.02) -> Optional[Dict]:
    """
    Двойная вершина / Двойное дно.
    
    Критерии:
    - Минимум 10 свечей между экстремумами
    - Два пика/впадины на одном уровне (±tolerance%)
    - Пробой шеи (линии поддержки/сопротивления)
    """
    if len(df) < 30:
        return None
    
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    
    # Ищем пики и впадины за последние 30-50 свечей
    window = min(50, len(df))
    recent_high = high[-window:]
    recent_low = low[-window:]
    recent_close = close[-window:]
    
    peaks, valleys = find_peaks_valleys(recent_high, order=order)
    peaks_v, valleys_v = find_peaks_valleys(recent_low, order=order)
    
    # Объединяем пики из high и close
    all_peaks = sorted(set(peaks) | set(peaks_v))
    all_valleys = sorted(set(valleys) | set(valleys_v))
    
    # --- Двойное дно ---
    if len(all_valleys) >= 2:
        for i in range(len(all_valleys)-1):
            v1_idx, v2_idx = all_valleys[i], all_valleys[i+1]
            if v2_idx - v1_idx < 10:  # минимум 10 свечей
                continue
            
            v1_price = recent_low[v1_idx]
            v2_price = recent_low[v2_idx]
            
            # Проверяем что впадины на одном уровне
            diff = abs(v1_price - v2_price) / max(v1_price, v2_price)
            if diff > tolerance:
                continue
            
            # Ищем пик между ними (шея)
            between = recent_high[v1_idx:v2_idx]
            if len(between) == 0:
                continue
            neck = between.max()
            
            # Проверяем пробой шеи
            last_close = recent_close[-1]
            if last_close > neck:
                target = neck + (neck - v2_price)  # высота фигуры
                return {
                    "pattern": "Двойное дно",
                    "pattern_en": "double_bottom",
                    "direction": "bull",
                    "weight": 0.75,
                    "neck": round(neck, 6),
                    "target": round(target, 6),
                    "valley1": round(v1_price, 6),
                    "valley2": round(v2_price, 6),
                }
    
    # --- Двойная вершина ---
    if len(all_peaks) >= 2:
        for i in range(len(all_peaks)-1):
            p1_idx, p2_idx = all_peaks[i], all_peaks[i+1]
            if p2_idx - p1_idx < 10:
                continue
            
            p1_price = recent_high[p1_idx]
            p2_price = recent_high[p2_idx]
            
            diff = abs(p1_price - p2_price) / max(p1_price, p2_price)
            if diff > tolerance:
                continue
            
            between = recent_low[p1_idx:p2_idx]
            if len(between) == 0:
                continue
            neck = between.min()
            
            last_close = recent_close[-1]
            if last_close < neck:
                target = neck - (p2_price - neck)
                return {
                    "pattern": "Двойная вершина",
                    "pattern_en": "double_top",
                    "direction": "bear",
                    "weight": 0.75,
                    "neck": round(neck, 6),
                    "target": round(target, 6),
                    "peak1": round(p1_price, 6),
                    "peak2": round(p2_price, 6),
                }
    
    return None


def detect_triangle(df: pd.DataFrame) -> Optional[Dict]:
    """
    Треугольник: восходящий, нисходящий, симметричный.
    
    Критерии:
    - Минимум 3 касания каждой стороны
    - Сужение диапазона
    - Наклон верхней и нижней границ
    """
    if len(df) < 30:
        return None
    
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    x = np.arange(len(close))
    
    window = min(40, len(df))
    recent_high = high[-window:]
    recent_low = low[-window:]
    recent_x = x[-window:]
    
    peaks, _ = find_peaks_valleys(recent_high, order=3)
    _, valleys = find_peaks_valleys(recent_low, order=3)
    
    if len(peaks) < 3 or len(valleys) < 3:
        return None
    
    # Регрессия по пикам (верхняя граница)
    peak_x = recent_x[peaks[-4:]]
    peak_y = recent_high[peaks[-4:]]
    if len(peak_x) >= 3:
        upper_slope, upper_int, _, _, _ = linregress(peak_x, peak_y)
    else:
        return None
    
    # Регрессия по впадинам (нижняя граница)
    val_x = recent_x[valleys[-4:]]
    val_y = recent_low[valleys[-4:]]
    if len(val_x) >= 3:
        lower_slope, lower_int, _, _, _ = linregress(val_x, val_y)
    else:
        return None
    
    # Нормализуем наклоны
    mid_price = recent_high[-1]
    upper_slope_pct = upper_slope / mid_price * 100
    lower_slope_pct = lower_slope / mid_price * 100
    
    # Сужение диапазона
    range_start = recent_high[-window] - recent_low[-window]
    range_end = recent_high[-1] - recent_low[-1]
    if range_start <= 0:
        return None
    compression = (range_start - range_end) / range_start
    
    if compression < 0.15:  # нет сужения
        return None
    
    # Классификация треугольника
    if upper_slope_pct < -0.02 and lower_slope_pct > 0.02:
        pattern = "Симметричный треугольник"
        pattern_en = "symm_triangle"
        direction = "neutral"
        weight = 0.5
    elif abs(upper_slope_pct) < 0.02 and lower_slope_pct > 0.02:
        pattern = "Восходящий треугольник"
        pattern_en = "asc_triangle"
        direction = "bull"
        weight = 0.65
    elif upper_slope_pct < -0.02 and abs(lower_slope_pct) < 0.02:
        pattern = "Нисходящий треугольник"
        pattern_en = "desc_triangle"
        direction = "bear"
        weight = 0.65
    else:
        return None
    
    return {
        "pattern": pattern,
        "pattern_en": pattern_en,
        "direction": direction,
        "weight": round(weight * (1 + compression), 2),
        "compression": round(compression * 100, 1),
    }


def detect_flag_pennant(df: pd.DataFrame) -> Optional[Dict]:
    """
    Флаг / Вымпел через структуру.
    
    Критерии:
    1. Сильный импульс (>4%) за 5-15 свечей
    2. Консолидация 5-15 свечей в узком диапазоне (<5%)
    3. Наклон консолидации против импульса (флаг) или горизонтальный (вымпел)
    """
    if len(df) < 30:
        return None
    
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    volume = df['Volume'].values
    
    # Ищем импульс
    best_impulse = None
    for start_offset in [15, 12, 10, 8]:
        if len(close) < start_offset + 8:
            continue
        imp_start = close[-(start_offset+8)]
        imp_end = close[-start_offset]
        pct = (imp_end - imp_start) / imp_start * 100
        
        if abs(pct) >= 4:
            direction = "bull" if pct > 0 else "bear"
            best_impulse = {
                "pct": abs(pct),
                "direction": direction,
                "start_idx": len(close) - start_offset,
                "end_idx": len(close) - 8,
            }
            break
    
    if not best_impulse:
        return None
    
    # Консолидация после импульса
    consol_high = high[-8:-1]
    consol_low = low[-8:-1]
    consol_vol = volume[-8:-1]
    
    range_pct = (consol_high.max() - consol_low.min()) / consol_low.min() * 100
    if range_pct > 6:
        return None
    
    # Объём должен падать в консолидации
    vol_start = consol_vol[:4].mean()
    vol_end = consol_vol[-4:].mean()
    vol_declining = vol_end < vol_start
    
    # Наклон консолидации
    x = np.arange(len(consol_high))
    slope, _ = np.polyfit(x, close[-8:-1], 1)
    slope_pct = slope / close[-2] * 100
    
    # Определяем тип
    if abs(slope_pct) < 0.03:
        pattern_type = "Вымпел"
        pattern_en = "pennant"
    elif (best_impulse["direction"] == "bull" and slope_pct < 0) or          (best_impulse["direction"] == "bear" and slope_pct > 0):
        pattern_type = "Флаг"
        pattern_en = "flag"
    else:
        return None
    
    return {
        "pattern": pattern_type,
        "pattern_en": pattern_en,
        "direction": best_impulse["direction"],
        "weight": 0.65 if pattern_en == "flag" else 0.55,
        "impulse_pct": round(best_impulse["pct"], 1),
        "compression_pct": round(range_pct, 1),
        "vol_declining": vol_declining,
    }


def detect_wedge(df: pd.DataFrame) -> Optional[Dict]:
    """
    Клин: восходящий (медвежий) / нисходящий (бычий).
    
    Критерии:
    - Обе границы наклонены в одну сторону
    - Сужение диапазона
    - Минимум 3 касания каждой стороны
    """
    if len(df) < 25:
        return None
    
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    x = np.arange(len(close))
    
    window = min(35, len(df))
    recent_high = high[-window:]
    recent_low = low[-window:]
    recent_x = x[-window:]
    
    peaks, _ = find_peaks_valleys(recent_high, order=3)
    _, valleys = find_peaks_valleys(recent_low, order=3)
    
    if len(peaks) < 3 or len(valleys) < 3:
        return None
    
    peak_x = recent_x[peaks[-3:]]
    peak_y = recent_high[peaks[-3:]]
    val_x = recent_x[valleys[-3:]]
    val_y = recent_low[valleys[-3:]]
    
    if len(peak_x) < 2 or len(val_x) < 2:
        return None
    
    upper_slope, _, _, _, _ = linregress(peak_x, peak_y)
    lower_slope, _, _, _, _ = linregress(val_x, val_y)
    
    mid = close[-1]
    us = upper_slope / mid * 100
    ls = lower_slope / mid * 100
    
    # Проверяем сужение
    range_start = recent_high[0] - recent_low[0]
    range_end = recent_high[-1] - recent_low[-1]
    if range_start <= 0:
        return None
    compression = (range_start - range_end) / range_start
    
    if compression < 0.1:
        return None
    
    # Классификация
    if us < -0.02 and ls < -0.02:
        # Обе границы вниз = нисходящий клин (бычий)
        return {
            "pattern": "Нисходящий клин (бычий)",
            "pattern_en": "desc_wedge_bull",
            "direction": "bull",
            "weight": 0.6,
            "compression": round(compression * 100, 1),
        }
    elif us > 0.02 and ls > 0.02:
        # Обе границы вверх = восходящий клин (медвежий)
        return {
            "pattern": "Восходящий клин (медвежий)",
            "pattern_en": "asc_wedge_bear",
            "direction": "bear",
            "weight": 0.6,
            "compression": round(compression * 100, 1),
        }
    
    return None


def detect_head_shoulders(df: pd.DataFrame) -> Optional[Dict]:
    """
    Голова и плечи / Перевёрнутая голова и плечи.
    
    Критерии:
    - Три пика/впадины
    - Средний экстремум выше/ниже крайних
    - Пробой шеи
    """
    if len(df) < 40:
        return None
    
    high = df['High'].values[-40:]
    low = df['Low'].values[-40:]
    close = df['Close'].values[-40:]
    
    peaks, valleys = find_peaks_valleys(high, order=4)
    _, valleys_low = find_peaks_valleys(low, order=4)
    
    # Голова и плечи (медвежий разворот)
    if len(peaks) >= 3:
        for i in range(len(peaks)-2):
            p1, p2, p3 = peaks[i], peaks[i+1], peaks[i+2]
            if p3 - p1 < 15:
                continue
            
            h1, h2, h3 = high[p1], high[p2], high[p3]
            # Голова выше плеч
            if h2 > h1 and h2 > h3:
                # Ищем шею между p1 и p2
                between = low[p1:p2]
                if len(between) == 0:
                    continue
                neck = between.min()
                
                if close[-1] < neck:
                    target = neck - (h2 - neck)
                    return {
                        "pattern": "Голова и плечи",
                        "pattern_en": "head_shoulders",
                        "direction": "bear",
                        "weight": 0.85,
                        "neck": round(neck, 6),
                        "target": round(target, 6),
                    }
    
    # Перевёрнутая голова и плечи (бычий разворот)
    all_valleys = sorted(set(valleys) | set(valleys_low))
    if len(all_valleys) >= 3:
        for i in range(len(all_valleys)-2):
            v1, v2, v3 = all_valleys[i], all_valleys[i+1], all_valleys[i+2]
            if v3 - v1 < 15:
                continue
            
            l1, l2, l3 = low[v1], low[v2], low[v3]
            if l2 < l1 and l2 < l3:
                between = high[v1:v2]
                if len(between) == 0:
                    continue
                neck = between.max()
                
                if close[-1] > neck:
                    target = neck + (neck - l2)
                    return {
                        "pattern": "Перевёрнутая голова и плечи",
                        "pattern_en": "inv_head_shoulders",
                        "direction": "bull",
                        "weight": 0.85,
                        "neck": round(neck, 6),
                        "target": round(target, 6),
                    }
    
    return None


def analyze_all_figures(df: pd.DataFrame) -> Optional[Dict]:
    """
    Анализирует все структурные фигуры.
    Возвращает лучшую найденную.
    """
    detectors = [
        detect_head_shoulders,
        detect_double_top_bottom,
        detect_triangle,
        detect_wedge,
        detect_flag_pennant,
    ]
    
    best = None
    for detector in detectors:
        try:
            result = detector(df)
            if result and (best is None or result["weight"] > best["weight"]):
                best = result
        except Exception:
            pass
    
    return best



# Test removed - run via: python3 -c 'from pattern_engine import analyze_all_figures; ...'
