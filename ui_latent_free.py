import numpy as np

# 🧹 Сглаживание сигнала
def smooth_signal(signal, window_size=5):
    if signal.ndim != 1:
        raise ValueError("Функция smooth_signal ожидает моно сигнал (1D)")
    return np.convolve(signal, np.ones(window_size) / window_size, mode='same')

# 📉 Вычисление порога
def compute_threshold(signal, quantile=0.97):
    nonzero = signal[signal > 0]
    if len(nonzero) == 0:
        return 0.01
    return max(np.percentile(nonzero, quantile * 100), 0.01)

# 🔍 Поиск активных сегментов на всём сигнале
def find_nonzero_segments(smoothed_signal, sr, threshold, merge_threshold=0.5, min_duration=0.5):
    if smoothed_signal.ndim != 1:
        raise ValueError("Функция find_nonzero_segments ожидает 1D моно сигнал")

    segments = []
    start = None

    # 🔍 Находим участки выше порога
    for i, val in enumerate(smoothed_signal):
        if val > threshold and start is None:
            start = i
        elif val <= threshold and start is not None:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, len(smoothed_signal)))

    # 🔗 Объединяем близкие участки
    merged = []
    for s, e in segments:
        if not merged:
            merged.append((s, e))
        else:
            last_s, last_e = merged[-1]
            if s <= last_e + int(merge_threshold * sr):
                merged[-1] = (last_s, e)
            else:
                merged.append((s, e))

    # ⏳ Удаляем слишком короткие
    final_segments = [(s / sr, e / sr) for s, e in merged if (e - s) / sr >= min_duration]

    return final_segments
