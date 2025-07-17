import numpy as np
import librosa

def change_audio_speed(y, sr, speed_factor=1.0):
    if speed_factor == 1.0:
        return y
    return librosa.resample(y, orig_sr=sr, target_sr=int(sr * speed_factor))

def smooth_signal(signal, window_size=5):
    """
    Сглаживает сигнал с помощью скользящего среднего.
    """
    if window_size < 1:
        return signal
    return np.convolve(signal, np.ones(window_size) / window_size, mode='same')

def compute_threshold(signal, quantile=0.96):
    """
    Возвращает значение, выше которого находится (1 - quantile)*100% сигнала.
    То есть, оставляет только самые мощные пики.
    """
    nonzero = signal[signal > 0]
    if len(nonzero) == 0:
        return 0
    sorted_signal = np.sort(nonzero)
    index = int(len(sorted_signal) * quantile)
    index = min(index, len(sorted_signal) - 1)
    return sorted_signal[index]

def find_nonzero_segments(signal, sr, threshold=0.03, merge_threshold=1.0, mode="свободный"):
    """
    Находит непрерывные интервалы, где сигнал выше порога.
    Объединяет интервалы, расположенные ближе merge_threshold (в секундах).
    
    mode: "свободный" или "5:6"
    - В режиме "5:6" обязательно должно быть найдено ровно 30 интервалов.
    """
    segments = []
    start = None
    for i, val in enumerate(signal):
        if val > threshold and start is None:
            start = i
        elif val <= threshold and start is not None:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, len(signal)))

    # Объединение близких сегментов
    merged = []
    for start, end in segments:
        if not merged:
            merged.append((start, end))
        else:
            last_start, last_end = merged[-1]
            if (start / sr) <= (last_end / sr + merge_threshold):
                merged[-1] = (last_start, end)
            else:
                merged.append((start, end))

    # Перевод в секунды
    final_segments = [(s / sr, e / sr) for s, e in merged]

    # Проверка на количество интервалов в эксперименте
    if mode == "5:6" and len(final_segments) != 30:
        raise ValueError(f"Ожидалось 30 интервалов в режиме '5:6', но найдено: {len(final_segments)}")

    return final_segments
