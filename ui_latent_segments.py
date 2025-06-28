import numpy as np

def smooth_signal(signal, window_size=5):
    return np.convolve(signal, np.ones(window_size) / window_size, mode='same')

def find_nonzero_segments(signal, sr, threshold=0.03, merge_threshold=2.0):
    """
    Возвращает интервалы (в секундах), где сигнал выше порога, объединяя близкие.
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

    return [(s / sr, e / sr) for s, e in merged]
