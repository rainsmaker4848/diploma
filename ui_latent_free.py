import numpy as np
import librosa

def change_audio_speed(y, sr, speed_factor=1.0):
    if speed_factor == 1.0:
        return y
    return librosa.resample(y, orig_sr=sr, target_sr=int(sr * speed_factor))

def smooth_signal(signal, window_size=5):
    return np.convolve(signal, np.ones(window_size) / window_size, mode='same')

def compute_threshold(signal, quantile=0.96):
    nonzero = signal[signal > 0]
    if len(nonzero) == 0:
        return 0
    sorted_signal = np.sort(nonzero)
    index = int(len(sorted_signal) * quantile)
    index = min(index, len(sorted_signal) - 1)
    return sorted_signal[index]

def find_nonzero_segments(signal, sr, threshold, merge_threshold):
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
