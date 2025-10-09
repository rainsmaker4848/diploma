import numpy as np

def apply_marker_zeroing_filter(y, sr, markers, buffer=4.0):
    """
    Зануляет участки аудио вне указанных временных маркеров (с учётом буфера).
    :param y: аудиосигнал (np.ndarray)
    :param sr: частота дискретизации
    :param markers: список float — временные точки (например, начала и конца фраз)
    :param buffer: дополнительное время до/после маркеров (в секундах)
    :return: отфильтрованный сигнал
    """
    duration = len(y) / sr
    filtered_elements = [markers[i] for i in range(len(markers)) if i % 2 == 0]
    intervals = []

    if filtered_elements and filtered_elements[0] > buffer:
        intervals.append((0, filtered_elements[0] - buffer))

    for i in range(len(filtered_elements) - 1):
        start = filtered_elements[i] + buffer
        end = filtered_elements[i + 1] - buffer
        if end > start:
            intervals.append((start, end))

    if filtered_elements and filtered_elements[-1] + buffer < duration:
        intervals.append((filtered_elements[-1] + buffer, duration))

    y_filtered = y.copy()
    for start, end in intervals:
        start_idx = int(start * sr)
        end_idx = int(end * sr)
        y_filtered[start_idx:end_idx] = 0

    return y_filtered
