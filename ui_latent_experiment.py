import numpy as np
import matplotlib.pyplot as plt

# 🧹 Сглаживание сигнала
def smooth_signal(signal, window_size=5):
    if signal.ndim == 1:
        return np.convolve(signal, np.ones(window_size) / window_size, mode='same')
    elif signal.ndim == 2:
        return np.array([np.convolve(channel, np.ones(window_size) / window_size, mode='same') for channel in signal])
    else:
        raise ValueError("Ожидается 1D или 2D сигнал для сглаживания")

# 📉 Вычисление порога
def compute_threshold(signal, quantile=0.97):
    nonzero = signal[signal > 0]
    if len(nonzero) == 0:
        return 0.01
    threshold = np.percentile(nonzero, quantile * 100)
    return max(threshold, 0.01)

# 🔍 Поиск одного сегмента в окне с адаптацией порогов
def find_segments_in_window(signal, sr, start_time, end_time, initial_q=0.97, min_duration=0.5):
    start_idx = int(start_time * sr)
    end_idx = int(end_time * sr)
    region = signal[start_idx:end_idx]
    if len(region) == 0:
        return []

    max_q_retries = 5
    max_m_retries = 5

    for q_step in range(max_q_retries):
        quantile = initial_q - 0.03 * q_step
        threshold = compute_threshold(region, quantile)

        for m_step in range(max_m_retries):
            merge_thr = 0.5 + 0.3 * m_step
            segments = []
            start = None

            for i, val in enumerate(region):
                if val > threshold and start is None:
                    start = i
                elif val <= threshold and start is not None:
                    segments.append((start_idx + start, start_idx + i))
                    start = None
            if start is not None:
                segments.append((start_idx + start, start_idx + len(region)))

            # Объединение
            merged = []
            for s, e in segments:
                if not merged:
                    merged.append((s, e))
                else:
                    last_s, last_e = merged[-1]
                    if s / sr <= last_e / sr + merge_thr:
                        merged[-1] = (last_s, e)
                    else:
                        merged.append((s, e))

            # Удаляем короткие
            merged = [(s, e) for s, e in merged if (e - s) / sr >= min_duration]

            if len(merged) == 1:
                return [(merged[0][0] / sr, merged[0][1] / sr)]

    return []

# 🧠 Главная функция
def find_nonzero_segments_stereo(signal_stereo, sr, markers, quantile=0.97, smooth_window=5):
    if signal_stereo.ndim != 2 or signal_stereo.shape[0] != 2:
        raise ValueError("Ожидается 2 канала: левый и правый.")

    # 🧠 Обеспечиваем C-contiguous массивы
    signal_stereo = np.ascontiguousarray(signal_stereo)

    left = signal_stereo[0]
    right = signal_stereo[1]
    smoothed_left = smooth_signal(np.abs(left), smooth_window)
    smoothed_right = smooth_signal(np.abs(right), smooth_window)

    buffer = 3.5
    q_segments = []
    a_segments = []

    for i in range(len(markers)):
        m1 = markers[i][0]

        # 🔹 Вопрос: вокруг метки в левом канале
        start_q = max(0, m1 - buffer)
        end_q = min(m1 + buffer, len(left) / sr)
        seg_q = find_segments_in_window(smoothed_left, sr, start_q, end_q, quantile, min_duration=0.5)
        if seg_q:
            q_segments.append(seg_q[0])
        else:
            print(f"❌ [Q{i+1}] не найден")

        # 🔸 Ответ: между метками в правом канале
        if i < len(markers) - 1:
            m2 = markers[i + 1][0]
            start_a = m1 + buffer
            end_a = m2 - buffer
        else:
            start_a = m1 + buffer
            end_a = len(right) / sr

        if end_a > start_a:
            seg_a = find_segments_in_window(smoothed_right, sr, start_a, end_a, quantile, min_duration=0.5)
            if seg_a:
                a_segments.append(seg_a[0])
            else:
                print(f"❌ [A{i+1}] не найден")
        else:
            print(f"⚠️ Пропущен интервал ответа между {i+1} и {i+2}")

    if len(q_segments) != 30 or len(a_segments) != 30:
        print(f"⚠️ Вопросов: {len(q_segments)}, ответов: {len(a_segments)}")

    return q_segments, a_segments

# 📈 Отрисовка
def plot_dual_channel_segments(signal_stereo, sr, q_segments, a_segments, markers):
    fig, axs = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    t = np.linspace(0, len(signal_stereo[0]) / sr, len(signal_stereo[0]))

    # Левый канал (вопросы)
    axs[0].plot(t, signal_stereo[0], color='blue', alpha=0.7)
    axs[0].set_title("Канал 1 — Вопросы")
    for start, end in q_segments:
        axs[0].axvspan(start, end, facecolor='green', alpha=0.3)
    for m in markers:
        axs[0].axvline(m[0], color='yellow', linestyle='--', linewidth=0.8)
        axs[0].axvspan(m[0] - 3.5, m[0] + 3.5, facecolor='grey', alpha=0.1, hatch='////')

    # Правый канал (ответы)
    axs[1].plot(t, signal_stereo[1], color='red', alpha=0.7)
    axs[1].set_title("Канал 2 — Ответы")
    for start, end in a_segments:
        axs[1].axvspan(start, end, facecolor='orange', alpha=0.3)
    for m in markers:
        axs[1].axvline(m[0], color='yellow', linestyle='--', linewidth=0.8)

    axs[1].set_xlabel("Время (сек)")
    plt.tight_layout()
    plt.show()
