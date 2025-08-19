# ui_latent_experiment.py
import os
import numpy as np
import matplotlib.pyplot as plt

# для экспорта фрагментов
try:
    import soundfile as sf
except Exception:
    sf = None  # чтобы модуль не падал, если soundfile не установлен


# =========================
# 🧹 Сглаживание (память-бережно)
# =========================
def smooth_signal(signal: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Сглаживание скользящим средним.
    Поддерживает 1D (моно) и 2D (каналы, N).
    Возвращает C-contiguous массив.
    """
    if window_size < 1:
        return np.ascontiguousarray(signal)

    kernel = np.ones(int(window_size), dtype=np.float32) / float(window_size)

    if signal.ndim == 1:
        out = np.convolve(signal, kernel, mode='same')
        return np.ascontiguousarray(out)
    elif signal.ndim == 2:
        # сглаживаем КАЖДЫЙ канал отдельно
        smoothed = [np.convolve(ch, kernel, mode='same') for ch in signal]
        return np.ascontiguousarray(np.vstack(smoothed))
    else:
        raise ValueError("Ожидается 1D или 2D сигнал для сглаживания")


# =========================
# 📉 Порог по квантилю
# =========================
def compute_threshold(signal_1d: np.ndarray, quantile: float = 0.97) -> float:
    """
    Порог = квантиль по положительным значениям (полу-волновая детекция).
    На вход только 1D массив (окно/канал).
    """
    if signal_1d.ndim != 1:
        raise ValueError("compute_threshold ожидает 1D массив.")
    nonzero = signal_1d[signal_1d > 0]
    if nonzero.size == 0:
        return 0.01
    thr = np.percentile(nonzero, quantile * 100.0)
    return max(float(thr), 0.01)


# =========================
# 🔎 Фолбэк: взять самый «энергичный» кусок в окне
# =========================
def _fallback_strongest_segment(region_abs: np.ndarray, sr: int, min_duration: float):
    """
    region_abs — модуль (или энергия) 1D участка окна.
    Находит пик энергии и расширяет границы до падения энергии.
    Возвращает (s_idx_local, e_idx_local) в индексах ВНУТРИ region_abs.
    Если совсем пусто — возвращает минимальный отрезок длиной min_duration вокруг максимума.
    """
    n = len(region_abs)
    if n == 0:
        return None

    # сглаженная энергия ~ 50 мс
    win = max(3, int(0.05 * sr))
    ker = np.ones(win, dtype=np.float32) / float(win)
    sm = np.convolve(region_abs.astype(np.float32), ker, mode='same')

    peak = int(np.argmax(sm))
    peak_val = sm[peak]
    if peak_val <= 0:
        return None

    # пороги расширения
    left = peak
    right = peak
    thr_down = 0.3 * peak_val

    # расширяем влево
    while left > 0 and sm[left] > thr_down:
        left -= 1
    # расширяем вправо
    while right < n - 1 and sm[right] > thr_down:
        right += 1

    # гарантируем длительность
    min_samp = max(1, int(min_duration * sr))
    if (right - left + 1) < min_samp:
        deficit = min_samp - (right - left + 1)
        add_left = deficit // 2
        add_right = deficit - add_left
        left = max(0, left - add_left)
        right = min(n - 1, right + add_right)

    return (left, right + 1)  # end exclusive


# =========================
# 🔍 Поиск одного сегмента в окне
# =========================
def find_segments_in_window(
    signal_1d: np.ndarray,
    sr: int,
    start_time: float,
    end_time: float,
    initial_q: float = 0.97,
    min_duration: float = 0.5
):
    """
    Ищет РОВНО ОДИН устойчивый энергетический сегмент в окне [start_time, end_time].
    Адаптирует квантиль (порог) и порог слияния.
    Если ничего не найдено — включает фолбэк по максимальной энергии.
    Возвращает список из одного кортежа (t_start, t_end) в секундах.
    """
    if signal_1d.ndim != 1:
        raise ValueError("find_segments_in_window ожидает 1D сигнал.")

    # индексы окна
    s_idx = max(0, int(start_time * sr))
    e_idx = min(len(signal_1d), int(end_time * sr))
    if e_idx <= s_idx:
        # пустое окно — вернём хотя бы минимальный отрезок в начале
        t0 = start_time
        t1 = min(end_time, start_time + min_duration)
        return [(t0, t1)]

    region = signal_1d[s_idx:e_idx]
    if region.size == 0:
        t0 = start_time
        t1 = min(end_time, start_time + min_duration)
        return [(t0, t1)]

    # базовая многошаговая попытка
    max_q_retries = 6
    max_m_retries = 6

    for q_step in range(max_q_retries):
        quantile = max(0.70, initial_q - 0.03 * q_step)  # не падаем ниже 0.70
        thr = compute_threshold(region, quantile)

        for m_step in range(max_m_retries):
            merge_thr_sec = 0.4 + 0.25 * m_step  # постепенно расширяем «сцепление»
            segments = []
            start = None

            # простая пороговая детекция
            for i, val in enumerate(region):
                if val > thr and start is None:
                    start = i
                elif val <= thr and start is not None:
                    segments.append((s_idx + start, s_idx + i))
                    start = None
            if start is not None:
                segments.append((s_idx + start, s_idx + len(region)))

            # слияние близких сегментов
            merged = []
            for s, e in segments:
                if not merged:
                    merged.append((s, e))
                else:
                    last_s, last_e = merged[-1]
                    # если пауза между сегментами меньше merge_thr_sec — сливаем
                    if (s / sr) <= (last_e / sr) + merge_thr_sec:
                        merged[-1] = (last_s, e)
                    else:
                        merged.append((s, e))

            # убираем коротышей
            merged = [(s, e) for s, e in merged if (e - s) / sr >= min_duration]

            if len(merged) == 1:
                s_samp, e_samp = merged[0]
                return [(s_samp / sr, e_samp / sr)]

    # ---- ФОЛБЭК: берём самый «энергичный» участок окна ----
    fb = _fallback_strongest_segment(np.abs(region), sr, min_duration)
    if fb is not None:
        ls, le = fb  # локальные индексы в пределах region
        s_abs = s_idx + ls
        e_abs = s_idx + le
        return [(s_abs / sr, e_abs / sr)]

    # Совсем без энергии — минимальный отрезок с начала окна
    t0 = start_time
    t1 = min(end_time, start_time + min_duration)
    return [(t0, t1)]


# =========================
# 🧠 Главная функция 2ch эксперимента
# =========================
def find_nonzero_segments_stereo(
    signal_stereo: np.ndarray,
    sr: int,
    markers: list,
    quantile: float = 0.99,
    smooth_window: int = 5,
    buffer_sec: float = 3,
    clean_ch2_periodic_noise: bool = True  # оставлено для совместимости; не используется
):
    """
    Основная логика для 2-х участников:
      - Канал 1 (левый): вопросы — окно вокруг каждой метки [mi-buffer_sec, mi+buffer_sec]
      - Канал 2 (правый): ответы — суженное окно:
            от конца окна вокруг метки (mi + buffer_sec)
            до середины промежутка между (mi + buffer_sec) и (m(i+1) - buffer_sec)
        Для последней метки — [mi + buffer_sec, конец правого канала]

    markers: список кортежей (start, end) — берём start (метка_i)
    Возвращает: (q_segments, a_segments) — списки кортежей (t_start, t_end), сек.
    Гарантируется по одному отрезку на каждую метку (при необходимости используется фолбэк).
    """
    if signal_stereo.ndim != 2 or signal_stereo.shape[0] not in (1, 2):
        raise ValueError("Ожидается массив формы (каналы, N) с 1 или 2 каналами.")

    # Приводим к (2, N): если моно — дублируем (для унификации логики)
    if signal_stereo.shape[0] == 1:
        signal_stereo = np.vstack([signal_stereo, signal_stereo])

    signal_stereo = np.ascontiguousarray(signal_stereo)

    # Каналы
    left = signal_stereo[0].astype(np.float32, copy=False)
    right_raw = signal_stereo[1].astype(np.float32, copy=False)

    # Левый/правый — модуль + сглаживание (без «линии уровня»)
    sm_left = smooth_signal(np.abs(left), smooth_window)
    sm_right = smooth_signal(np.abs(right_raw), smooth_window)

    dur_sec_left = len(left) / float(sr)
    dur_sec_right = len(right_raw) / float(sr)

    q_segments = []
    a_segments = []

    for i in range(len(markers)):
        m1 = float(markers[i][0])

        # --- ВОПРОС (канал 1): [mi - buffer_sec, mi + buffer_sec] ---
        start_q = max(0.0, m1 - buffer_sec)
        end_q = min(m1 + buffer_sec, dur_sec_left)
        seg_q = find_segments_in_window(sm_left, sr, start_q, end_q,
                                        initial_q=quantile, min_duration=0.5)
        q_segments.append(seg_q[0])  # гарантированно один

        # --- ОТВЕТ (канал 2): от (mi + buffer_sec) до середины с (m(i+1) - buffer_sec) ---
        if i < len(markers) - 1:
            m2 = float(markers[i + 1][0])
            start_a_edge = m1 + buffer_sec
            end_a_edge = m2 - buffer_sec
            mid = 0.5 * (start_a_edge + end_a_edge)
            start_a = max(0.0, start_a_edge)
            end_a = max(start_a, min(mid, dur_sec_right))
        else:
            start_a = max(0.0, m1 + buffer_sec)
            end_a = dur_sec_right

        seg_a = find_segments_in_window(sm_right, sr, start_a, end_a,
                                        initial_q=quantile, min_duration=0.4)
        a_segments.append(seg_a[0])  # гарантированно один

    return q_segments, a_segments


# =========================
# 📈 Отрисовка (опционально для дебага)
# =========================
def plot_dual_channel_segments(signal_stereo: np.ndarray, sr: int, q_segments, a_segments, markers):
    """
    Визуализация двух каналов + найденных сегментов.
    """
    if signal_stereo.shape[0] == 1:
        signal_stereo = np.vstack([signal_stereo, signal_stereo])

    t = np.arange(signal_stereo.shape[1]) / float(sr)

    fig, axs = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    # Левый канал (вопросы)
    axs[0].plot(t, signal_stereo[0], color='blue', alpha=0.7)
    axs[0].set_title("Канал 1 — Вопросы")
    for start, end in q_segments:
        axs[0].axvspan(start, end, facecolor='green', alpha=0.3)
    for m in markers:
        axs[0].axvline(m[0], color='yellow', linestyle='--', linewidth=0.8)
        s = max(0.0, m[0] - 3.0)
        e = m[0] + 3.0
        axs[0].axvspan(s, e, facecolor='lightgrey', alpha=0.15)

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


# =========================
# 💾 Нарезка WAV по найденным интервалам (вопросы/ответы)
# =========================
def export_segments_to_files(signal_stereo: np.ndarray,
                             sr: int,
                             q_segments: list,
                             a_segments: list,
                             out_dir: str):
    """
    Сохраняет нарезанные куски аудио в out_dir:
      - question_01.wav, ..., question_30.wav
      - answer_01.wav,   ..., answer_30.wav

    Ожидает массив формы (C, N): C=1 или 2. Если моно, дублировать не будем.
    """
    if sf is None:
        raise RuntimeError("Для экспорта нужен пакет 'soundfile' (pip install soundfile).")

    if signal_stereo.ndim == 1:
        sig = signal_stereo[np.newaxis, :]
    elif signal_stereo.ndim == 2 and signal_stereo.shape[0] in (1, 2):
        sig = signal_stereo
    else:
        raise ValueError("Ожидается аудиомассив формы (N,) или (C, N) где C=1 или 2.")

    os.makedirs(out_dir, exist_ok=True)

    def _save(name: str, t0: float, t1: float):
        s = max(0, int(t0 * sr))
        e = min(sig.shape[1], int(t1 * sr))
        if e <= s:
            return
        # (N,) или (N, C) для soundfile
        chunk = sig[:, s:e].T if sig.shape[0] > 1 else sig[0, s:e]
        sf.write(os.path.join(out_dir, name), chunk, sr)

    # Вопросы
    for i, (s, e) in enumerate(q_segments, start=1):
        _save(f"question_{i:02d}.wav", float(s), float(e))

    # Ответы
    for i, (s, e) in enumerate(a_segments, start=1):
        _save(f"answer_{i:02d}.wav", float(s), float(e))
