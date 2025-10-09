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
# 🔊 Вычитание цикличного шума (канал 2)
# =========================
def subtract_periodic_noise_from_channel(
    chan: np.ndarray,
    sr: int,
    markers: list,
    learn_seconds: float = 3.0,
    pre_margin: float = 0.5,
    fmin_hz: float = 40.0,
    fmax_hz: float = 500.0,
    adapt_window_periods: int = 12
):
    """
    Оценка и вычитание периодического (цикличного) шума.
    1) Берём участок ДО первой метки (на втором канале): [m0 - pre_margin - learn_seconds, m0 - pre_margin]
    2) По автокорреляции оцениваем период T (в сэмплах), ограничивая частоты [fmin_hz, fmax_hz].
    3) Строим средний шаблон одного периода и вычитаем его из всего канала (с адаптивной нормировкой по окнам).

    Возвращает (chan_clean, info_dict).
    """
    n = len(chan)
    if n == 0 or len(markers) == 0:
        return chan.copy(), {"status": "no_data"}

    m0 = float(markers[0][0])
    end_t = max(0.0, m0 - pre_margin)
    start_t = max(0.0, end_t - learn_seconds)
    s = int(start_t * sr)
    e = int(end_t * sr)
    if e <= s + 10:
        return chan.copy(), {"status": "no_learn_segment"}

    learn = chan[s:e].astype(np.float32)
    learn = learn - np.mean(learn)  # центрируем

    # рамки периодов
    pmin = max(2, int(sr / fmax_hz))
    pmax = max(pmin + 1, int(sr / fmin_hz))

    # автокорреляция (нормированная)
    ac = np.correlate(learn, learn, mode='full')
    ac = ac[len(ac)//2:]  # лаги >= 0
    ac = ac / (np.max(ac) + 1e-9)

    # ищем лаг с максимумом в [pmin, pmax)
    search = ac[pmin:min(len(ac), pmax)]
    if len(search) == 0 or np.all(np.isnan(search)):
        return chan.copy(), {"status": "no_peak"}

    lag = int(np.argmax(search)) + pmin
    period = max(2, lag)

    # строим средний шаблон одного периода
    n_cycles = (len(learn) // period)
    if n_cycles < 2:
        return chan.copy(), {"status": "few_cycles"}

    trimmed = learn[:n_cycles * period]
    template = trimmed.reshape(n_cycles, period).mean(axis=0)
    template = template - np.mean(template)

    # подготовим тайлы на всю длину сигнала
    reps = (n + period - 1) // period
    tiled = np.tile(template, reps)[:n]

    # Адаптивная нормировка по окнам (чтобы не «пересубтракнуть»)
    win_len = max(period * adapt_window_periods, period * 6)
    hop = win_len // 3
    chan = chan.astype(np.float32)
    out = np.zeros_like(chan)
    wsum = np.zeros_like(chan)

    # окно Ханна для мягкой склейки
    hann = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(win_len) / max(1, (win_len - 1)))).astype(np.float32)

    for start in range(0, n, hop):
        end = min(n, start + win_len)
        seg = chan[start:end]
        tmp = tiled[start:end]
        if len(seg) < 8:
            break
        # alpha = <seg,tmp> / <tmp,tmp> (LS)
        denom = float(np.dot(tmp, tmp)) + 1e-9
        alpha = float(np.dot(seg, tmp) / denom)
        clean = seg - alpha * tmp
        # окно для склейки
        w = hann[:len(clean)]
        out[start:end] += clean * w
        wsum[start:end] += w

    nz = (wsum > 1e-9)
    out[nz] /= wsum[nz]
    out[~nz] = chan[~nz]

    info = {
        "status": "ok",
        "period_samples": period,
        "period_hz": float(sr) / float(period),
        "learn_segment": (start_t, end_t),
    }
    return out.astype(np.float32, copy=False), info


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
    clean_ch2_periodic_noise: bool = True
):
    """
    Основная логика для 2-х участников:
      - Канал 1 (левый): вопросы — окно вокруг каждой метки [mi-buffer_sec, mi+buffer_sec]
      - Канал 2 (правый): ответы — ОКНО ФИКС. ДЛИНЫ 1.5 c после конца вопроса:
            start_a = end_q
            end_a   = start_a + 1.5
        Порог для 2-го канала повышенный: quantile_ch2 = max(0.99, quantile).
        Для последней метки — логика та же (окно после её вопроса).

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

    # Левый — по модулю + сглаживание
    sm_left = smooth_signal(np.abs(left), smooth_window)

    # Правый: опционально очищаем цикличный шум, затем берём модуль и сглаживаем
    if clean_ch2_periodic_noise and len(markers) > 0:
        right_clean, _ = subtract_periodic_noise_from_channel(
            right_raw, sr, markers,
            learn_seconds=3.0,
            pre_margin=0.6,
            fmin_hz=40.0,
            fmax_hz=500.0,
            adapt_window_periods=12
        )
        sm_right = smooth_signal(np.abs(right_clean), window_size=max(3, smooth_window))
    else:
        sm_right = smooth_signal(np.abs(right_raw), smooth_window)

    dur_sec_left = len(left) / float(sr)
    dur_sec_right = len(right_raw) / float(sr)

    q_segments = []
    a_segments = []

    # Повышенный порог для канала 2
    quantile_ch2 = max(0.99, float(quantile))

    for i in range(len(markers)):
        m1 = float(markers[i][0])

        # --- ВОПРОС (канал 1): [mi - buffer_sec, mi + buffer_sec] ---
        start_q = max(0.0, m1 - buffer_sec)
        end_q   = min(m1 + buffer_sec, dur_sec_left)
        seg_q = find_segments_in_window(
            sm_left, sr, start_q, end_q,
            initial_q=float(quantile),
            min_duration=0.5
        )
        # гарантированно один
        q_s, q_e = seg_q[0]
        q_segments.append((q_s, q_e))

        # --- ОТВЕТ (канал 2): окно фиксированной длины 1.5 c ПОСЛЕ конца вопроса ---
        start_a = max(0.0, float(q_e))
        end_a   = min(dur_sec_right, start_a + 1.5)

        seg_a = find_segments_in_window(
            sm_right, sr, start_a, end_a,
            initial_q=quantile_ch2,   # повышенный порог
            min_duration=0.4
        )
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
