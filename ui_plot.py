# ui_plot.py
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ------------------------------
# Вспомогательные утилиты
# ------------------------------
def _ensure_2d_channels_first(y: np.ndarray) -> np.ndarray:
    """
    Приводит звук к формату (channels, length).
    Принимает (length,), (channels, length) или (length, channels).
    Возвращает None, если y=None.
    """
    if y is None:
        return None
    if y.ndim == 1:
        return y[np.newaxis, :]
    # считаем, что каналов меньше, чем длина
    if y.shape[0] > y.shape[1]:
        return y.T
    return y


def _downsample_for_plot(sig: np.ndarray, max_points: int = 400_000):
    """
    Прореживает сигнал для экономии памяти при отрисовке.
    Возвращает (sig_ds, step), где step — шаг прореживания.
    """
    n = sig.shape[-1]
    if n <= max_points:
        return sig, 1
    step = int(np.ceil(n / max_points))
    return sig[..., ::step], step


def _seconds_axis(n_points: int, sr: int, step: int) -> np.ndarray:
    """
    Создаёт ось времени (секунды) для прореженного сигнала.
    """
    return (np.arange(n_points, dtype=np.float32) * step) / float(sr)


def _safe_len_seconds(y: np.ndarray, sr: int) -> float:
    if y is None or sr is None:
        return 0.0
    y2d = _ensure_2d_channels_first(y)
    return float(y2d.shape[1]) / float(sr)


# ------------------------------
# Таблицы интервалов (универсально)
# ------------------------------
def _normalize_rows(rows, markers, is_answers=False):
    """
    Приводит данные интервалов к списку словарей:
    {"#","Начало","Метка","Конец","Длит."}

    rows может быть:
      - списком словарей {"Начало","Метка","Конец"}  (как передаётся из main сейчас)
      - списком кортежей (start, end)                (старый формат)
    markers — либо display_markers [(t,label)], либо markers [(start,end)].
    """
    out = []
    # подготовим список времён меток
    marker_times = []
    if isinstance(markers, (list, tuple)) and len(markers) > 0:
        # display_markers: (time, label), markers: (start, end)
        for m in markers:
            try:
                marker_times.append(float(m[0]))
            except Exception:
                marker_times.append(None)

    if not isinstance(rows, (list, tuple)) or len(rows) == 0:
        return out

    # формат словарей?
    if isinstance(rows[0], dict):
        for i, r in enumerate(rows):
            try:
                s = float(r.get("Начало", 0.0))
            except Exception:
                s = 0.0
            try:
                e = float(r.get("Конец", 0.0))
            except Exception:
                e = 0.0
            m = r.get("Метка", "-")
            dur = max(0.0, e - s)
            out.append({
                "#": i + 1,
                "Начало": f"{s:.2f}",
                "Метка": f"{m}" if isinstance(m, str) else (f"{m:.2f}" if m is not None else "-"),
                "Конец": f"{e:.2f}",
                "Длит.": f"{dur:.2f}",
            })
        return out

    # иначе считаем кортежи (start, end)
    for i, seg in enumerate(rows):
        # допускаем вложенные структуры — аккуратно достаём первые 2 числа
        s, e = None, None
        if isinstance(seg, (list, tuple)) and len(seg) >= 2:
            try:
                s = float(seg[0])
                e = float(seg[1])
            except Exception:
                pass
        if s is None or e is None:
            continue
        m = marker_times[i] if i < len(marker_times) else None
        dur = max(0.0, float(e) - float(s))
        out.append({
            "#": i + 1,
            "Начало": f"{float(s):.2f}",
            "Метка": f"{m:.2f}" if m is not None else "-",
            "Конец": f"{float(e):.2f}",
            "Длит.": f"{dur:.2f}",
        })
    return out


def _build_interval_tables(parent: tk.Widget, rows_q: list, rows_a: list):
    """
    Строит две таблицы (вопросы и ответы) с колонками:
    #, Начало, Метка, Конец, Длит.
    rows_q / rows_a — уже нормализованные списки словарей.
    """
    frame = tk.Frame(parent)
    frame.pack(fill="x", pady=(10, 0))

    def _make_table(title: str):
        lab = tk.Label(frame, text=title, font=("Arial", 10, "bold"))
        lab.pack(anchor="w")
        cols = ("#", "Начало", "Метка", "Конец", "Длит.")
        tree = ttk.Treeview(frame, columns=cols, show="headings", height=12)
        for c, w in zip(cols, (50, 100, 120, 100, 80)):
            tree.heading(c, text=c)
            tree.column(c, width=w, anchor="center")
        tree.pack(fill="x", pady=5)
        return tree

    tree_q = _make_table("Канал 1 — Вопросы (до 30)")
    tree_a = _make_table("Канал 2 — Ответы (до 30)")

    for row in rows_q[:30]:
        tree_q.insert("", "end", values=(row["#"], row["Начало"], row["Метка"], row["Конец"], row["Длит."]))
    for row in rows_a[:30]:
        tree_a.insert("", "end", values=(row["#"], row["Начало"], row["Метка"], row["Конец"], row["Длит."]))


def show_latent_tables(app, q_rows, a_rows):
    """
    Открывает окно с двумя таблицами.
    Принимает ИЛИ:
      - списки словарей {"Начало","Метка","Конец"} (как формирует main сейчас)
      - списки кортежей (start,end)  + время меток берётся из app.display_markers/app.markers
    """
    win = tk.Toplevel(app.root)
    win.title("Интервальные таблицы (Q/A)")

    markers_source = getattr(app, "display_markers", None)
    if not markers_source:
        markers_source = getattr(app, "markers", None)

    rows_q = _normalize_rows(q_rows, markers_source, is_answers=False)
    rows_a = _normalize_rows(a_rows, markers_source, is_answers=True)

    _build_interval_tables(win, rows_q, rows_a)


# ------------------------------
# Основной график
# ------------------------------
def draw_waveform(
    app,
    segments=None,
    threshold=None,
    series_lines=None,
    max_points_per_channel: int = 400_000,
    q_segments=None,
    a_segments=None,
    segments_ch2=None,   # оставлено для обратной совместимости
):
    """
    Рисует волны оригинала и обработанного сигнала.
    - Если моно (свободный режим): 'segments' подсвечиваются на единственном графике.
    - Если стерео (5:6): подсветка разнесена по каналам:
        левый — вопросы (q_segments), правый — ответы (a_segments/segments_ch2).
      Если q_segments/a_segments не переданы, пытаемся взять из app.current_q_segments / app.current_a_segments.
    Память экономится за счёт прореживания (max_points_per_channel).
    """
    # очищаем контейнер
    for w in app.canvas_frame.winfo_children():
        w.destroy()

    y = _ensure_2d_channels_first(app.audio_data)
    y_orig = _ensure_2d_channels_first(app.original_audio_data)
    sr = getattr(app, "sr", None)
    if y is None or sr is None:
        return

    is_stereo = (y.shape[0] == 2)
    n_ch = 2 if is_stereo else 1

    # Попытка автоматически взять q/a интервалы из приложения
    if q_segments is None:
        q_segments = getattr(app, "current_q_segments", None)
    if a_segments is None:
        a_segments = getattr(app, "current_a_segments", None)
    if a_segments is None and segments_ch2 is not None:
        a_segments = segments_ch2

    # Создаём фигуру
    fig, axs = plt.subplots(
        n_ch, 1,
        figsize=(12, 6 if is_stereo else 4),
        dpi=100,
        sharex=True
    )
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])

    # Прореживаем и рисуем
    for ch in range(n_ch):
        sig_proc = y[ch] if is_stereo else y[0]
        sig_orig = y_orig[ch] if (y_orig is not None and is_stereo) else (y_orig[0] if y_orig is not None else None)

        sig_proc_ds, step = _downsample_for_plot(sig_proc, max_points_per_channel)
        t_ds = _seconds_axis(len(sig_proc_ds), sr, step)

        axs[ch].plot(t_ds, sig_proc_ds, label='Обработанный', alpha=0.9)

        if sig_orig is not None:
            sig_orig_ds, _ = _downsample_for_plot(sig_orig, max_points_per_channel)
            m = min(len(sig_orig_ds), len(t_ds))
            axs[ch].plot(t_ds[:m], sig_orig_ds[:m], label='Оригинал', alpha=0.5)

        axs[ch].set_ylabel(f"Канал {ch+1}" if is_stereo else "Амплитуда")

    # --- Подсветка интервалов ---
    if not is_stereo:
        # моно («свободный») — подсвечиваем на единственном графике
        if segments:
            for start, end in segments:
                axs[0].axvline(start, color='green', linestyle='--', linewidth=1)
                axs[0].axvline(end, color='red', linestyle='--', linewidth=1)
                axs[0].axvspan(start, end, facecolor='green', alpha=0.2)
    else:
        # стерео (5:6)
        # Вопросы на канале 1
        if q_segments:
            for start, end in q_segments:
                axs[0].axvspan(start, end, facecolor='green', alpha=0.25)
        # Ответы на канале 2
        if a_segments:
            for start, end in a_segments:
                axs[1].axvspan(start, end, facecolor='orange', alpha=0.25)

    # --- Порог ---
    if threshold is not None:
        axs[0].axhline(threshold, color='purple', linestyle='--', label='Порог')

    # --- Маркеры и штриховка (на всех видимых каналах) ---
    if getattr(app, "display_markers", None):
        for ch in range(n_ch):
            ax = axs[ch]
            for x, label in app.display_markers:
                ax.axvline(x, color='yellow', linestyle='-', linewidth=0.8)
                # мягкая штриховка +/-3.5 сек
                ax.axvspan(x - 3.5, x + 3.5, facecolor='grey', alpha=0.08, hatch='////')
                ax.text(
                    x, 0.95, label,
                    transform=ax.get_xaxis_transform(),
                    rotation=90, va='top', ha='center',
                    fontsize=8, color='darkorange'
                )

    # --- Разделители серий ---
    if series_lines:
        for x in series_lines:
            axs[0].axvline(x, color='purple', linestyle='-.', linewidth=1.5)

    axs[-1].set_xlabel("Время (сек)")
    axs[0].set_title("Сигнал: оригинал vs обработанный")
    axs[0].legend(loc='upper right')

    plt.tight_layout()

    # Встраиваем в Tk
    canvas = FigureCanvasTkAgg(fig, master=app.canvas_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)


# ------------------------------
# Графики по сериям (также с прореживанием)
# ------------------------------
def plot_series_segments(app, max_points_per_channel: int = 400_000):
    """
    Окна по сериям (6 меток на серию). В стерео-режиме отображаются оба канала.
    """
    if not getattr(app, "markers", None) or app.audio_data is None:
        return

    y = _ensure_2d_channels_first(app.audio_data)
    y_orig = _ensure_2d_channels_first(app.original_audio_data)
    sr = app.sr
    is_stereo = (y.shape[0] == 2)
    n_ch = 2 if is_stereo else 1

    num_series = len(app.markers) // 6
    for i in range(num_series):
        idx_start = i * 6
        idx_end = idx_start + 5

        start_time = app.markers[idx_start][0]
        end_time = app.markers[idx_end][1]
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        dur_samples = end_sample - start_sample
        if dur_samples <= 0:
            continue

        fig, axs = plt.subplots(n_ch, 1, figsize=(10, 5 if is_stereo else 4), sharex=True, dpi=100)
        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])

        # Подготовим время после прореживания
        step = int(np.ceil(dur_samples / max_points_per_channel))
        step = max(1, step)
        idx = np.arange(start_sample, end_sample, step, dtype=np.int64)
        t_ds = (idx - start_sample) / float(sr) + start_time  # время в секундах

        for ch in range(n_ch):
            sig = y[ch] if is_stereo else y[0]
            seg_proc = sig[start_sample:end_sample:step]
            axs[ch].plot(t_ds, seg_proc, label="Обработанный", alpha=0.9)

            if y_orig is not None:
                sig0 = y_orig[ch] if is_stereo else y_orig[0]
                seg_orig = sig0[start_sample:end_sample:step]
                m = min(len(seg_orig), len(t_ds))
                axs[ch].plot(t_ds[:m], seg_orig[:m], label="Оригинал", alpha=0.5)

        # метки
        for j in range(idx_start, idx_end + 1):
            x = app.markers[j][0]
            for ch in range(n_ch):
                axs[ch].axvline(x, color='yellow', linestyle='-', linewidth=0.8)
                # подпись, если есть
                if getattr(app, "display_markers", None) and j < len(app.display_markers):
                    label = app.display_markers[j][1]
                    axs[ch].text(
                        x, 0.95, label, transform=axs[ch].get_xaxis_transform(),
                        rotation=90, va='top', ha='center',
                        fontsize=8, color='darkorange'
                    )
                axs[ch].axvspan(x - 3.5, x + 3.5, facecolor='grey', alpha=0.08, hatch='////')

        # латентные интервалы
        if getattr(app, "current_q_segments", None):
            for start, end in app.current_q_segments:
                if start >= start_time and end <= end_time:
                    axs[0].axvspan(start, end, facecolor='green', alpha=0.25)
        if is_stereo and getattr(app, "current_a_segments", None):
            for start, end in app.current_a_segments:
                if start >= start_time and end <= end_time:
                    axs[1].axvspan(start, end, facecolor='orange', alpha=0.25)

        axs[0].legend(loc="upper right")
        axs[-1].set_xlabel("Время (сек)")
        axs[0].set_title(f"Серия {i+1}: {start_time:.2f} – {end_time:.2f} сек")

        plt.tight_layout()

        win = tk.Toplevel(app.root)
        win.title(f"Серия {i+1}")
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
