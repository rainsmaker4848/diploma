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


# ===============================
# Интерактивный график
# ===============================
class _InteractiveWaveform:
    """
    Создаёт фигуру один раз. Далее:
      - колёсико = прокрутка xlim
      - Ctrl + колёсико = зум к курсору
      - клик ЛКМ = переставить плейхед (app.playhead_time)
      - update_playhead(t) двигает красную линию без полной перерисовки
    """
    def __init__(self, app):
        self.app = app
        self.fig = None
        self.canvas = None
        self.axs = None
        self.lines_proc = []
        self.lines_orig = []
        self.playhead_lines = []
        self.t_ds = None
        self.sr = None
        self.is_stereo = False
        self.n_samples = 0
        self.max_points = 400_000

    # критерий, нужно ли пересоздавать
    def needs_rebuild(self, sr, n_samples, is_stereo):
        return (self.fig is None) or (self.sr != sr) or (self.n_samples != n_samples) or (self.is_stereo != is_stereo)

    def build(self, segments=None, threshold=None, series_lines=None,
              q_segments=None, a_segments=None):
        # очистка контейнера
        for w in self.app.canvas_frame.winfo_children():
            w.destroy()

        y = _ensure_2d_channels_first(self.app.audio_data)
        y0 = _ensure_2d_channels_first(self.app.original_audio_data)
        self.sr = self.app.sr
        if y is None or self.sr is None:
            return

        self.is_stereo = (y.shape[0] == 2)
        self.n_samples = y.shape[1]
        n_ch = 2 if self.is_stereo else 1

        # общая ось времени по каналу 0
        sig0_ds, step0 = _downsample_for_plot(y[0] if self.is_stereo else y[0], self.max_points)
        self.t_ds = _seconds_axis(len(sig0_ds), self.sr, step0)

        # создаём фигуру
        self.fig, self.axs = plt.subplots(n_ch, 1, figsize=(12, 6 if self.is_stereo else 4), dpi=100, sharex=True)
        if not isinstance(self.axs, np.ndarray):
            self.axs = np.array([self.axs])

        self.lines_proc, self.lines_orig, self.playhead_lines = [], [], []

        # рисуем каналы
        for ch in range(n_ch):
            sig_p = y[ch] if self.is_stereo else y[0]
            sig_p_ds, step = _downsample_for_plot(sig_p, self.max_points)
            lp, = self.axs[ch].plot(self.t_ds[:len(sig_p_ds)], sig_p_ds, label='Обработанный', alpha=0.9)
            self.lines_proc.append(lp)

            lo = None
            if y0 is not None:
                sig_o = y0[ch] if self.is_stereo else y0[0]
                if sig_o is not None:
                    sig_o_ds, _ = _downsample_for_plot(sig_o, self.max_points)
                    m = min(len(sig_o_ds), len(self.t_ds))
                    lo, = self.axs[ch].plot(self.t_ds[:m], sig_o_ds[:m], label='Оригинал', alpha=0.5)
            self.lines_orig.append(lo)

            self.axs[ch].set_ylabel(f"Канал {ch+1}" if self.is_stereo else "Амплитуда")

            # плейхед (вертикальная линия)
            ph, = self.axs[ch].plot([0, 0], [0, 0], color='red', linewidth=2, alpha=0.9)
            self.playhead_lines.append(ph)

        # подсветки сегментов
        if not self.is_stereo:
            if segments:
                for s, e in segments:
                    self.axs[0].axvspan(s, e, facecolor='green', alpha=0.2)
        else:
            if q_segments:
                for s, e in q_segments:
                    self.axs[0].axvspan(s, e, facecolor='green', alpha=0.25)
            if a_segments:
                for s, e in a_segments:
                    self.axs[1].axvspan(s, e, facecolor='orange', alpha=0.25)

        if threshold is not None:
            self.axs[0].axhline(threshold, color='purple', linestyle='--', label='Порог')

        if getattr(self.app, "display_markers", None):
            for ch in range(n_ch):
                ax = self.axs[ch]
                for x, label in self.app.display_markers:
                    ax.axvline(x, color='yellow', linestyle='-', linewidth=0.8)
                    ax.axvspan(x - 3.5, x + 3.5, facecolor='grey', alpha=0.08, hatch='////')
                    ax.text(x, 0.95, label, transform=ax.get_xaxis_transform(),
                            rotation=90, va='top', ha='center', fontsize=8, color='darkorange')

        if series_lines:
            for x in series_lines:
                self.axs[0].axvline(x, color='purple', linestyle='-.', linewidth=1.5)

        self.axs[-1].set_xlabel("Время (сек)")
        self.axs[0].set_title("Сигнал: оригинал vs обработанный")
        self.axs[0].legend(loc='upper right')
        self.fig.tight_layout()

        # embed
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.app.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # события
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

        # начальная позиция плейхеда
        self.update_playhead(getattr(self.app, "playhead_time", None))

        # xlim по умолчанию — весь диапазон
        xmax = self.t_ds[-1] if len(self.t_ds) else 1.0
        for ax in self.axs:
            ax.set_xlim(0, xmax)
        self.canvas.draw_idle()

    # --- интерактив ---
    def _on_scroll(self, event):
        # определяем направление
        step_val = 0
        if hasattr(event, "step") and event.step is not None:
            step_val = 1 if event.step > 0 else -1
        else:
            # старые бэкенды: event.button in {'up','down'}
            step_val = 1 if getattr(event, "button", "up") == "up" else -1

        # масштаб/панорамирование
        key = (event.key or "").lower() if hasattr(event, "key") else ""
        is_ctrl = ("control" in key) or ("ctrl" in key)

        ax = event.inaxes or (self.axs[-1] if len(self.axs) else None)
        if ax is None:
            return

        xmin, xmax = ax.get_xlim()
        span = max(1e-6, xmax - xmin)
        center = float(event.xdata) if event.xdata is not None else (xmin + xmax) / 2.0

        if is_ctrl:
            # Zoom к курсору
            scale = 0.9 if step_val > 0 else 1.1
            new_span = max(1e-3, span * scale)
            left = center - (center - xmin) * (new_span / span)
            right = left + new_span
            for a in self.axs:
                a.set_xlim(left, right)
        else:
            # Панорамирование
            shift = span * 0.1 * (-1 if step_val > 0 else 1)
            for a in self.axs:
                a.set_xlim(xmin + shift, xmax + shift)

        self.canvas.draw_idle()

    def _on_click(self, event):
        if event.inaxes is None or event.xdata is None:
            return
        # клик — устанавливаем плейхед
        t = max(0.0, float(event.xdata))
        self.app.playhead_time = t
        self.update_playhead(t)

    def update_playhead(self, t):
        if t is None:
            t = -1e9  # спрятать линию
        for ax, ph in zip(self.axs, self.playhead_lines):
            ymin, ymax = ax.get_ylim()
            ph.set_data([t, t], [ymin, ymax])
        if self.canvas:
            self.canvas.draw_idle()


# ------------------------------
# Основной API
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
    playhead_time=None,  # позиция плейхеда в секундах
):
    """
    Интерактивный график:
      - первый вызов создаёт фигуру;
      - повторные вызовы без параметров просто двигают плейхед (без полной перерисовки);
      - если данные/режим поменялись или переданы новые сегменты/порог — график пересобирается.
    """
    y = _ensure_2d_channels_first(app.audio_data)
    sr = getattr(app, "sr", None)
    if y is None or sr is None:
        return

    # дефолт плейхеда
    if playhead_time is None:
        playhead_time = getattr(app, "playhead_time", None)

    is_stereo = (y.shape[0] == 2)
    n_samples = y.shape[1]

    force_rebuild = any(v is not None for v in (segments, threshold, series_lines, q_segments, a_segments, segments_ch2))

    if not hasattr(app, "_interactive_plot"):
        app._interactive_plot = _InteractiveWaveform(app)
        app._interactive_plot.build(segments, threshold, series_lines,
                                    q_segments or getattr(app, "current_q_segments", None),
                                    a_segments or getattr(app, "current_a_segments", None))
    else:
        ip = app._interactive_plot
        if ip.needs_rebuild(sr, n_samples, is_stereo) or force_rebuild:
            app._interactive_plot = _InteractiveWaveform(app)
            app._interactive_plot.build(segments, threshold, series_lines,
                                        q_segments or getattr(app, "current_q_segments", None),
                                        a_segments or getattr(app, "current_a_segments", None))
        else:
            app._interactive_plot.update_playhead(playhead_time)


# ------------------------------
# Графики по сериям (отдельные окна)
# ------------------------------
def plot_series_segments(app, max_points_per_channel: int = 400_000, playhead_time=None):
    """
    Окна по сериям (6 меток на серию). В стерео-режиме отображаются оба канала.
    Если задан playhead_time (или у app есть app.playhead_time) — рисуем вертикальную линию.
    """
    if not getattr(app, "markers", None) or app.audio_data is None:
        return

    if playhead_time is None:
        playhead_time = getattr(app, "playhead_time", None)

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

        # плейхед
        if playhead_time is not None:
            for ax in axs:
                if start_time <= playhead_time <= end_time:
                    ymin, ymax = ax.get_ylim()
                    ax.plot([playhead_time, playhead_time], [ymin, ymax], color='red', linewidth=2, alpha=0.9)

        axs[0].legend(loc="upper right")
        axs[-1].set_xlabel("Время (сек)")
        axs[0].set_title(f"Серия {i+1}: {start_time:.2f} – {end_time:.2f} сек")

        plt.tight_layout()

        win = tk.Toplevel(app.root)
        win.title(f"Серия {i+1}")
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
