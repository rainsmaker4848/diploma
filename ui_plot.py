import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def draw_waveform(app, segments=None, threshold=None, series_lines=None):
    for widget in app.canvas_frame.winfo_children():
        widget.destroy()

    y = app.audio_data
    y_orig = app.original_audio_data
    sr = app.sr
    is_stereo = y.shape[0] == 2  # 2ch режим

    fig, axs = plt.subplots(2 if is_stereo else 1, 1, figsize=(12, 4 if not is_stereo else 6), dpi=100, sharex=True)
    if not isinstance(axs, np.ndarray):
        axs = [axs]

    t = np.linspace(0, y.shape[1] / sr if is_stereo else y.shape[0] / sr, y.shape[1] if is_stereo else y.shape[0])

    # --- Отображение сигналов ---
    for ch in range(2 if is_stereo else 1):
        sig_proc = y[ch] if is_stereo else y
        sig_orig = y_orig[ch] if is_stereo else y_orig
        axs[ch].plot(t, sig_orig, label='Оригинал', alpha=0.5, color='gray')
        axs[ch].plot(t, sig_proc, label='Обработанный', alpha=0.9, color='blue')
        axs[ch].set_ylabel(f"Канал {ch+1}" if is_stereo else "Амплитуда")

    # --- Сегменты ---
    if segments:
        for start, end in segments:
            axs[0].axvline(start, color='green', linestyle='--')
            axs[0].axvline(end, color='red', linestyle='--')
            axs[0].axvspan(start, end, color='green', alpha=0.2)

    # --- Порог ---
    if threshold is not None:
        axs[0].axhline(threshold, color='purple', linestyle='--', label='Порог')

    # --- Маркеры + штриховка ---
    for ch in range(1 if not is_stereo else 2):
        for x, label in app.display_markers:
            axs[ch].axvline(x, color='yellow', linestyle='-')
            axs[ch].axvspan(x - 3.5, x + 3.5, color='grey', alpha=0.1, hatch='////', edgecolor='orange')
            axs[ch].text(x, 0.95, label, transform=axs[ch].get_xaxis_transform(),
                         rotation=90, verticalalignment='top', horizontalalignment='center',
                         fontsize=8, color='darkorange')

    # --- Разделители серий ---
    if series_lines:
        for x in series_lines:
            axs[0].axvline(x, color='purple', linestyle='-.', linewidth=2)

    axs[-1].set_xlabel("Время (сек)")
    axs[0].set_title("Сигнал: оригинал vs обработанный")
    axs[0].legend(loc='upper right')
    plt.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=app.canvas_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

def plot_series_segments(app):
    if not app.markers or app.audio_data is None:
        return

    y = app.audio_data
    y_orig = app.original_audio_data
    sr = app.sr
    is_stereo = y.shape[0] == 2
    num_series = len(app.markers) // 6

    for i in range(num_series):
        idx_start = i * 6
        idx_end = idx_start + 5

        start_time = app.markers[idx_start][0]
        end_time = app.markers[idx_end][1]
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        fig, axs = plt.subplots(2 if is_stereo else 1, 1, figsize=(10, 5), sharex=True)
        if not isinstance(axs, np.ndarray):
            axs = [axs]

        t = np.linspace(start_time, end_time, end_sample - start_sample)

        for ch in range(2 if is_stereo else 1):
            y_cut = y[ch][start_sample:end_sample] if is_stereo else y[start_sample:end_sample]
            y_orig_cut = y_orig[ch][start_sample:end_sample] if is_stereo else y_orig[start_sample:end_sample]
            axs[ch].plot(t, y_orig_cut, label='Оригинал', alpha=0.5, color='gray')
            axs[ch].plot(t, y_cut, label='Обработанный', alpha=0.9, color='blue')

        for j in range(idx_start, idx_end + 1):
            x = app.markers[j][0]
            for ch in range(1 if not is_stereo else 2):
                axs[ch].axvline(x, color='yellow', linestyle='-')
                if j < len(app.display_markers):
                    label = app.display_markers[j][1]
                    axs[ch].text(x, 0.95, label, transform=axs[ch].get_xaxis_transform(),
                                 rotation=90, verticalalignment='top', horizontalalignment='center',
                                 fontsize=8, color='darkorange')
                axs[ch].axvspan(x - 3.5, x + 3.5, color='grey', alpha=0.1, hatch='////', edgecolor='orange')

        if app.current_segments:
            for start, end in app.current_segments:
                if start >= start_time and end <= end_time:
                    axs[0].axvspan(start, end, color='green', alpha=0.2)

        axs[0].legend()
        axs[-1].set_xlabel("Время (сек)")
        axs[0].set_title(f"Серия {i+1}: {start_time:.2f} – {end_time:.2f} сек")
        plt.tight_layout()
        win = tk.Toplevel(app.root)
        win.title(f"Серия {i+1}")
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
