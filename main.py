# --- 📦 Импорт стандартных библиотек ---
import tkinter as tk
import numpy as np
import pandas as pd
import os
from tkinter import messagebox, filedialog

# --- 🔧 Импорт внутренних модулей ---
from ui_buttons import setup_interface
from ui_player import load_audio, save_audio, play_audio
from ui_markers import load_markers_from_file
from ui_plot import draw_waveform, plot_series_segments
from ui_speed import change_audio_speed
from ui_phoneme_analysis import PhonemeAnalyzer
from ui_slice_filter import apply_marker_zeroing_filter
from ui_latent_free import smooth_signal, compute_threshold, find_nonzero_segments
import ui_latent_experiment
from ui_preprocessing import apply_preprocessing_pipeline


class AudioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Аудио Обработчик")
        self.root.geometry("1300x1000")

        # --- Состояния и данные ---
        self.audio_data = None
        self.original_audio_data = None
        self.sr = None
        self.filepath = ""
        self.marker_path = ""
        self.markers = []
        self.display_markers = []
        self.current_segments = None
        self.phoneme_table = None

        # --- UI Элементы ---
        self.left_panel = tk.Frame(root, bg="black", width=200)
        self.left_panel.pack(side="left", fill="y")

        self.graph_frame = tk.Frame(root, bg="orange")
        self.graph_frame.pack(side="left", fill="both", expand=True)
        self.canvas_container = tk.Canvas(self.graph_frame, bg="white")
        self.scroll_x = tk.Scrollbar(self.graph_frame, orient="horizontal", command=self.canvas_container.xview)
        self.scroll_y = tk.Scrollbar(self.graph_frame, orient="vertical", command=self.canvas_container.yview)
        self.canvas_container.configure(xscrollcommand=self.scroll_x.set, yscrollcommand=self.scroll_y.set)
        self.scroll_x.pack(side="bottom", fill="x")
        self.scroll_y.pack(side="right", fill="y")
        self.canvas_container.pack(side="left", fill="both", expand=True)
        self.canvas_frame = tk.Frame(self.canvas_container)
        self.canvas_container.create_window((0, 0), window=self.canvas_frame, anchor="nw")
        self.canvas_frame.bind("<Configure>", lambda e: self.canvas_container.configure(scrollregion=self.canvas_container.bbox("all")))

        # --- Переменные интерфейса ---
        self.flag1 = tk.BooleanVar()
        self.flag2 = tk.BooleanVar()
        self.flag3 = tk.BooleanVar()
        self.flag4 = tk.BooleanVar()
        self.flag5 = tk.BooleanVar()

        self.speed_factor = tk.DoubleVar(value=1.0)
        self.quantile = tk.DoubleVar(value=0.97)
        self.merge_threshold = tk.DoubleVar(value=1.0)
        self.smooth_window = tk.IntVar(value=5)
        self.experiment_type = tk.StringVar(value="свободный")
        self.experiment_part = tk.StringVar(value="1 ч.")

        # --- Настройка интерфейса ---
        setup_interface(self)

    def load_audio(self):
        load_audio(self)

    def save_audio(self):
        save_audio(self)

    def play_audio(self):
        play_audio(self)

    def load_markers(self):
        result = load_markers_from_file(self)
        if result:
            path, markers, labels = result
            self.marker_path = path
            self.markers = markers
            self.display_markers = labels
            messagebox.showinfo("Метки загружены", f"Всего: {len(markers)} меток")

    def analyze_audio(self):
        if self.audio_data is not None:
            analyzer = PhonemeAnalyzer(self.root, self.audio_data, self.sr)
            analyzer.analyze()
            self.phoneme_table = analyzer.get_phoneme_dataframe()
        else:
            messagebox.showwarning("Нет аудио", "Сначала загрузите и обработайте аудиофайл.")

    def process_audio(self):
        if self.original_audio_data is None:
            messagebox.showwarning("Нет файла", "Сначала загрузите аудиофайл.")
            return

        y, sr = change_audio_speed(self.original_audio_data.copy(), self.sr, self.speed_factor.get())
        self.sr = sr

        y = apply_preprocessing_pipeline(y, sr, self.flag1.get(), self.flag2.get(), self.flag3.get())

        if self.flag4.get():
            if not self.markers:
                messagebox.showwarning("Нет меток", "Сначала загрузите файл с метками.")
                return
            y = apply_marker_zeroing_filter(y, sr, self.markers)

        segments, threshold, series_lines = None, None, []

        if self.flag5.get():
            energy = np.abs(y)
            if energy.ndim == 2:
                if self.experiment_type.get() == "свободный":
                    energy = energy[0]  # используем первый канал
                smoothed = smooth_signal(energy, self.smooth_window.get())
            else:
                smoothed = smooth_signal(energy, self.smooth_window.get())

            if self.experiment_type.get() == "свободный":
                threshold = compute_threshold(smoothed, self.quantile.get())
                segments = find_nonzero_segments(smoothed, sr, threshold, self.merge_threshold.get())
            elif self.experiment_type.get() == "5:6" and self.markers:
                if y.ndim == 1:
                    messagebox.showerror("Ошибка", "Для режима 2ch требуется стерео (2 канала)")
                    return
                mode = "2ch" if self.experiment_part.get() == "2 ч." else "1ch"
                q_segments, a_segments = ui_latent_experiment.find_nonzero_segments_stereo(
                    y, sr, self.markers,
                    quantile=self.quantile.get(),
                    smooth_window=self.smooth_window.get()
                )
                segments = q_segments + a_segments
                series_lines = [m[0] for m in self.markers]

        self.audio_data = y
        self.current_segments = segments
        draw_waveform(self, segments=segments, threshold=threshold, series_lines=series_lines)
        messagebox.showinfo("Готово", "Обработка завершена!")

    def export_report(self):
        if not self.current_segments:
            messagebox.showerror("Ошибка", "Сначала выполните обработку аудио.")
            return

        save_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")])
        if not save_path:
            return

        df_segments = pd.DataFrame([{
            "Начало (сек)": start,
            "Конец (сек)": end,
            "Длительность (сек)": end - start
        } for start, end in self.current_segments])

        stats = df_segments["Длительность (сек)"].describe().rename("Статистика")
        df_stats = pd.DataFrame(stats)

        df_metrics = pd.DataFrame([{
            "Файл": os.path.basename(self.filepath),
            "Частота": self.sr,
            "Длительность": len(self.audio_data[0] if self.audio_data.ndim == 2 else self.audio_data) / self.sr,
            "Скорость": self.speed_factor.get(),
            "Квантиль": self.quantile.get(),
            "Сглаживание": self.smooth_window.get(),
            "Тип": self.experiment_type.get(),
            "Часть": self.experiment_part.get()
        }])

        df_phonemes = self.phoneme_table.copy() if self.phoneme_table is not None else pd.DataFrame([{"Фонемы": "не проводился"}])

        with pd.ExcelWriter(save_path) as writer:
            df_metrics.to_excel(writer, sheet_name="Метрики", index=False)
            df_segments.to_excel(writer, sheet_name="Интервалы", index=False)
            df_stats.to_excel(writer, sheet_name="Статистика")
            df_phonemes.to_excel(writer, sheet_name="Фонемы", index=False)

        messagebox.showinfo("Отчёт", f"Сохранён: {os.path.basename(save_path)}")

    def plot_series_segments(self):
        plot_series_segments(self)


# --- 🚀 Точка входа ---
if __name__ == "__main__":
    root = tk.Tk()
    app = AudioApp(root)
    root.mainloop()
