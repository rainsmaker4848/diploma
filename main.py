# --- 📦 Импорт стандартных и сторонних библиотек ---
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import librosa
import librosa.display
import simpleaudio as sa
import soundfile as sf
import os
import pandas as pd

# --- 🔧 Импорт пользовательских фильтров ---
from ui_noise import apply_noise_filter
from ui_normalize import apply_normalization
from ui_trim import apply_trim_silence
from ui_phoneme_analysis import PhonemeAnalyzer
from ui_slice_filter import apply_marker_zeroing_filter
from ui_latent_free import smooth_signal, compute_threshold
import ui_latent_free
import ui_latent_experiment


# --- 🧠 Класс приложения с GUI ---
class AudioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Аудио Обработчик")
        self.root.geometry("1300x1000")

        self.audio_data = None
        self.original_audio_data = None
        self.sr = None
        self.filepath = ""
        self.current_segments = None
        self.phoneme_table = None

        # --- 📌 Боковая панель слева ---
        self.left_panel = tk.Frame(root, bg="black", width=200)
        self.left_panel.pack(side="left", fill="y")

        self.flag1 = tk.BooleanVar()
        self.flag2 = tk.BooleanVar()
        self.flag3 = tk.BooleanVar()
        self.flag4 = tk.BooleanVar()
        self.flag5 = tk.BooleanVar()

        tk.Checkbutton(self.left_panel, text="Фильтр шума", variable=self.flag1,
                       bg="black", fg="white", selectcolor="gray20").pack(anchor="w")
        tk.Checkbutton(self.left_panel, text="Нормализация", variable=self.flag2,
                       bg="black", fg="white", selectcolor="gray20").pack(anchor="w")
        tk.Checkbutton(self.left_panel, text="Обрезка тишины", variable=self.flag3,
                       bg="black", fg="white", selectcolor="gray20").pack(anchor="w")
        tk.Checkbutton(self.left_panel, text="Фонемы → зануление вне", variable=self.flag4,
                       bg="black", fg="white", selectcolor="gray20").pack(anchor="w")
        tk.Checkbutton(self.left_panel, text=" Энергетические интервалы", variable=self.flag5,
                       bg="black", fg="white", selectcolor="gray20").pack(anchor="w")

        # --- 🔧 Параметры анализа ---
        tk.Label(self.left_panel, text="⚙ Параметры анализа", bg="black", fg="white").pack(anchor="w", pady=(10, 0))

        self.speed_factor = tk.DoubleVar(value=1.0)
        tk.Label(self.left_panel, text="Скорость", bg="black", fg="white").pack(anchor="w")
        tk.Scale(self.left_panel, from_=0.5, to=2.0, resolution=0.1, orient="horizontal",
                 variable=self.speed_factor, bg="black", fg="white").pack(fill="x")

        self.quantile = tk.DoubleVar(value=0.92)
        tk.Label(self.left_panel, text="Квантиль", bg="black", fg="white").pack(anchor="w")
        tk.Scale(self.left_panel, from_=0.5, to=0.99, resolution=0.01, orient="horizontal",
                 variable=self.quantile, bg="black", fg="white").pack(fill="x")

        self.merge_threshold = tk.DoubleVar(value=1.0)
        tk.Label(self.left_panel, text="Слияние (сек)", bg="black", fg="white").pack(anchor="w")
        tk.Scale(self.left_panel, from_=0.1, to=3.0, resolution=0.1, orient="horizontal",
                 variable=self.merge_threshold, bg="black", fg="white").pack(fill="x")

        self.smooth_window = tk.IntVar(value=5)
        tk.Label(self.left_panel, text="Сглаживание", bg="black", fg="white").pack(anchor="w")
        tk.Scale(self.left_panel, from_=1, to=21, resolution=2, orient="horizontal",
                 variable=self.smooth_window, bg="black", fg="white").pack(fill="x")

        # --- Выбор режима эксперимента ---
        tk.Label(self.left_panel, text="Тип эксперимента", bg="black", fg="white").pack(anchor="w", pady=(10, 0))
        self.experiment_type = tk.StringVar(value="свободный")
        tk.OptionMenu(self.left_panel, self.experiment_type, "свободный", "5:6").pack(fill="x")

        # --- 🧰 Кнопки управления ---
        self.controls_frame = tk.Frame(self.left_panel, bg="black")
        self.controls_frame.pack(side="bottom", pady=10)

        tk.Button(self.controls_frame, text="Загрузить аудиофайл", command=self.load_audio).pack(fill="x", pady=2)
        tk.Button(self.controls_frame, text="▶ Прослушать", command=self.play_audio).pack(fill="x", pady=2)
        tk.Button(self.controls_frame, text="💾 Сохранить", command=self.save_audio).pack(fill="x", pady=2)
        tk.Button(self.controls_frame, text="ОБРАБОТАТЬ", command=self.process_audio,
                  font=("Arial", 12), bg="white").pack(fill="x", pady=10)
        tk.Button(self.controls_frame, text="📊 Анализ речи", command=self.analyze_audio).pack(fill="x", pady=5)
        tk.Button(self.controls_frame, text="📤 Выгрузить отчёт", command=self.export_report).pack(fill="x", pady=5)

        # --- 📊 График ---
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

    def load_audio(self):
        self.filepath = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
        if self.filepath:
            self.original_audio_data, self.sr = librosa.load(self.filepath, sr=None)
            self.audio_data = self.original_audio_data.copy()
            self.current_segments = None
            self.draw_waveform()

    def draw_waveform(self, segments=None, threshold=None, series_lines=None):
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(10, 3), dpi=100)

        if self.original_audio_data is not None:
            librosa.display.waveshow(self.original_audio_data, sr=self.sr, ax=ax,
                                     alpha=0.5, color='gray', label='Оригинал')
        if self.audio_data is not None:
            librosa.display.waveshow(self.audio_data, sr=self.sr, ax=ax,
                                     alpha=0.9, color='blue', label='Обработанный')

        segments_to_draw = segments if segments is not None else self.current_segments
        if segments_to_draw:
            for start, end in segments_to_draw:
                ax.axvline(x=start, color='green', linestyle='--')
                ax.axvline(x=end, color='red', linestyle='--')
                ax.axvspan(start, end, color='green', alpha=0.2)

        if threshold:
            ax.axhline(y=threshold, color='purple', linestyle='--', label='Порог')

        if series_lines:
            for x in series_lines:
                ax.axvline(x=x, color='purple', linestyle='-.', linewidth=2)

        ax.set_title("Сравнение аудиосигналов")
        ax.legend(loc="upper right")

        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def process_audio(self):
        if self.original_audio_data is None:
            messagebox.showwarning("Нет файла", "Сначала загрузите файл.")
            return

        y = self.original_audio_data.copy()
        self.current_segments = None
        threshold = None
        series_lines = []

        if self.flag1.get():
            y = apply_noise_filter(y)
        if self.flag2.get():
            y = apply_normalization(y)
        if self.flag3.get():
            y = apply_trim_silence(y, self.sr)
        if self.flag4.get():
            markers = [1.5, 3.0, 6.2, 7.5]
            y = apply_marker_zeroing_filter(y, self.sr, markers, buffer=0.5)
        if self.flag5.get():
            speed = self.speed_factor.get()
            quantile = self.quantile.get()
            merge = self.merge_threshold.get()
            window = self.smooth_window.get()

            y = librosa.resample(y, orig_sr=self.sr, target_sr=int(self.sr * speed))
            self.sr = int(self.sr * speed)

            energy = np.abs(y)
            smoothed = smooth_signal(energy, window)
            threshold = compute_threshold(smoothed, quantile)

            if self.experiment_type.get() == "свободный":
                segments = ui_latent_free.find_nonzero_segments(smoothed, self.sr, threshold, merge)
            else:
                try:
                    segments = ui_latent_experiment.find_nonzero_segments(smoothed, self.sr, threshold, merge)
                except ValueError as e:
                    messagebox.showerror("Ошибка", str(e))
                    return

                series_lines = [segments[i * 6][0] for i in range(1, 5)]

            self.current_segments = segments
            self.audio_data = y
            self.draw_waveform(segments=segments, threshold=threshold, series_lines=series_lines)

            text = "\n".join([f"{start:.2f} – {end:.2f} сек" for start, end in segments])
            messagebox.showinfo(" Энергетические интервалы", f"Найдено: {len(segments)}\n\n{text}")
            messagebox.showinfo("Готово", "Обработка завершена!")
            return

        self.audio_data = y
        self.draw_waveform()
        messagebox.showinfo("Готово", "Обработка завершена!")

    def play_audio(self):
        if self.audio_data is not None:
            audio = (self.audio_data * 32767).astype(np.int16)
            sa.play_buffer(audio, 1, 2, self.sr)

    def save_audio(self):
        if self.audio_data is not None:
            out_path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV", "*.wav")])
            if out_path:
                sf.write(out_path, self.audio_data, self.sr)
                messagebox.showinfo("Сохранено", f"Файл сохранён как {os.path.basename(out_path)}")

    def analyze_audio(self):
        if self.audio_data is not None:
            analyzer = PhonemeAnalyzer(self.root, self.audio_data, self.sr)
            analyzer.analyze()
            self.phoneme_table = analyzer.get_phoneme_dataframe()
        else:
            messagebox.showwarning("Нет аудио", "Сначала загрузите и обработайте аудиофайл.")

    def export_report(self):
        if not self.current_segments:
            messagebox.showerror("Ошибка", "Сначала выполните обработку аудио с поиском латентных интервалов.")
            return

        save_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")])
        if not save_path:
            return

        # --- 📑 1. Латентные интервалы ---
        df_segments = pd.DataFrame([{
            "Начало (сек)": start,
            "Конец (сек)": end,
            "Длительность (сек)": end - start
        } for start, end in self.current_segments])

        # --- 📊 2. Статистика латентных интервалов ---
        stats = df_segments["Длительность (сек)"].describe().rename("Статистика")
        df_stats = pd.DataFrame(stats)

        # --- ⚙ 3. Общие метрики сигнала ---
        signal_metrics = {
            "Имя файла": [os.path.basename(self.filepath)],
            "Частота дискретизации": [self.sr],
            "Длительность (сек)": [len(self.audio_data) / self.sr],
            "Средняя мощность": [np.mean(self.audio_data**2)],
            "Скорость": [self.speed_factor.get()],
            "Квантиль": [self.quantile.get()],
            "Порог слияния": [self.merge_threshold.get()],
            "Сглаживание": [self.smooth_window.get()],
            "Тип эксперимента": [self.experiment_type.get()]
        }
        df_metrics = pd.DataFrame(signal_metrics)

        # --- 🧠 4. Фонемный анализ ---
        if self.phoneme_table is not None and not self.phoneme_table.empty:
            df_phonemes = self.phoneme_table.copy()
        else:
            df_phonemes = pd.DataFrame([{"Сообщение": "Фонемный анализ не проводился или не дал результатов."}])

        # --- 💾 Сохраняем всё в Excel ---
        try:
            with pd.ExcelWriter(save_path) as writer:
                df_metrics.to_excel(writer, sheet_name="Общие метрики", index=False)
                df_segments.to_excel(writer, sheet_name="Латентные интервалы", index=False)
                df_stats.to_excel(writer, sheet_name="Статистика по длительности")
                df_phonemes.to_excel(writer, sheet_name="Фонемы", index=False)

            messagebox.showinfo("Отчёт сохранён", f"Файл сохранён как: {os.path.basename(save_path)}")
        except Exception as e:
            messagebox.showerror("Ошибка при сохранении", str(e))



# --- 🚀 Точка входа ---
if __name__ == "__main__":
    root = tk.Tk()
    app = AudioApp(root)
    root.mainloop()
