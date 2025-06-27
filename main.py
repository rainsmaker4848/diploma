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

# --- 🔧 Импорт пользовательских фильтров ---
from ui_noise import apply_noise_filter
from ui_normalize import apply_normalization
from ui_trim import apply_trim_silence
from ui_phoneme_analysis import PhonemeAnalyzer  # <-- новый импорт

# --- 🧠 Класс приложения с GUI ---
class AudioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Аудио Обработчик")
        self.root.geometry("1000x600")

        # 🧾 Хранилище данных
        self.audio_data = None
        self.original_audio_data = None
        self.sr = None
        self.filepath = ""

        # --- 📌 Боковая панель слева (флажки) ---
        self.left_panel = tk.Frame(root, bg="black", width=200)
        self.left_panel.pack(side="left", fill="y")

        self.flag1 = tk.BooleanVar()
        self.flag2 = tk.BooleanVar()
        self.flag3 = tk.BooleanVar()

        tk.Checkbutton(self.left_panel, text="Фильтр шума", variable=self.flag1,
                       bg="black", fg="white", selectcolor="gray20", activebackground="black").pack(anchor="w")
        tk.Checkbutton(self.left_panel, text="Нормализация", variable=self.flag2,
                       bg="black", fg="white", selectcolor="gray20", activebackground="black").pack(anchor="w")
        tk.Checkbutton(self.left_panel, text="Обрезка тишины", variable=self.flag3,
                       bg="black", fg="white", selectcolor="gray20", activebackground="black").pack(anchor="w")

        # --- 🧰 Кнопки управления ---
        self.controls_frame = tk.Frame(self.left_panel, bg="black")
        self.controls_frame.pack(side="bottom", pady=10)

        tk.Button(self.controls_frame, text="Загрузить аудиофайл", command=self.load_audio).pack(fill="x", pady=2)
        tk.Button(self.controls_frame, text="▶ Прослушать", command=self.play_audio).pack(fill="x", pady=2)
        tk.Button(self.controls_frame, text="💾 Сохранить", command=self.save_audio).pack(fill="x", pady=2)
        tk.Button(self.controls_frame, text="ОБРАБОТАТЬ", command=self.process_audio,
                  font=("Arial", 12), bg="white").pack(fill="x", pady=10)
        tk.Button(self.controls_frame, text="📊 Анализ речи", command=self.analyze_audio).pack(fill="x", pady=5)  # <-- новая кнопка

        # --- 📊 Центральная панель с графиком ---
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
            self.draw_waveform()

    def draw_waveform(self):
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(10, 3), dpi=100)
        if self.original_audio_data is not None:
            librosa.display.waveshow(self.original_audio_data, sr=self.sr, ax=ax,
                                     alpha=0.5, color='gray', label='Оригинал')
        if self.audio_data is not None:
            librosa.display.waveshow(self.audio_data, sr=self.sr, ax=ax,
                                     alpha=0.9, color='blue', label='Обработанный')

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

        if self.flag1.get():
            y = apply_noise_filter(y)
        if self.flag2.get():
            y = apply_normalization(y)
        if self.flag3.get():
            y = apply_trim_silence(y, self.sr)

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
        else:
            messagebox.showwarning("Нет аудио", "Сначала загрузите и обработайте аудиофайл.")

# --- 🚀 Точка входа ---
if __name__ == "__main__":
    root = tk.Tk()
    app = AudioApp(root)
    root.mainloop()
