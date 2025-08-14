import soundfile as sf
import simpleaudio as sa
import librosa
import numpy as np
from tkinter import filedialog, messagebox
from ui_plot import draw_waveform

# 🔊 Загрузка аудиофайла
def load_audio(app):
    path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
    if path:
        y, sr = librosa.load(path, sr=None, mono=False)
        
        # Обеспечиваем 2D формат (1, N) или (2, N)
        if y.ndim == 1:
            y = np.expand_dims(y, axis=0)
        elif y.shape[0] > y.shape[1]:
            y = y.T  # Приводим к (каналы, длина)

        y = np.ascontiguousarray(y)

        app.filepath = path
        app.original_audio_data = y
        app.audio_data = y.copy()
        app.sr = sr
        app.current_segments = None
        messagebox.showinfo("Аудио загружено", f"Файл: {path}")

        draw_waveform(app)


# 💾 Сохранение аудиофайла
def save_audio(app):
    if app.audio_data is not None:
        out_path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV", "*.wav")])
        if out_path:
            # Приведение к shape (N,) или (N, 2)
            audio = app.audio_data
            if audio.ndim == 2:
                audio = audio.T  # (каналы, N) → (N, каналы)
            sf.write(out_path, audio, app.sr)
            messagebox.showinfo("Сохранено", f"Файл сохранён как {out_path}")

# ▶ Воспроизведение
def play_audio(app):
    if app.audio_data is not None:
        audio = (app.audio_data * 32767).astype('int16')

        # Обеспечим C-contiguous для simpleaudio
        audio = np.ascontiguousarray(audio)

        if audio.ndim == 1 or audio.shape[0] == 1:
            sa.play_buffer(audio.flatten(), 1, 2, app.sr)
        elif audio.shape[0] == 2:
            sa.play_buffer(audio.T, 2, 2, app.sr)
        else:
            messagebox.showerror("Ошибка", "Неподдерживаемое количество каналов.")
