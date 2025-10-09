import tkinter as tk
import librosa
import numpy as np

def change_audio_speed(y, sr, speed_factor=1.0):
    """Изменяет скорость аудиофайла с пересчётом частоты дискретизации."""
    if speed_factor == 1.0:
        return y, sr
    new_sr = int(sr * speed_factor)
    y_new = librosa.resample(y, orig_sr=sr, target_sr=new_sr)
    return y_new, new_sr

class SpeedAdjuster:
    """Окно управления скоростью аудио с возможностью применить изменения."""
    def __init__(self, parent, audio_data, sr, apply_callback):
        self.audio_data = audio_data
        self.sr = sr
        self.apply_callback = apply_callback
        self.speed_factor = tk.DoubleVar(value=1.0)

        self.window = tk.Toplevel(parent)
        self.window.title("Изменение скорости")

        tk.Label(self.window, text="Коэффициент скорости (0.5–2.0):").pack()
        tk.Scale(self.window, from_=0.5, to=2.0, resolution=0.1, orient="horizontal",
                 variable=self.speed_factor).pack()

        tk.Button(self.window, text="Применить", command=self.apply).pack(pady=10)

    def apply(self):
        """Применяет изменение скорости и вызывает обратный вызов с результатом."""
        factor = self.speed_factor.get()
        new_audio, new_sr = change_audio_speed(self.audio_data, self.sr, factor)
        self.apply_callback(new_audio, new_sr)
        self.window.destroy()

def apply_speed_change(parent, audio_data, sr, callback):
    """
    Запускает окно изменения скорости.
    :param parent: родительский TK-интерфейс
    :param audio_data: массив аудиосигнала
    :param sr: частота дискретизации
    :param callback: функция, вызываемая с (new_audio, new_sr)
    """
    SpeedAdjuster(parent, audio_data, sr, callback)
