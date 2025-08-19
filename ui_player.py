# ui_player.py
import soundfile as sf
import simpleaudio as sa
import librosa
import numpy as np
from tkinter import filedialog, messagebox

# Пытаемся использовать качественный ресэмплинг, если есть scipy
try:
    from scipy.signal import resample_poly
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# ----------------- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ -----------------

def _to_channels_first(y: np.ndarray) -> np.ndarray:
    """
    Привести сигнал к формату (channels, length).

    Возможные входы:
      - (length,)               -> (1, length)
      - (channels, length)      -> без изменений
      - (length, channels)      -> транспонируем в (channels, length)
    """
    if y.ndim == 1:
        return y[np.newaxis, :]
    # предполагаем, что число каналов гораздо меньше длины
    if y.shape[0] > y.shape[1]:
        return y.T
    return y


def _resample_by_factor(y: np.ndarray, sr: int, factor: int):
    """
    Понизить частоту дискретизации в 'factor' раз (factor >= 1).

    Вход:
      y  — (channels, length) или (length,)
      sr — исходный sr
    Выход:
      (y_resampled, new_sr) с типом float32 и C-contiguous.

    Используем scipy.signal.resample_poly при наличии.
    Если scipy нет — аккуратно децимируем (y[..., ::factor]) с предупреждением.
    """
    factor = max(1, int(factor))
    if factor == 1:
        y = np.ascontiguousarray(y.astype(np.float32, copy=False))
        return y, sr

    target_sr = max(1, sr // factor)

    def _resample_1ch(x: np.ndarray) -> np.ndarray:
        if _HAVE_SCIPY:
            # Удобный и качественный поли-фазный ресэмплер (антиалиас включён)
            x_res = resample_poly(x, up=1, down=factor)
        else:
            # Фоллбек: простая децимация. Качество ниже, но без внешних зависимостей.
            x_res = x[::factor]
        return x_res.astype(np.float32, copy=False)

    if y.ndim == 1:
        y_res = _resample_1ch(y)
    else:
        chans = []
        for ch in range(y.shape[0]):
            chans.append(_resample_1ch(y[ch]))
        # На всякий случай выравниваем длины (могут отличаться на 1 сэмпл)
        min_len = min(len(c) for c in chans)
        chans = [c[:min_len] for c in chans]
        y_res = np.vstack([c[np.newaxis, :] for c in chans])

    y_res = np.ascontiguousarray(y_res)
    return y_res, target_sr


# ----------------- ПУБЛИЧНЫЕ ФУНКЦИИ ДЛЯ APP -----------------

def load_audio(app):
    """
    Загрузка аудио:
      - не сводим к моно (mono=False)
      - приводим к (channels, length)
      - понижаем частоту в выбранное число раз (×1 / ×2 / ×4)
      - храним float32, C-contiguous
    """
    path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.ogg")])
    if not path:
        return

    # Загружаем без ресэмплинга со стороны librosa, чтобы не требовался resampy
    y, sr = librosa.load(path, sr=None, mono=False)  # float64

    # Приводим к (channels, length)
    y = _to_channels_first(y)

    # Фактор понижения частоты из интерфейса (если нет — по умолчанию x1)
    factor = 1
    if hasattr(app, "downsample_var"):
        try:
            factor = int(app.downsample_var.get())
        except Exception:
            factor = 1
    factor = max(1, factor)

    # Ресэмплинг (понижение частоты)
    y, sr = _resample_by_factor(y, sr, factor)

    # Приводим тип и layout
    y = np.ascontiguousarray(y.astype(np.float32, copy=False))

    # Сохраняем в состояние приложения
    app.filepath = path
    app.original_audio_data = y.copy()
    app.audio_data = y.copy()
    app.sr = sr
    app.current_segments = None

    # Предупреждение, если работаем без scipy
    if factor > 1 and not _HAVE_SCIPY:
        messagebox.showwarning(
            "Внимание",
            "Ресэмплинг выполнен без scipy (простая децимация). "
            "Качество может быть ниже. Установите пакет 'scipy' для лучшего качества."
        )

    messagebox.showinfo("Аудио загружено", f"Файл: {path}\nЧастота: {sr} Гц")
    # Отрисовка графика делает внешний модуль ui_plot.draw_waveform(app)
    if hasattr(app, "canvas_frame"):
        try:
            from ui_plot import draw_waveform  # локальный импорт, чтобы избежать циклических
            draw_waveform(app)
        except Exception as e:
            messagebox.showerror("Ошибка графика", str(e))


def save_audio(app):
    if app.audio_data is None:
        return
    out_path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV", "*.wav")])
    if not out_path:
        return

    audio = app.audio_data
    # soundfile ожидает (length,) или (length, channels)
    if audio.ndim == 2:  # (channels, length) -> (length, channels)
        audio_out = audio.T
    else:
        audio_out = audio

    sf.write(out_path, audio_out, app.sr)
    messagebox.showinfo("Сохранено", f"Файл сохранён как {out_path}")


def play_audio(app):
    if app.audio_data is None:
        return

    audio = (app.audio_data * 32767.0).astype(np.int16, copy=False)
    audio = np.ascontiguousarray(audio)  # simpleaudio требует C-contiguous

    try:
        if audio.ndim == 1 or audio.shape[0] == 1:
            # моно: (1, N) или (N,) -> flat
            sa.play_buffer(audio.flatten(), 1, 2, app.sr)
        elif audio.shape[0] == 2:
            # стерео: (2, N) -> (N, 2)
            sa.play_buffer(audio.T, 2, 2, app.sr)
        else:
            messagebox.showerror("Ошибка", "Неподдерживаемое количество каналов для воспроизведения.")
    except Exception as e:
        messagebox.showerror("Ошибка воспроизведения", str(e))
