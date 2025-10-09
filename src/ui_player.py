# ui_player.py
import soundfile as sf
import simpleaudio as sa
import librosa
import numpy as np
import time
from tkinter import filedialog, messagebox

# Попробуем качественный ресэмплинг при наличии SciPy
try:
    from scipy.signal import resample_poly
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# =========================
# ПАРАМЕТРЫ
# =========================
_TICK_HZ = 60  # частота обновления плейхеда (FPS)


# =========================
# УТИЛИТЫ
# =========================
def _to_channels_first(y: np.ndarray) -> np.ndarray:
    """Привести сигнал к формату (channels, length)."""
    if y.ndim == 1:
        return y[np.newaxis, :]
    if y.shape[0] > y.shape[1]:
        return y.T
    return y


def _resample_by_factor(y: np.ndarray, sr: int, factor: int):
    """Понизить частоту дискретизации в 'factor' раз."""
    factor = max(1, int(factor))
    if factor == 1:
        y = np.ascontiguousarray(y.astype(np.float32, copy=False))
        return y, sr

    target_sr = max(1, sr // factor)

    def _resample_1ch(x: np.ndarray) -> np.ndarray:
        if _HAVE_SCIPY:
            x_res = resample_poly(x, up=1, down=factor)
        else:
            x_res = x[::factor]
        return x_res.astype(np.float32, copy=False)

    if y.ndim == 1:
        y_res = _resample_1ch(y)
    else:
        chans = [_resample_1ch(y[ch]) for ch in range(y.shape[0])]
        min_len = min(len(c) for c in chans)
        chans = [c[:min_len] for c in chans]
        y_res = np.vstack([c[np.newaxis, :] for c in chans])

    y_res = np.ascontiguousarray(y_res)
    return y_res, target_sr


def _duration_sec(app) -> float:
    """Длительность текущего аудио (сек)."""
    if getattr(app, "audio_data", None) is None or getattr(app, "sr", None) is None:
        return 0.0
    if app.audio_data.ndim == 2:
        return app.audio_data.shape[1] / float(app.sr)
    return app.audio_data.shape[0] / float(app.sr)


# =========================
# ТАЙМЕР / ПЛЕЙХЕД
# =========================
def _stop_tick(app):
    """Остановить таймер обновления."""
    if getattr(app, "_tick_job", None):
        try:
            app.root.after_cancel(app._tick_job)
        except Exception:
            pass
        app._tick_job = None
    app._last_tick_time = None  # сбрасываем опорное время


def _tick(app):
    """Обновление плейхеда по реальному времени (ось X = секунды)."""
    if not getattr(app, "is_playing", False):
        return

    # если simpleaudio уже закончил буфер — корректно остановим
    try:
        if getattr(app, "play_obj", None) and not app.play_obj.is_playing():
            _stop_audio(app)
            return
    except Exception:
        # если у бэкенда нет is_playing(), просто идём дальше
        pass

    now = time.perf_counter()
    last = getattr(app, "_last_tick_time", None)
    dt = 0.0 if last is None else max(0.0, now - last)
    app._last_tick_time = now

    app.playhead_time += dt
    dur = _duration_sec(app)
    if app.playhead_time >= dur:
        app.playhead_time = dur
        _stop_audio(app)
        return

    # обновляем только линию плейхеда (ui_plot сам делает blit/EMA)
    try:
        from ui_plot import draw_waveform
        draw_waveform(app, playhead_time=app.playhead_time)
    except Exception:
        pass

    app._tick_job = app.root.after(int(1000 / _TICK_HZ), lambda: _tick(app))


def _start_tick(app):
    _stop_tick(app)
    app._last_tick_time = time.perf_counter()
    app._tick_job = app.root.after(int(1000 / _TICK_HZ), lambda: _tick(app))


# =========================
# ПРОИГРЫВАНИЕ
# =========================
def _start_audio_from(app, start_time: float):
    """Запуск проигрывания с позиции start_time (сек)."""
    _stop_audio(app)
    if getattr(app, "audio_data", None) is None:
        return

    sr = app.sr
    start_time = max(0.0, min(start_time, _duration_sec(app)))
    app.playhead_time = start_time

    start_idx = int(start_time * sr)

    try:
        if app.audio_data.ndim == 1 or app.audio_data.shape[0] == 1:
            # моно
            tail = app.audio_data.flatten()[start_idx:]
            buf = np.ascontiguousarray((tail * 32767.0).astype(np.int16, copy=False))
            app.play_obj = sa.play_buffer(buf, 1, 2, sr)
        elif app.audio_data.shape[0] == 2:
            # стерео
            tailL = app.audio_data[0, start_idx:]
            tailR = app.audio_data[1, start_idx:]
            min_len = min(len(tailL), len(tailR))
            stereo = np.stack([tailL[:min_len], tailR[:min_len]], axis=1)  # (N,2)
            buf = np.ascontiguousarray((stereo * 32767.0).astype(np.int16, copy=False))
            app.play_obj = sa.play_buffer(buf, 2, 2, sr)
        else:
            messagebox.showerror("Ошибка", "Неподдерживаемое количество каналов.")
            return
    except Exception as e:
        messagebox.showerror("Ошибка воспроизведения", str(e))
        return

    app.is_playing = True
    _start_tick(app)


def _stop_audio(app):
    """Полная остановка проигрывания."""
    if getattr(app, "play_obj", None):
        try:
            app.play_obj.stop()
        except Exception:
            pass
    app.is_playing = False
    app.play_obj = None
    _stop_tick(app)


# =========================
# ПУБЛИЧНЫЕ ФУНКЦИИ ДЛЯ APP
# =========================
def load_audio(app):
    """Загрузка аудио и первичная отрисовка."""
    path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.ogg")])
    if not path:
        return

    # читаем без ресэмплинга/моно
    y, sr = librosa.load(path, sr=None, mono=False)
    y = _to_channels_first(y)

    # downsample factor
    factor = 1
    for name in ("downsample_factor", "downsample_var"):
        if hasattr(app, name):
            try:
                factor = int(getattr(app, name).get())
                break
            except Exception:
                pass

    y, sr = _resample_by_factor(y, sr, factor)
    y = np.ascontiguousarray(y.astype(np.float32, copy=False))

    # состояние
    app.filepath = path
    app.original_audio_data = y.copy(order="C")
    app.audio_data = y.copy(order="C")
    app.sr = sr
    app.current_segments = None

    app.playhead_time = 0.0
    app.is_playing = False
    app.play_obj = None
    app._tick_job = None
    app._last_tick_time = None

    if factor > 1 and not _HAVE_SCIPY:
        messagebox.showwarning(
            "Внимание",
            "Ресэмплинг выполнен без scipy (простая децимация). "
            "Качество может быть ниже. Установите пакет 'scipy' для лучшего качества."
        )

    messagebox.showinfo("Аудио загружено", f"Файл: {path}\nЧастота: {sr} Гц")

    # первичная отрисовка
    if hasattr(app, "canvas_frame"):
        try:
            from ui_plot import draw_waveform
            draw_waveform(app, playhead_time=0.0)
        except Exception as e:
            messagebox.showerror("Ошибка графика", str(e))

    # хоткеи
    app.root.bind("<space>", lambda e: _toggle_play_pause(app))  # Play/Pause
    app.root.bind("<Shift-space>", lambda e: stop_audio(app))    # Stop
    app.root.bind("<Escape>", lambda e: stop_audio(app))         # Stop


def save_audio(app):
    """Сохранение аудио."""
    if getattr(app, "audio_data", None) is None:
        return

    out_path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV", "*.wav")])
    if not out_path:
        return

    audio = app.audio_data
    audio_out = np.ascontiguousarray(audio.T) if audio.ndim == 2 else np.ascontiguousarray(audio)
    sf.write(out_path, audio_out, app.sr)
    messagebox.showinfo("Сохранено", f"Файл сохранён как {out_path}")


def play_audio(app):
    """▶ Воспроизвести с текущей позиции (плейхед)."""
    start_time = float(getattr(app, "playhead_time", 0.0))
    _start_audio_from(app, start_time)


def pause_audio(app):
    """⏸ Пауза (останавливает звук, позиция сохраняется)."""
    if getattr(app, "play_obj", None):
        try:
            app.play_obj.stop()
        except Exception:
            pass
    app.is_playing = False
    _stop_tick(app)


def stop_audio(app):
    """⏹ Полная остановка и возврат в начало."""
    _stop_audio(app)
    app.playhead_time = 0.0
    try:
        from ui_plot import draw_waveform
        draw_waveform(app, playhead_time=0.0)
    except Exception:
        pass


def _toggle_play_pause(app):
    """Пробел — переключение Play/Pause."""
    if getattr(app, "is_playing", False):
        pause_audio(app)
    else:
        play_audio(app)


# =========================
# ХЕЛПЕР ДЛЯ ui_plot
# =========================
def jump_and_play(app, t: float):
    """
    Перемотать к t (сек) и сразу воспроизвести.
    Удобно вызывать из ui_plot при клике по графику.
    """
    t = float(max(0.0, min(t, _duration_sec(app))))
    app.playhead_time = t
    try:
        from ui_plot import draw_waveform
        draw_waveform(app, playhead_time=t)
    except Exception:
        pass
    play_audio(app)
