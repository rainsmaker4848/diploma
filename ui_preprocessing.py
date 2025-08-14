import numpy as np
import librosa
import noisereduce as nr

# --- 🎛️ Нормализация ---
def apply_normalization(y):
    max_val = np.max(np.abs(y))
    if max_val > 0:
        return y / max_val
    return y

# --- 🔇 Обрезка тишины ---
def apply_trim_silence(y, sr, top_db=20):
    trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return trimmed

# --- 🔉 Фильтр шума ---
def apply_noise_filter(y, sr=22050):
    reduced = nr.reduce_noise(y=y, sr=sr)
    return reduced

# --- 🧪 Применение всех трёх при необходимости ---
def apply_preprocessing_pipeline(y, sr, use_noise=True, use_norm=True, use_trim=True):
    """
    Применяет указанные фильтры к аудиосигналу. Работает как с моно, так и с двумя каналами (2ch).
    :param y: np.ndarray — аудиосигнал (1D или 2D)
    :param sr: int — частота дискретизации
    :param use_noise: bool — применять шумоподавление
    :param use_norm: bool — применять нормализацию
    :param use_trim: bool — обрезать тишину
    :return: np.ndarray — обработанный аудиосигнал
    """
    if y.ndim == 1:
        y = _process_one_channel(y, sr, use_noise, use_norm, use_trim)
    elif y.ndim == 2 and y.shape[0] == 2:
        y[0] = _process_one_channel(y[0], sr, use_noise, use_norm, use_trim)
        y[1] = _process_one_channel(y[1], sr, use_noise, use_norm, use_trim)
    else:
        raise ValueError("Ожидается 1D (моно) или 2D (2ch) аудиосигнал.")
    return y

def _process_one_channel(channel, sr, use_noise, use_norm, use_trim):
    if use_noise:
        channel = apply_noise_filter(channel, sr)
    if use_norm:
        channel = apply_normalization(channel)
    if use_trim:
        channel = apply_trim_silence(channel, sr)
    return channel
