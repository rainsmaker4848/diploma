import numpy as np
import librosa
import noisereduce as nr

# --- 🎛️ Нормализация ---
def apply_normalization(y: np.ndarray) -> np.ndarray:
    """
    Нормализация амплитуды до [-1, 1] по каждому каналу отдельно.
    Поддерживает 1D (моно) и 2D (2ch: shape = (2, N)).
    """
    y = np.asarray(y)
    if y.ndim == 1:
        max_val = np.max(np.abs(y)) if y.size else 0.0
        return y / max_val if max_val > 0 else y
    elif y.ndim == 2:
        out = []
        for ch in range(y.shape[0]):
            ch_sig = y[ch]
            max_val = np.max(np.abs(ch_sig)) if ch_sig.size else 0.0
            out.append(ch_sig / max_val if max_val > 0 else ch_sig)
        return np.ascontiguousarray(np.vstack(out))
    else:
        raise ValueError("apply_normalization ожидает 1D (моно) или 2D (2ch) массив")


# --- 🔇 Обрезка тишины ---
def apply_trim_silence(y: np.ndarray, sr: int, top_db: float = 20) -> np.ndarray:
    """
    Обрезка тишины.
    - Для 1D: стандартный librosa.effects.trim для моно.
    - Для 2D: считаем моно-микс (среднее по каналам) -> получаем единые границы (idx)
      и применяем эти же границы к обоим каналам. Так каналы сохраняют одинаковую длину.
    """
    y = np.asarray(y)

    if y.ndim == 1:
        trimmed, _ = librosa.effects.trim(y, top_db=top_db)
        return np.ascontiguousarray(trimmed)

    if y.ndim == 2:
        # моно-микс для определения общих границ
        mix = np.mean(y, axis=0)
        _, idx = librosa.effects.trim(mix, top_db=top_db)
        start, end = idx[0], idx[1]

        # защита от пустых диапазонов
        start = int(max(0, start))
        end = int(min(y.shape[1], end))
        if end <= start:
            # если вдруг всё "тишина" — вернём как есть
            return np.ascontiguousarray(y)

        # одинаково режем оба канала
        left = y[0, start:end]
        right = y[1, start:end]
        return np.ascontiguousarray(np.vstack([left, right]))

    raise ValueError("apply_trim_silence ожидает 1D (моно) или 2D (2ch) массив")


# --- 🔉 Фильтр шума ---
def apply_noise_filter(y: np.ndarray, sr: int = 22050) -> np.ndarray:
    """
    Шумоподавление через noisereduce.
    - Для 1D: напрямую.
    - Для 2D: по канально, затем собираем обратно (2, N).
    """
    y = np.asarray(y)

    if y.ndim == 1:
        reduced = nr.reduce_noise(y=y, sr=sr)
        return np.ascontiguousarray(reduced)

    if y.ndim == 2:
        left = nr.reduce_noise(y=y[0], sr=sr)
        right = nr.reduce_noise(y=y[1], sr=sr)
        return np.ascontiguousarray(np.vstack([left, right]))

    raise ValueError("apply_noise_filter ожидает 1D (моно) или 2D (2ch) массив")


# --- 🧪 Единый пайплайн предобработки ---
def apply_preprocessing_pipeline(
    y: np.ndarray,
    sr: int,
    use_noise: bool = True,
    use_norm: bool = True,
    use_trim: bool = True,
    top_db: float = 20,
) -> np.ndarray:
    """
    Применяет выбранные шаги предобработки к аудиосигналу.
    Поддерживает 1D (моно) и 2D (2ch: (2, N)).

    Порядок шагов:
      1) Обрезка тишины (общая для обоих каналов, чтобы длина совпадала)
      2) Шумоподавление (по канально)
      3) Нормализация (по канально)

    Возвращает C-contiguous numpy массив.
    """
    y = np.asarray(y)

    # 1) Trim — для стерео применяем общие границы, чтобы не получить разные длины каналов
    if use_trim:
        y = apply_trim_silence(y, sr, top_db=top_db)

    # 2) Noise reduction
    if use_noise:
        y = apply_noise_filter(y, sr=sr)

    # 3) Normalization
    if use_norm:
        y = apply_normalization(y)

    return np.ascontiguousarray(y)
