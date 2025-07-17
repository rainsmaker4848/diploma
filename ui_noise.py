import numpy as np
import scipy.signal

def apply_noise_filter(signal, sr=22050, background_quantile=0.1, peak_quantile=0.96):
    """
    Убирает шум и фон из аудиосигнала:
    - Фон убирается по нижнему квантилю амплитуд
    - Затем отбрасываются значения ниже верхнего квантиля (максимумов)
    - Сглаживание применяется для финальной чистки
    """
    # --- Удаление фоновой активности ---
    abs_signal = np.abs(signal)
    bg_level = np.quantile(abs_signal, background_quantile)
    no_bg = np.where(abs_signal >= bg_level, signal, 0)

    # --- Удаление слабых шумов по квантилю максимума ---
    cleaned_abs = np.abs(no_bg)
    if np.any(cleaned_abs > 0):
        peak_level = np.quantile(cleaned_abs[cleaned_abs > 0], peak_quantile)
        cleaned = np.where(cleaned_abs >= peak_level, no_bg, 0)
    else:
        cleaned = no_bg

    # --- Сглаживание для устранения скачков ---
    smoothed = scipy.signal.medfilt(cleaned, kernel_size=5)
    return smoothed
