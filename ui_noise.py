import librosa

def apply_noise_filter(y):
    # Простейший pre-emphasis
    return librosa.effects.preemphasis(y)
