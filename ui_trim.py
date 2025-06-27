import librosa

def apply_trim_silence(y, sr):
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    return y_trimmed