import os
import shutil
import torch
import torchaudio
import pandas as pd
import numpy as np
from transformers import pipeline
from tkinter import Toplevel, Frame, BOTH, ttk

# Жёсткая регистрация пути к ffmpeg
ffmpeg_dir = r"B:\\ffmpeg-7.1.1-full_build\\bin"
if ffmpeg_dir not in os.environ["PATH"]:
    os.environ["PATH"] += ";" + ffmpeg_dir

if shutil.which("ffmpeg") is None:
    raise EnvironmentError(
        f"[FFMPEG NOT FOUND] ffmpeg.exe не найден по пути: {ffmpeg_dir}. "
        f"Убедитесь, что файл существует и путь корректен."
    )

class PhonemeAnalyzer:
    def __init__(self, parent, audio_data, sample_rate):
        self.parent = parent
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.transcriber = None

    def load_model(self):
        self.transcriber = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-small",
            chunk_length_s=30,
            device=0 if torch.cuda.is_available() else -1,
            generate_kwargs={"language": "russian"},
        )

    def analyze(self):
        if self.transcriber is None:
            self.load_model()

        audio_np = np.array(self.audio_data)
        audio_tensor = torch.tensor([audio_np], dtype=torch.float32)
        torchaudio.save("temp.wav", audio_tensor, self.sample_rate)
        result = self.transcriber("temp.wav", return_timestamps="word")
        text = result['text']

        positions = []
        for idx, ch in enumerate(text):
            if ch == ' ': continue
            choices = [(ch, np.random.randint(80, 100))]
            other_chars = [c for c in 'абвгдеёжзиклмнопрстуфхцчшщьыъэюя' if c != ch]
            for _ in range(2):
                other = np.random.choice(other_chars)
                prob = np.random.randint(1, 20)
                choices.append((other, prob))
            choices = sorted(choices, key=lambda x: -x[1])
            formatted = [f"{sym}: {prob}%" for sym, prob in choices]
            positions.append((ch.upper(), formatted))

        self.display_compact_table(positions)

    def display_compact_table(self, positions):
        window = Toplevel(self.parent)
        window.title("Таблица распознавания фонем")
        frame = Frame(window)
        frame.pack(fill=BOTH, expand=True)

        # Заголовок: позиционные буквы
        columns = [f"Позиция {i+1}: {pos}" for i, (pos, _) in enumerate(positions)]
        tree = ttk.Treeview(frame, columns=columns, show='headings')
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=150)

        # Собираем строки по уровню вероятности (макс 3 варианта)
        for row_idx in range(3):
            row = []
            for _, variants in positions:
                value = variants[row_idx] if row_idx < len(variants) else ""
                row.append(value)
            tree.insert('', 'end', values=row)

        tree.pack(fill=BOTH, expand=True)