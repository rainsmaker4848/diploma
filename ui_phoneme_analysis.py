import os
import shutil
import torch
import torchaudio
import numpy as np
import pandas as pd
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
        self.positions = []  # Сохраняем для экспорта

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

        self.positions = []

        for chunk in result.get("chunks", []):
            word = chunk["text"].strip()
            start, end = chunk.get("timestamp", [None, None])
            if not word or start is None or end is None:
                continue

            duration = end - start
            time_per_char = duration / len(word)

            for i, ch in enumerate(word):
                if ch == ' ':
                    continue

                # Пример вероятностного выбора
                choices = [(ch, np.random.randint(80, 100))]
                other_chars = [c for c in 'абвгдеёжзиклмнопрстуфхцчшщьыъэюя' if c != ch]
                for _ in range(2):
                    other = np.random.choice(other_chars)
                    prob = np.random.randint(1, 20)
                    choices.append((other, prob))

                choices = sorted(choices, key=lambda x: -x[1])
                formatted = [f"{sym}: {prob}%" for sym, prob in choices]

                char_start = start + i * time_per_char
                char_end = char_start + time_per_char

                label = f"{ch.upper()} ({char_start:.1f}–{char_end:.1f}с)"
                self.positions.append((label, formatted, (char_start, char_end)))

        self.display_compact_table()

    def display_compact_table(self):
        if not self.positions:
            return

        window = Toplevel(self.parent)
        window.title("Таблица распознавания фонем")
        frame = Frame(window)
        frame.pack(fill=BOTH, expand=True)

        columns = [f"{i+1}: {label}" for i, (label, _, _) in enumerate(self.positions)]
        tree = ttk.Treeview(frame, columns=columns, show='headings')
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=150)

        for row_idx in range(3):
            row = []
            for _, variants, _ in self.positions:
                value = variants[row_idx] if row_idx < len(variants) else ""
                row.append(value)
            tree.insert('', 'end', values=row)

        tree.pack(fill=BOTH, expand=True)

    def get_phoneme_dataframe(self):
        if not self.positions:
            return pd.DataFrame()

        rows = []
        for label, variants, (start, end) in self.positions:
            row = {
                "Символ": label,
                "Вариант 1": variants[0] if len(variants) > 0 else "",
                "Вариант 2": variants[1] if len(variants) > 1 else "",
                "Вариант 3": variants[2] if len(variants) > 2 else "",
                "Начало (сек)": round(start, 3),
                "Конец (сек)": round(end, 3),
                "Длительность (сек)": round(end - start, 3)
            }
            rows.append(row)

        return pd.DataFrame(rows)
