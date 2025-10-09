import os
import re
import shutil
import tempfile
import torch
import torchaudio
import numpy as np
import pandas as pd
import whisperx
from tkinter import Toplevel, Frame, BOTH, ttk, messagebox

# === ЖЁСТКИЙ ПУТЬ К FFmpeg (как просил) ===
ffmpeg_dir = r"B:\ffmpeg-7.1.1-full_build\bin"   # <- при необходимости замени на свой
if ffmpeg_dir and ffmpeg_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

if shutil.which("ffmpeg") is None:
    raise EnvironmentError(
        f"[FFMPEG NOT FOUND] ffmpeg.exe не найден. Проверь путь: {ffmpeg_dir}"
    )


class PhonemeAnalyzer:
    """
    Реальная фонетическая разметка (RU) на базе WhisperX:
      1) ASR (русский)
      2) Forced alignment (слова/фонемы)
      3) Таблица фонем + аккуратная стенограмма
      4) Уверенности (score) по фонемам/словам, нормировка внутри слова
    """
    def __init__(self, parent, audio_data, sample_rate):
        self.parent = parent
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.whisperx_model = None
        self.align_model = None
        self.align_metadata = None

        # Хранилища результатов:
        # positions: (label, variants, (start, end), word, confidence)
        self.positions = []
        # исходные сегменты ASR WhisperX, для построения стенограммы
        self._asr_segments = None

    # --------- Вспомогательное ----------
    def _to_mono_tensor(self):
        """
        Приводим вход к (1, T) float32 CPU для torchaudio.save.
        """
        arr = np.asarray(self.audio_data)
        if arr.ndim == 2:
            # Приводим к (C, T)
            x = arr if arr.shape[0] < arr.shape[1] else arr.T
            mono = np.mean(x, axis=0, dtype=np.float32, keepdims=True)  # (1, T)
        else:
            mono = np.expand_dims(arr.astype(np.float32), 0)
        return torch.tensor(mono, dtype=torch.float32, device="cpu")

    # --------- Модели ----------
    def load_models(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # фикс ошибки "Requested float16..." на CPU
        compute_type = "float16" if device == "cuda" else "int8"

        # ASR-модель WhisperX — "medium" и RU
        self.whisperx_model = whisperx.load_model(
            "medium", device=device, language="ru", compute_type=compute_type
        )
        # Модель выравнивания (RU)
        self.align_model, self.align_metadata = whisperx.load_align_model(
            language_code="ru", device=device
        )

    # --- Адаптивный вызов transcribe() для разных версий whisperx ---
    def _safe_transcribe(self, audio):
        """
        Пробуем богатыми параметрами; если версия whisperx их не поддерживает —
        откатываемся на совместимый набор аргументов.
        """
        # Набор "полный": может не поддерживаться твоей версией
        rich_kwargs = dict(
            batch_size=8,
            # В некоторых версиях whisperx этих аргументов нет:
            vad_filter=True,
            condition_on_previous_text=False,
            initial_prompt="Илья, Дарья, Сергей, Мария, Андрей, Екатерина",
            temperature=0.0,
            beam_size=5,
        )
        try:
            return self.whisperx_model.transcribe(audio, **rich_kwargs)
        except TypeError:
            # Уберём потенциально неподдерживаемые ключи и попробуем ещё раз
            safe_kwargs = dict(batch_size=8)
            try:
                return self.whisperx_model.transcribe(audio, **safe_kwargs)
            except TypeError:
                # Самый совместимый путь
                return self.whisperx_model.transcribe(audio)

    # --------- Основной анализ ----------
    def analyze(self):
        """
        Полный цикл:
          - сохранить WAV во временный файл,
          - ASR (WhisperX),
          - forced alignment,
          - сбор фонем в self.positions (с уверенностями),
          - показ таблицы,
          - (параллельно доступны методы build_transcript/get_transcript_text)
        """
        if self.whisperx_model is None or self.align_model is None:
            self.load_models()

        # Временный WAV
        audio_tensor = self._to_mono_tensor()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav = tmp.name
        torchaudio.save(tmp_wav, audio_tensor.cpu(), self.sample_rate)

        try:
            # 1) ASR (адаптивный вызов)
            audio = whisperx.load_audio(tmp_wav)
            asr_result = self._safe_transcribe(audio)
            self._asr_segments = asr_result.get("segments", [])  # для стенограммы

            # 2) Forced alignment
            device = "cuda" if torch.cuda.is_available() else "cpu"
            aligned = whisperx.align(
                asr_result["segments"], self.align_model, self.align_metadata, audio, device
            )

            # 3) Сбор фонем с уверенностями
            self.positions = []
            for wseg in aligned.get("word_segments", []) or []:
                word_text = (wseg.get("word") or "").strip()
                phonemes = wseg.get("phonemes", None)
                word_score = wseg.get("score", None)  # может отсутствовать

                if not phonemes:
                    # fallback на слово, если фонем нет
                    wstart, wend = wseg.get("start"), wseg.get("end")
                    if wstart is None or wend is None or not word_text:
                        continue
                    label = f"{word_text} ({wstart:.2f}–{wend:.2f}с)"
                    conf = float(word_score) if word_score is not None else 1.0
                    self.positions.append(
                        (label, [f"{word_text}: {int(round(conf*100))}%"], (wstart, wend), word_text, conf)
                    )
                    continue

                # Сначала собираем фонемы и их «сырые» conf (если нет score — берём 1.0)
                word_ph_items = []
                for ph in phonemes:
                    ph_sym = (ph.get("phoneme") or "").strip()
                    ph_start = ph.get("start", None)
                    ph_end = ph.get("end", None)
                    ph_score = ph.get("score", None)   # может отсутствовать

                    if not ph_sym or ph_start is None or ph_end is None:
                        continue

                    conf = float(ph_score) if ph_score is not None else 1.0
                    word_ph_items.append((ph_sym, ph_start, ph_end, conf))

                if not word_ph_items:
                    # на всякий случай
                    wstart, wend = wseg.get("start"), wseg.get("end")
                    if wstart is not None and wend is not None and word_text:
                        label = f"{word_text} ({wstart:.2f}–{wend:.2f}с)"
                        conf = float(word_score) if word_score is not None else 1.0
                        self.positions.append(
                            (label, [f"{word_text}: {int(round(conf*100))}%"], (wstart, wend), word_text, conf)
                        )
                    continue

                # Нормировка вероятностей ВНУТРИ СЛОВА (распределение между фонемами)
                sum_conf = sum(c for (_, _, _, c) in word_ph_items)
                for ph_sym, ph_start, ph_end, conf in word_ph_items:
                    p = (conf / sum_conf) if sum_conf > 0 else (1.0 / len(word_ph_items))
                    label = f"{ph_sym} ({ph_start:.2f}–{ph_end:.2f}с)"
                    variants = [f"{ph_sym}: {int(round(p * 100))}%"]
                    self.positions.append((label, variants, (ph_start, ph_end), word_text, p))

            # 4) Показ таблицы фонем
            self.display_compact_table()

        finally:
            try:
                os.remove(tmp_wav)
            except OSError:
                pass

    # --------- Стенограмма ---------
    @staticmethod
    def _normalize_spaces(text: str) -> str:
        """
        Правила пробелов и пунктуации для аккуратной строки.
        """
        # убрать пробелы перед знаками препинания
        text = re.sub(r"\s+([,.;:!?…])", r"\1", text)
        # тире: пробелы по обе стороны (если это не дефис внутри слова)
        text = re.sub(r"\s*—\s*", " — ", text)
        # схлопнуть повторные пробелы
        text = re.sub(r"[ \t]{2,}", " ", text)
        # убрать пробелы в начале/конце строки
        return text.strip()

    def build_transcript(self, pause_threshold: float = 0.7) -> str:
        """
        Собирает стенограмму из распознанных сегментов:
          - корректные пробелы между сегментами,
          - перенос строки, если сегмент оканчивается на [.?!…] ИЛИ пауза >= threshold.
        """
        if not self._asr_segments:
            return ""

        lines = []
        cur_line = []

        def flush_line():
            if not cur_line:
                return
            line = " ".join(cur_line)
            line = self._normalize_spaces(line)
            if line:
                lines.append(line)
            cur_line.clear()

        prev_end = None
        for seg in self._asr_segments:
            text = (seg.get("text") or "").strip()
            if not text:
                continue

            start = seg.get("start", None)
            end = seg.get("end", None)

            # перенос по большой паузе
            if prev_end is not None and start is not None:
                if start - prev_end >= pause_threshold:
                    flush_line()

            cur_line.append(text)
            prev_end = end if end is not None else prev_end

            # перенос строки по концу предложения
            if re.search(r"[.?!…]\s*$", text):
                flush_line()

        flush_line()
        return "\n".join(lines)

    def get_transcript_text(self, pause_threshold: float = 0.7) -> str:
        """
        Публичный метод: вернуть собранную стенограмму.
        """
        return self.build_transcript(pause_threshold=pause_threshold)

    # --------- UI (таблица фонем) ----------
    def display_compact_table(self):
        if not self.positions:
            messagebox.showinfo("Фонемный анализ", "Данных для отображения нет.")
            return

        window = Toplevel(self.parent)
        window.title("Фонемная разметка (WhisperX)")
        frame = Frame(window)
        frame.pack(fill=BOTH, expand=True)

        columns = ("Слово", "Фонема", "Начало (с)", "Конец (с)", "Длительность (с)", "Уверенность (%)")
        tree = ttk.Treeview(frame, columns=columns, show='headings')

        for col in columns:
            tree.heading(col, text=col)
            if col == "Слово":
                tree.column(col, width=220, anchor="w")
            elif col == "Фонема":
                tree.column(col, width=120, anchor="center")
            else:
                tree.column(col, width=130, anchor="center")

        for label, variants, (start, end), word, conf in self.positions:
            phoneme = variants[0].split(":")[0] if variants else ""
            dur = round(float(end - start), 3)
            conf_pct = int(round(float(conf) * 100))
            tree.insert('', 'end', values=(word, phoneme, round(start, 3), round(end, 3), dur, conf_pct))

        tree.pack(fill=BOTH, expand=True)

    # --------- Экспорт ----------
    def get_phoneme_dataframe(self):
        """
        Табличный экспорт фонем (совместим с текущим пайплайном) + уверенность.
        """
        if not self.positions:
            return pd.DataFrame()

        rows = []
        for label, variants, (start, end), word, conf in self.positions:
            phoneme = variants[0] if len(variants) > 0 else ""
            rows.append({
                "Слово": word,
                "Символ/Фонема": label,   # например: "t͡s (1.23–1.31с)"
                "Вариант 1": phoneme,     # "<фонема>: XX%"
                "Вариант 2": "",
                "Вариант 3": "",
                "Начало (сек)": round(float(start), 3),
                "Конец (сек)": round(float(end), 3),
                "Длительность (сек)": round(float(end - start), 3),
                "Уверенность (%)": int(round(float(conf) * 100)),
            })
        return pd.DataFrame(rows)
