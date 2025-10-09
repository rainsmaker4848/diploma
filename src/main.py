# --- 📦 Импорт стандартных библиотек ---
import tkinter as tk
import numpy as np
import pandas as pd
import os
import re
from tkinter import messagebox, filedialog

# --- 🖼️ Добавлено: безвыводная отрисовка графиков в файлы ---
import matplotlib
matplotlib.use("Agg")  # чтобы не открывать окна при сохранении картинок
import matplotlib.pyplot as plt

# --- 🔧 Импорт внутренних модулей ---
from ui_buttons import setup_interface
from ui_player import load_audio, save_audio, play_audio
from ui_markers import load_markers_from_file
from ui_plot import draw_waveform, plot_series_segments, show_latent_tables
from ui_speed import change_audio_speed
from ui_phoneme_analysis import PhonemeAnalyzer   # <-- WhisperX-вариант
from ui_slice_filter import apply_marker_zeroing_filter

# ВНИМАНИЕ: ui_latent_free работает с МОНО (1D)!
from ui_latent_free import smooth_signal, compute_threshold, find_nonzero_segments

# 2ch режим 5:6
import ui_latent_experiment
from ui_preprocessing import apply_preprocessing_pipeline


class AudioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Аудио Обработчик")
        self.root.geometry("1300x1000")

        # --- Состояния и данные ---
        self.audio_data = None
        self.original_audio_data = None
        self.sr = None
        self.filepath = ""
        self.marker_path = ""
        self.markers = []            # [(start, end), ...]
        self.display_markers = []    # [(x, label), ...] только чётные
        self.current_segments = None
        self.current_q_segments = None  # вопросы (канал 1) для 5:6
        self.current_a_segments = None  # ответы  (канал 2) для 5:6
        self.phoneme_table = None         # общий (если считали по сведённому сигналу)
        # по-канальным фонемным таблицам/стенограммам
        self.phoneme_table_left = None
        self.phoneme_table_right = None
        self.transcript_text = None       # общий (если моно)
        self.transcript_left = None
        self.transcript_right = None

        # --- UI Элементы ---
        self.left_panel = tk.Frame(root, bg="black", width=220)
        self.left_panel.pack(side="left", fill="y")

        self.graph_frame = tk.Frame(root, bg="orange")
        self.graph_frame.pack(side="left", fill="both", expand=True)

        self.canvas_container = tk.Canvas(self.graph_frame, bg="white")
        self.scroll_x = tk.Scrollbar(self.graph_frame, orient="horizontal", command=self.canvas_container.xview)
        self.scroll_y = tk.Scrollbar(self.graph_frame, orient="vertical", command=self.canvas_container.yview)
        self.canvas_container.configure(xscrollcommand=self.scroll_x.set, yscrollcommand=self.scroll_y.set)
        self.scroll_x.pack(side="bottom", fill="x")
        self.scroll_y.pack(side="right", fill="y")
        self.canvas_container.pack(side="left", fill="both", expand=True)

        self.canvas_frame = tk.Frame(self.canvas_container)
        self.canvas_container.create_window((0, 0), window=self.canvas_frame, anchor="nw")
        self.canvas_frame.bind(
            "<Configure>",
            lambda e: self.canvas_container.configure(scrollregion=self.canvas_container.bbox("all"))
        )

        # --- Переменные интерфейса ---
        self.flag1 = tk.BooleanVar()  # шум
        self.flag2 = tk.BooleanVar()  # нормализация
        self.flag3 = tk.BooleanVar()  # обрезка
        self.flag4 = tk.BooleanVar()  # зануление вне меток
        self.flag5 = tk.BooleanVar()  # энергетические интервалы

        self.speed_factor = tk.DoubleVar(value=1.0)
        self.quantile = tk.DoubleVar(value=0.99)  # начинаем сверху
        self.merge_threshold = tk.DoubleVar(value=1.0)
        self.smooth_window = tk.IntVar(value=5)
        self.experiment_type = tk.StringVar(value="свободный")
        self.experiment_part = tk.StringVar(value="1 ч.")
        self.downsample_factor = tk.IntVar(value=1)  # управляется в ui_buttons/ui_player

        # --- Настройка интерфейса ---
        setup_interface(self)

        # Кнопка стенограммы (добавляем к уже созданным в setup_interface)
        try:
            self.btn_transcript = tk.Button(self.controls_frame, text="📝 Стенограмма", command=self.make_transcript)
            self.btn_transcript.pack(fill="x", pady=5)
        except Exception:
            # На всякий случай, если controls_frame не создан — добавим на левую панель
            self.btn_transcript = tk.Button(self.left_panel, text="📝 Стенограмма", command=self.make_transcript)
            self.btn_transcript.pack(fill="x", pady=5, side="bottom")

    # --- Делегаты ---
    def load_audio(self):
        load_audio(self)

    def save_audio(self):
        save_audio(self)

    def play_audio(self):
        play_audio(self)

    # --- Маркеры ---
    def load_markers(self):
        result = load_markers_from_file(self)
        if result:
            path, markers, labels = result
            self.marker_path = path
            self.markers = markers
            self.display_markers = labels
            messagebox.showinfo("Метки загружены", f"Всего: {len(markers)} меток")
            if self.audio_data is not None:
                draw_waveform(self)

    # --- Анализ фонем (простой, единый) ---
    def analyze_audio(self):
        if self.audio_data is None:
            messagebox.showwarning("Нет аудио", "Сначала загрузите и обработайте аудиофайл.")
            return
        analyzer = PhonemeAnalyzer(self.root, self.audio_data, self.sr)
        analyzer.analyze()
        self.phoneme_table = analyzer.get_phoneme_dataframe()
        # если моно, можно сразу собрать стенограмму из сегментов WhisperX:
        self.transcript_text = analyzer.get_transcript_text(pause_threshold=0.7)

    # --- Вспомогательное: превращаем (C,N) в отдельные 1D каналы ---
    def _split_channels(self, y):
        if y is None:
            return None, None
        if isinstance(y, np.ndarray) and y.ndim == 2:
            # (C, N)
            if y.shape[0] == 2:
                return y[0], y[1]
            # (N, C)
            if y.shape[1] == 2:
                return y[:, 0], y[:, 1]
            # иначе сведём в моно
            mono = np.mean(y, axis=0)
            return mono, None
        if isinstance(y, np.ndarray) and y.ndim == 1:
            return y, None
        return None, None

    # --- Построение стенограммы (WhisperX, пробелы/переносы из сегментов) ---
    def make_transcript(self):
        """
        Если стерео — считаем по каждому каналу отдельно.
        Если моно — один раз по всему сигналу.
        """
        if self.audio_data is None:
            messagebox.showwarning("Нет аудио", "Сначала загрузите и обработайте аудиофайл.")
            return

        chL, chR = self._split_channels(self.audio_data)

        # левый канал
        self.transcript_left = None
        self.phoneme_table_left = None
        if chL is not None:
            try:
                analyzerL = PhonemeAnalyzer(self.root, chL, self.sr)
                analyzerL.analyze()
                self.phoneme_table_left = analyzerL.get_phoneme_dataframe()
                # ключевая строка: аккуратная стенограмма из сегментов WhisperX
                self.transcript_left = analyzerL.get_transcript_text(pause_threshold=0.7)
            except Exception as e:
                print(f"[Transcript L] Ошибка: {e}")

        # правый канал
        self.transcript_right = None
        self.phoneme_table_right = None
        if chR is not None:
            try:
                analyzerR = PhonemeAnalyzer(self.root, chR, self.sr)
                analyzerR.analyze()
                self.phoneme_table_right = analyzerR.get_phoneme_dataframe()
                self.transcript_right = analyzerR.get_transcript_text(pause_threshold=0.7)
            except Exception as e:
                print(f"[Transcript R] Ошибка: {e}")

        # если моно — сохраним совместимость со старым полем
        if chR is None and self.transcript_left is not None:
            self.transcript_text = self.transcript_left

        # Показ в отдельном окне (только текст)
        win = tk.Toplevel(self.root)
        win.title("Стенограмма (по каналам)")
        txt = tk.Text(win, wrap="word", height=25)
        txt.pack(fill="both", expand=True)

        if chR is None:
            txt.insert("1.0", self.transcript_text if self.transcript_text else "(пусто)")
        else:
            sL = self.transcript_left if self.transcript_left else "(пусто)"
            sR = self.transcript_right if self.transcript_right else "(пусто)"
            txt.insert("1.0", f"[Левый]\n{sL}\n\n[Правый]\n{sR}")

        def _save_txt():
            path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text", "*.txt")]
            )
            if not path:
                return
            try:
                if chR is None:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(self.transcript_text or "")
                else:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write("[Левый]\n")
                        f.write((self.transcript_left or "") + "\n\n")
                        f.write("[Правый]\n")
                        f.write(self.transcript_right or "")
                messagebox.showinfo("Готово", f"Стенограмма сохранена: {os.path.basename(path)}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить файл:\n{e}")

        tk.Button(win, text="💾 Сохранить .txt", command=_save_txt).pack(pady=5)

    # --- Основная обработка ---
    def process_audio(self):
        if self.original_audio_data is None:
            messagebox.showwarning("Нет файла", "Сначала загрузите аудиофайл.")
            return

        # 1) Изменение скорости
        y, sr = change_audio_speed(self.original_audio_data.copy(), self.sr, self.speed_factor.get())
        self.sr = sr

        # 2) Предобработка
        y = apply_preprocessing_pipeline(
            y, sr,
            use_noise=self.flag1.get(),
            use_norm=self.flag2.get(),
            use_trim=self.flag3.get()
        )

        # 3) Зануление вне меток
        if self.flag4.get():
            if not self.markers:
                messagebox.showwarning("Нет меток", "Сначала загрузите файл с метками.")
                return
            y = apply_marker_zeroing_filter(y, sr, self.markers)

        # 4) Поиск латентных интервалов
        segments, threshold, series_lines = None, None, []
        self.current_q_segments = None
        self.current_a_segments = None
        q_rows, a_rows = None, None

        if self.flag5.get():
            exp_type = self.experiment_type.get()

            if exp_type == "свободный":
                # ui_latent_free ожидает МОНО. Если 2ch — сведём к моно по модулю.
                if isinstance(y, np.ndarray) and y.ndim == 2:
                    mono = np.mean(np.abs(y), axis=0) * np.sign(np.sum(y, axis=0) + 1e-12)
                else:
                    mono = y

                energy = np.abs(mono)
                smoothed = smooth_signal(energy, self.smooth_window.get())
                threshold = compute_threshold(smoothed, self.quantile.get())
                segments = find_nonzero_segments(smoothed, sr, threshold, self.merge_threshold.get())

            elif exp_type == "5:6":
                if y.ndim != 2 or y.shape[0] != 2:
                    messagebox.showerror("Ошибка", "Для режима 5:6 требуется стерео (2 канала).")
                    return

                q_segments, a_segments = ui_latent_experiment.find_nonzero_segments_stereo(
                    y, sr, self.markers,
                    quantile=float(self.quantile.get()),
                    smooth_window=int(self.smooth_window.get())
                )
                # сохраним отдельно для UI
                self.current_q_segments = q_segments
                self.current_a_segments = a_segments

                # для совместимости — все сегменты в один список (например, для экспорта)
                segments = (q_segments or []) + (a_segments or [])
                series_lines = [m[0] for m in self.markers]

                # таблицы на 30 строк по каналу
                q_rows, a_rows = [], []
                for i in range(min(30, len(q_segments or []))):
                    m1 = float(self.markers[i][0]) if i < len(self.markers) else None
                    s1, e1 = q_segments[i]
                    q_rows.append({"Начало": s1, "Метка": (m1 if m1 is not None else "-"), "Конец": e1})
                for i in range(min(30, len(a_segments or []))):
                    m1 = float(self.markers[i][0]) if i < len(self.markers) else None
                    if i < len(self.markers) - 1:
                        m2 = float(self.markers[i + 1][0])
                        m_str = f"{m1:.3f} → {m2:.3f}"
                    else:
                        m_str = f"{m1:.3f} → END" if m1 is not None else "-"
                    s2, e2 = a_segments[i]
                    a_rows.append({"Начало": s2, "Метка": m_str, "Конец": e2})

                # 💾 Нарезка аудио (вопросы/ответы) + сохранение картинок графиков фрагментов
                try:
                    base = os.path.splitext(os.path.basename(self.filepath))[0] if self.filepath else "audio"
                    out_dir_base = filedialog.askdirectory(title="Папка для нарезанных фрагментов (Q/A) и графиков")
                    if out_dir_base:
                        out_dir = os.path.join(out_dir_base, f"{base}_segments")
                        ui_latent_experiment.export_segments_to_files(y, sr, q_segments, a_segments, out_dir)

                        # --- Добавлено: сохраняем PNG-графики каждого фрагмента ---
                        def _downsample(arr: np.ndarray, max_points: int = 200_000):
                            n = arr.shape[-1]
                            if n <= max_points:
                                return arr, 1
                            step = int(np.ceil(n / max_points))
                            return arr[::step], step

                        def _save_plot(sig_1d: np.ndarray, sr: int, t0: float, t1: float, filepath_png: str, title: str):
                            s = max(0, int(t0 * sr))
                            e = min(len(sig_1d), int(t1 * sr))
                            if e <= s:
                                return
                            seg = sig_1d[s:e]
                            seg_ds, step = _downsample(seg)
                            t = (np.arange(len(seg_ds)) * step) / float(sr) + t0  # абсолютное время
                            fig = plt.figure(figsize=(10, 3), dpi=120)
                            plt.plot(t, seg_ds, linewidth=0.9)
                            plt.title(title)
                            plt.xlabel("Время (сек)")
                            plt.ylabel("Амплитуда")
                            plt.tight_layout()
                            fig.savefig(filepath_png)
                            plt.close(fig)

                        # каналы (вопросы — левый 0, ответы — правый 1)
                        if y.ndim == 2 and y.shape[0] == 2:
                            ch_q = y[0]
                            ch_a = y[1]
                        else:
                            # на всякий случай: моно
                            ch_q = y[0] if y.ndim == 2 else y
                            ch_a = y[0] if y.ndim == 2 else y

                        # сохранить PNG для каждого вопроса
                        for i, (s1, e1) in enumerate(q_segments, start=1):
                            png_path = os.path.join(out_dir, f"question_{i:02d}.png")
                            _save_plot(ch_q, sr, float(s1), float(e1), png_path, f"Вопрос {i}: {s1:.2f}–{e1:.2f} c")

                        # сохранить PNG для каждого ответа
                        for i, (s2, e2) in enumerate(a_segments, start=1):
                            png_path = os.path.join(out_dir, f"answer_{i:02d}.png")
                            _save_plot(ch_a, sr, float(s2), float(e2), png_path, f"Ответ {i}: {s2:.2f}–{e2:.2f} c")

                        messagebox.showinfo(
                            "Нарезка завершена",
                            f"Файлы сохранены в папку:\n{out_dir}\n(аудио и PNG-графики каждого фрагмента)"
                        )
                except Exception as e:
                    messagebox.showwarning("Нарезка не выполнена", f"Не удалось сохранить фрагменты/графики:\n{e}")

        # 5) Обновление состояния и график
        self.audio_data = y
        self.current_segments = segments

        draw_waveform(
            self,
            segments=segments if self.experiment_type.get() == "свободный" else None,
            threshold=threshold,
            series_lines=series_lines,
            q_segments=self.current_q_segments,
            a_segments=self.current_a_segments
        )

        # Таблицы — только для 5:6
        if q_rows is not None and a_rows is not None:
            show_latent_tables(self, q_rows, a_rows)

        messagebox.showinfo("Готово", "Обработка завершена!")

    # --- Экспорт отчёта ---
    def export_report(self):
        if not self.current_segments:
            messagebox.showerror("Ошибка", "Сначала выполните обработку аудио.")
            return

        save_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")])
        if not save_path:
            return

        df_segments = pd.DataFrame([{
            "Начало (сек)": start,
            "Конец (сек)": end,
            "Длительность (сек)": end - start
        } for start, end in self.current_segments])

        stats = df_segments["Длительность (сек)"].describe().rename("Статистика")
        df_stats = pd.DataFrame(stats)

        # Длительность файла с учётом формата (C, N) или (N,)
        if isinstance(self.audio_data, np.ndarray) and self.audio_data.ndim == 2:
            length_samples = self.audio_data.shape[1]
        else:
            length_samples = len(self.audio_data)

        df_metrics = pd.DataFrame([{
            "Файл": os.path.basename(self.filepath),
            "Частота": self.sr,
            "Длительность (сек)": length_samples / self.sr,
            "Скорость": self.speed_factor.get(),
            "Квантиль": self.quantile.get(),
            "Сглаживание (окно)": self.smooth_window.get(),
            "Тип эксперимента": self.experiment_type.get(),
            "Часть": self.experiment_part.get()
        }])

        # Базовая «общая» таблица фонем (если есть)
        df_phonemes = (
            self.phoneme_table.copy()
            if self.phoneme_table is not None
            else pd.DataFrame([{"Фонемы": "не проводился"}])
        )

        # Стенограммы (чистый текст)
        if self.transcript_left is not None or self.transcript_right is not None:
            df_transcript_L = pd.DataFrame([{"Стенограмма (левый)": self.transcript_left or ""}])
            df_transcript_R = pd.DataFrame([{"Стенограмма (правый)": self.transcript_right or ""}])
        else:
            df_transcript_L = None
            df_transcript_R = None

        df_transcript = pd.DataFrame([{"Стенограмма": self.transcript_text}]) if self.transcript_text else pd.DataFrame([{"Стенограмма": ""}])

        # По-канальные фонемные таблицы (если считали в make_transcript)
        df_phonemes_L = self.phoneme_table_left.copy() if self.phoneme_table_left is not None else None
        df_phonemes_R = self.phoneme_table_right.copy() if self.phoneme_table_right is not None else None

        # Надёжная запись Excel с fallback'ами (без обязательного xlsxwriter)
        wrote = False
        try:
            with pd.ExcelWriter(save_path) as writer:
                df_metrics.to_excel(writer, sheet_name="Метрики", index=False)
                df_segments.to_excel(writer, sheet_name="Интервалы", index=False)
                df_stats.to_excel(writer, sheet_name="Статистика")
                df_phonemes.to_excel(writer, sheet_name="Фонемы", index=False)
                # стенограммы
                if df_transcript_L is not None and df_transcript_R is not None:
                    df_transcript_L.to_excel(writer, sheet_name="Стенограмма_L", index=False)
                    df_transcript_R.to_excel(writer, sheet_name="Стенограмма_R", index=False)
                else:
                    df_transcript.to_excel(writer, sheet_name="Стенограмма", index=False)
                # по-канальные фонемы
                if df_phonemes_L is not None:
                    df_phonemes_L.to_excel(writer, sheet_name="Фонемы_L", index=False)
                if df_phonemes_R is not None:
                    df_phonemes_R.to_excel(writer, sheet_name="Фонемы_R", index=False)
            wrote = True
        except Exception:
            pass

        if not wrote:
            try:
                with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
                    df_metrics.to_excel(writer, sheet_name="Метрики", index=False)
                    df_segments.to_excel(writer, sheet_name="Интервалы", index=False)
                    df_stats.to_excel(writer, sheet_name="Статистика")
                    df_phonemes.to_excel(writer, sheet_name="Фонемы", index=False)
                    if df_transcript_L is not None and df_transcript_R is not None:
                        df_transcript_L.to_excel(writer, sheet_name="Стенограмма_L", index=False)
                        df_transcript_R.to_excel(writer, sheet_name="Стенограмма_R", index=False)
                    else:
                        df_transcript.to_excel(writer, sheet_name="Стенограмма", index=False)
                    if df_phonemes_L is not None:
                        df_phonemes_L.to_excel(writer, sheet_name="Фонемы_L", index=False)
                    if df_phonemes_R is not None:
                        df_phonemes_R.to_excel(writer, sheet_name="Фонемы_R", index=False)
                wrote = True
            except Exception:
                pass

        if not wrote:
            base = os.path.splitext(save_path)[0]
            out_dir = base + "_csv_export"
            os.makedirs(out_dir, exist_ok=True)
            try:
                df_metrics.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
                df_segments.to_csv(os.path.join(out_dir, "segments.csv"), index=False)
                df_stats.to_csv(os.path.join(out_dir, "stats.csv"))
                df_phonemes.to_csv(os.path.join(out_dir, "phonemes.csv"), index=False)
                # стенограммы
                if df_transcript_L is not None and df_transcript_R is not None:
                    df_transcript_L.to_csv(os.path.join(out_dir, "transcript_left.csv"), index=False)
                    df_transcript_R.to_csv(os.path.join(out_dir, "transcript_right.csv"), index=False)
                else:
                    df_transcript.to_csv(os.path.join(out_dir, "transcript.csv"), index=False)
                # по-канальные фонемы
                if df_phonemes_L is not None:
                    df_phonemes_L.to_csv(os.path.join(out_dir, "phonemes_left.csv"), index=False)
                if df_phonemes_R is not None:
                    df_phonemes_R.to_csv(os.path.join(out_dir, "phonemes_right.csv"), index=False)

                messagebox.showwarning(
                    "Excel недоступен",
                    f"Библиотеки Excel-движка не найдены. Данные выгружены в CSV:\n{out_dir}"
                )
                wrote = True
            except Exception as e:
                messagebox.showerror("Ошибка экспорта", f"Не удалось сохранить ни в XLSX, ни в CSV:\n{e}")
                return

        if wrote:
            messagebox.showinfo("Отчёт", f"Сохранён: {os.path.basename(save_path)}")

    def plot_series_segments(self):
        plot_series_segments(self)


# --- 🚀 Точка входа ---
if __name__ == "__main__":
    root = tk.Tk()
    app = AudioApp(root)
    root.mainloop()
