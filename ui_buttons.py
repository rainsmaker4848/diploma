# ui_buttons.py
import tkinter as tk

def setup_interface(app):
    """
    Создаёт весь левый интерфейс и вешает обработчики на методы app.
    Должен вызываться из __init__ после инициализации переменных приложения.
    """
    create_left_panel_controls(app)


def create_left_panel_controls(app):
    # --- Левая панель ---
    # ВАЖНО: сначала pack слева, чтобы панель точно оказалась слева от графиков
    app.left_panel = tk.Frame(app.root, bg="black", width=240)
    app.left_panel.pack(side="left", fill="y")

    # ==============================
    #   ПРЕДОБРАБОТКА / АНАЛИЗ
    # ==============================
    app.flag1 = tk.BooleanVar()  # шум
    app.flag2 = tk.BooleanVar()  # нормализация
    app.flag3 = tk.BooleanVar()  # обрезка тишины
    app.flag4 = tk.BooleanVar()  # зануление вне меток
    app.flag5 = tk.BooleanVar()  # поиск латентных интервалов

    tk.Checkbutton(
        app.left_panel, text="Фильтр шума", variable=app.flag1,
        bg="black", fg="white", selectcolor="gray20", anchor="w"
    ).pack(anchor="w", fill="x")
    tk.Checkbutton(
        app.left_panel, text="Нормализация", variable=app.flag2,
        bg="black", fg="white", selectcolor="gray20", anchor="w"
    ).pack(anchor="w", fill="x")
    tk.Checkbutton(
        app.left_panel, text="Обрезка тишины", variable=app.flag3,
        bg="black", fg="white", selectcolor="gray20", anchor="w"
    ).pack(anchor="w", fill="x")
    tk.Checkbutton(
        app.left_panel, text="Фонемы → зануление вне", variable=app.flag4,
        bg="black", fg="white", selectcolor="gray20", anchor="w"
    ).pack(anchor="w", fill="x")
    tk.Checkbutton(
        app.left_panel, text="Энергетические интервалы", variable=app.flag5,
        bg="black", fg="white", selectcolor="gray20", anchor="w"
    ).pack(anchor="w", fill="x")

    # ==============================
    #   ПАРАМЕТРЫ АНАЛИЗА
    # ==============================
    tk.Label(app.left_panel, text="⚙ Параметры анализа", bg="black", fg="white").pack(anchor="w", pady=(10, 0))

    # Скорость
    app.speed_factor = tk.DoubleVar(value=1.0)
    tk.Label(app.left_panel, text="Скорость", bg="black", fg="white").pack(anchor="w")
    tk.Scale(
        app.left_panel, from_=0.5, to=2.0, resolution=0.1, orient="horizontal",
        variable=app.speed_factor, bg="black", fg="white", highlightthickness=0
    ).pack(fill="x")

    # Квантиль (для свободного режима)
    app.quantile_frame = tk.Frame(app.left_panel, bg="black")
    tk.Label(app.quantile_frame, text="Квантиль", bg="black", fg="white").pack(anchor="w")
    app.quantile = tk.DoubleVar(value=0.97)
    tk.Scale(
        app.quantile_frame, from_=0.5, to=0.99, resolution=0.01, orient="horizontal",
        variable=app.quantile, bg="black", fg="white", highlightthickness=0
    ).pack(fill="x")
    app.quantile_frame.pack(fill="x")

    # Слияние сегментов
    app.merge_threshold = tk.DoubleVar(value=1.0)
    tk.Label(app.left_panel, text="Слияние (сек)", bg="black", fg="white").pack(anchor="w")
    tk.Scale(
        app.left_panel, from_=0.1, to=3.0, resolution=0.1, orient="horizontal",
        variable=app.merge_threshold, bg="black", fg="white", highlightthickness=0
    ).pack(fill="x")

    # Сглаживание
    app.smooth_window = tk.IntVar(value=5)
    tk.Label(app.left_panel, text="Сглаживание (окно)", bg="black", fg="white").pack(anchor="w")
    tk.Scale(
        app.left_panel, from_=1, to=21, resolution=2, orient="horizontal",
        variable=app.smooth_window, bg="black", fg="white", highlightthickness=0
    ).pack(fill="x")

    # ==============================
    #   ДИСКРЕТИЗАЦИЯ (ЗАГРУЗКА)
    # ==============================
    # Эти переменные считываются ui_player.load_audio
    # для понижения частоты дискретизации сразу при загрузке файла.
    tk.Label(app.left_panel, text="Частота при загрузке", bg="black", fg="white").pack(anchor="w", pady=(12, 0))

    app.downsample_enable = tk.BooleanVar(value=True)   # по умолчанию включено
    app.downsample_factor = tk.IntVar(value=2)          # по умолчанию в 2 раза
    app.sr_text = tk.StringVar(value="SR: —")           # сюда ui_player установит актуальную SR

    def _toggle_ds():
        state = tk.NORMAL if app.downsample_enable.get() else tk.DISABLED
        ds_scale.configure(state=state)
        ds_label.configure(state=state)

    tk.Checkbutton(
        app.left_panel,
        text="Понижать частоту",
        variable=app.downsample_enable,
        command=_toggle_ds,
        bg="black", fg="white", selectcolor="gray20", anchor="w"
    ).pack(anchor="w", fill="x")

    ds_label = tk.Label(app.left_panel, text="Во сколько раз (1–4)", bg="black", fg="white")
    ds_label.pack(anchor="w")

    ds_scale = tk.Scale(
        app.left_panel, from_=1, to=4, resolution=1, orient="horizontal",
        variable=app.downsample_factor, bg="black", fg="white", highlightthickness=0
    )
    ds_scale.pack(fill="x")

    # Текущая SR (обновляется после загрузки файла)
    tk.Label(app.left_panel, textvariable=app.sr_text, bg="black", fg="white").pack(anchor="w", pady=(2, 8))

    # ==============================
    #   ТИП ЭКСПЕРИМЕНТА
    # ==============================
    tk.Label(app.left_panel, text="Тип эксперимента", bg="black", fg="white").pack(anchor="w", pady=(10, 0))
    app.experiment_type = tk.StringVar(value="свободный")
    app.experiment_part = tk.StringVar(value="1 ч.")

    tk.OptionMenu(
        app.left_panel,
        app.experiment_type,
        "свободный", "5:6",
        command=lambda choice: on_experiment_change(app, choice)
    ).pack(fill="x")

    app.experiment_part_frame = tk.Frame(app.left_panel, bg="black")
    tk.Label(app.experiment_part_frame, text="Часть эксперимента", bg="black", fg="white").pack(anchor="w")
    tk.OptionMenu(app.experiment_part_frame, app.experiment_part, "1 ч.", "2 ч.").pack(fill="x")

    # ==============================
    #   КНОПКИ УПРАВЛЕНИЯ
    # ==============================
    app.controls_frame = tk.Frame(app.left_panel, bg="black")
    app.controls_frame.pack(side="bottom", pady=10, fill="x")

    tk.Button(app.controls_frame, text="Загрузить аудиофайл", command=app.load_audio).pack(fill="x", pady=2)
    tk.Button(app.controls_frame, text="Загрузить метки",    command=app.load_markers).pack(fill="x", pady=2)
    tk.Button(app.controls_frame, text="▶ Прослушать",       command=app.play_audio).pack(fill="x", pady=2)
    tk.Button(app.controls_frame, text="💾 Сохранить",        command=app.save_audio).pack(fill="x", pady=2)
    tk.Button(app.controls_frame, text="ОБРАБОТАТЬ",          command=app.process_audio, font=("Arial", 12), bg="white").pack(fill="x", pady=10)
    tk.Button(app.controls_frame, text="📊 Анализ речи",      command=app.analyze_audio).pack(fill="x", pady=5)
    tk.Button(app.controls_frame, text="📤 Выгрузить отчёт",  command=app.export_report).pack(fill="x", pady=5)

    # Кнопка «по сериям» показывается только в режиме 5:6
    app.series_plot_button = tk.Button(app.controls_frame, text="📈 Отобразить по сериям", command=app.plot_series_segments)

    # Первичная инициализация зависящих от типа элементов
    on_experiment_change(app, app.experiment_type.get())

    # Применяем начальное состояние для блока дискретизации
    _toggle_ds()


def on_experiment_change(app, choice):
    """
    Переключает видимость элементов, зависящих от выбранного типа эксперимента.
    - 'свободный': показываем квантиль, скрываем «часть эксперимента» и кнопку серий
    - '5:6'      : скрываем квантиль, показываем «часть эксперимента» и кнопку серий
    """
    if choice == "свободный":
        # показать квантиль
        if app.quantile_frame.winfo_manager() == "":
            app.quantile_frame.pack(fill="x")
        # скрыть блок части эксперимента
        app.experiment_part_frame.pack_forget()
        # скрыть кнопку серий
        if hasattr(app, "series_plot_button"):
            app.series_plot_button.pack_forget()

    elif choice == "5:6":
        # скрыть квантиль
        app.quantile_frame.pack_forget()
        # показать блок части
        if app.experiment_part_frame.winfo_manager() == "":
            app.experiment_part_frame.pack(fill="x")
        # показать кнопку серий
        if hasattr(app, "series_plot_button"):
            app.series_plot_button.pack(fill="x", pady=5)
