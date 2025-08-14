import tkinter as tk

def setup_interface(app):
    create_left_panel_controls(app)

def create_left_panel_controls(app):
    app.left_panel = tk.Frame(app.root, bg="black", width=200)
    app.left_panel.pack(side="left", fill="y")

    # --- Чекбоксы ---
    app.flag1 = tk.BooleanVar()
    app.flag2 = tk.BooleanVar()
    app.flag3 = tk.BooleanVar()
    app.flag4 = tk.BooleanVar()
    app.flag5 = tk.BooleanVar()

    tk.Checkbutton(app.left_panel, text="Фильтр шума", variable=app.flag1, bg="black", fg="white", selectcolor="gray20").pack(anchor="w")
    tk.Checkbutton(app.left_panel, text="Нормализация", variable=app.flag2, bg="black", fg="white", selectcolor="gray20").pack(anchor="w")
    tk.Checkbutton(app.left_panel, text="Обрезка тишины", variable=app.flag3, bg="black", fg="white", selectcolor="gray20").pack(anchor="w")
    tk.Checkbutton(app.left_panel, text="Фонемы → зануление вне", variable=app.flag4, bg="black", fg="white", selectcolor="gray20").pack(anchor="w")
    tk.Checkbutton(app.left_panel, text="Энергетические интервалы", variable=app.flag5, bg="black", fg="white", selectcolor="gray20").pack(anchor="w")

    # --- Параметры анализа ---
    tk.Label(app.left_panel, text="⚙ Параметры анализа", bg="black", fg="white").pack(anchor="w", pady=(10, 0))

    app.speed_factor = tk.DoubleVar(value=1.0)
    tk.Label(app.left_panel, text="Скорость", bg="black", fg="white").pack(anchor="w")
    tk.Scale(app.left_panel, from_=0.5, to=2.0, resolution=0.1, orient="horizontal",
             variable=app.speed_factor, bg="black", fg="white").pack(fill="x")

    app.quantile_frame = tk.Frame(app.left_panel, bg="black")
    tk.Label(app.quantile_frame, text="Квантиль", bg="black", fg="white").pack(anchor="w")
    app.quantile = tk.DoubleVar(value=0.97)
    tk.Scale(app.quantile_frame, from_=0.5, to=0.99, resolution=0.01, orient="horizontal",
             variable=app.quantile, bg="black", fg="white").pack(fill="x")
    app.quantile_frame.pack(fill="x")

    app.merge_threshold = tk.DoubleVar(value=1.0)
    tk.Label(app.left_panel, text="Слияние (сек)", bg="black", fg="white").pack(anchor="w")
    tk.Scale(app.left_panel, from_=0.1, to=3.0, resolution=0.1, orient="horizontal",
             variable=app.merge_threshold, bg="black", fg="white").pack(fill="x")

    app.smooth_window = tk.IntVar(value=5)
    tk.Label(app.left_panel, text="Сглаживание", bg="black", fg="white").pack(anchor="w")
    tk.Scale(app.left_panel, from_=1, to=21, resolution=2, orient="horizontal",
             variable=app.smooth_window, bg="black", fg="white").pack(fill="x")

    # --- Тип эксперимента ---
    tk.Label(app.left_panel, text="Тип эксперимента", bg="black", fg="white").pack(anchor="w", pady=(10, 0))
    app.experiment_type = tk.StringVar(value="свободный")
    app.experiment_part = tk.StringVar(value="1 ч.")
    tk.OptionMenu(app.left_panel, app.experiment_type, "свободный", "5:6",
                  command=lambda choice: on_experiment_change(app, choice)).pack(fill="x")

    app.experiment_part_frame = tk.Frame(app.left_panel, bg="black")
    tk.Label(app.experiment_part_frame, text="Часть эксперимента", bg="black", fg="white").pack(anchor="w")
    tk.OptionMenu(app.experiment_part_frame, app.experiment_part, "1 ч.", "2 ч.").pack(fill="x")

    # --- Кнопки управления ---
    app.controls_frame = tk.Frame(app.left_panel, bg="black")
    app.controls_frame.pack(side="bottom", pady=10)
    tk.Button(app.controls_frame, text="Загрузить аудиофайл", command=app.load_audio).pack(fill="x", pady=2)
    tk.Button(app.controls_frame, text="Загрузить метки", command=app.load_markers).pack(fill="x", pady=2)
    tk.Button(app.controls_frame, text="▶ Прослушать", command=app.play_audio).pack(fill="x", pady=2)
    tk.Button(app.controls_frame, text="💾 Сохранить", command=app.save_audio).pack(fill="x", pady=2)
    tk.Button(app.controls_frame, text="ОБРАБОТАТЬ", command=app.process_audio, font=("Arial", 12), bg="white").pack(fill="x", pady=10)
    tk.Button(app.controls_frame, text="📊 Анализ речи", command=app.analyze_audio).pack(fill="x", pady=5)
    tk.Button(app.controls_frame, text="📤 Выгрузить отчёт", command=app.export_report).pack(fill="x", pady=5)

    app.series_plot_button = tk.Button(app.controls_frame, text="📈 Отобразить по сериям", command=app.plot_series_segments)

    # --- Первичная инициализация ---
    on_experiment_change(app, app.experiment_type.get())


def on_experiment_change(app, choice):
    if choice == "свободный":
        app.quantile_frame.pack(fill="x")
        app.experiment_part_frame.pack_forget()
        if hasattr(app, 'series_plot_button'):
            app.series_plot_button.pack_forget()
    elif choice == "5:6":
        app.quantile_frame.pack_forget()
        app.experiment_part_frame.pack(fill="x")
        if hasattr(app, 'series_plot_button'):
            app.series_plot_button.pack(fill="x", pady=5)
