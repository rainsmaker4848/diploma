from tkinter import filedialog, messagebox

def load_markers_from_file(app):
    path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    if not path:
        return None, None, None  # ⛔️ ничего не выбрано

    markers = []
    display_labels = []

    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            for idx, line in enumerate(f):
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    start = float(parts[0])
                    end = float(parts[1])
                    label = parts[3].strip()
                    markers.append((start, end))
                    if idx % 2 == 0:
                        display_labels.append((start, label))

        app.filepath = path
        app.markers = markers
        app.display_markers = display_labels
        messagebox.showinfo("Метки загружены", f"Всего: {len(markers)} меток")
        return path, markers, display_labels

    except Exception as e:
        messagebox.showerror("Ошибка чтения файла", str(e))
        return None, None, None
