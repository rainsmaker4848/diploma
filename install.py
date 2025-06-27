import subprocess
import sys

def upgrade_pip():
    try:
        print("[..] Обновляется pip до последней версии...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        print("[OK] pip обновлён.")
    except subprocess.CalledProcessError:
        print("[!!] Не удалось обновить pip.")

def install_packages():
    required_packages = [
        "matplotlib",
        "librosa",
        "numpy",
        "simpleaudio",
        "soundfile",  # для сохранения .wav
        "tk"          # для GUI (обычно встроен в Python, но на Linux иногда надо явно)
    ]

    for package in required_packages:
        try:
            print(f"[..] Устанавливается: {package}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"[OK] Установлено: {package}")
        except subprocess.CalledProcessError:
            print(f"[!!] Не удалось установить: {package}")

if __name__ == "__main__":
    upgrade_pip()
    install_packages()
