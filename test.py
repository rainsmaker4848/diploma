import subprocess
import sys
import shutil
import os

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
        "soundfile",
        "tk",           # для GUI
        "transformers",
        "torchaudio",
        "pandas",
        "torch"
    ]

    for package in required_packages:
        try:
            print(f"[..] Устанавливается: {package}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"[OK] Установлено: {package}")
        except subprocess.CalledProcessError:
            print(f"[!!] Не удалось установить: {package}")

def check_ffmpeg():
    print("[..] Проверка наличия ffmpeg...")
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        print(f"[OK] ffmpeg найден: {ffmpeg_path}")
        return True
    else:
        print("[!!] ffmpeg НЕ найден в PATH.")
        return False

def add_ffmpeg_to_path():
    target_path = r"B:\ffmpeg-7.1.1-full_build\bin"
    print(f"[..] Добавляется путь в PATH: {target_path}")
    try:
        # Получаем текущий PATH
        current_path = os.environ.get("PATH", "")
        if target_path.lower() in current_path.lower():
            print("[OK] Путь уже добавлен в PATH.")
            return

        # Добавление в переменные среды пользователя
        import winreg
        reg_key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                                 "Environment", 0, winreg.KEY_ALL_ACCESS)
        try:
            existing_path, _ = winreg.QueryValueEx(reg_key, "Path")
        except FileNotFoundError:
            existing_path = ""
        if target_path not in existing_path:
            new_path = existing_path + ";" + target_path if existing_path else target_path
            winreg.SetValueEx(reg_key, "Path", 0, winreg.REG_EXPAND_SZ, new_path)
            print("[OK] Путь добавлен в PATH. Перезапусти PowerShell или перезайди в систему.")
        else:
            print("[OK] Путь уже был в PATH.")
        winreg.CloseKey(reg_key)
    except Exception as e:
        print(f"[!!] Не удалось изменить PATH автоматически: {e}")
        print(">>> Добавь вручную:", target_path)

if __name__ == "__main__":
    upgrade_pip()
    install_packages()
    if not check_ffmpeg():
        add_ffmpeg_to_path()
