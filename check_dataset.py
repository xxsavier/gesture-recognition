"""
Проверка датасета на битые изображения.

Проходит по всем файлам в dataset/, пытается открыть каждый.
Битые файлы выводит в консоль и (по желанию) удаляет.

Запуск:  python check_dataset.py
"""

import os
from PIL import Image

DATA_DIR = "dataset"

# Если True — битые файлы будут удалены автоматически.
# Сначала запусти с False, посмотри список, и только потом включай удаление.
AUTO_DELETE = False


def check_dataset():
    broken_files = []
    total = 0

    for class_name in os.listdir(DATA_DIR):
        class_path = os.path.join(DATA_DIR, class_name)
        if not os.path.isdir(class_path):
            continue

        for filename in os.listdir(class_path):
            filepath = os.path.join(class_path, filename)
            total += 1

            try:
                # verify() проверяет структуру файла, не загружая полностью в память
                img = Image.open(filepath)
                img.verify()

                # Дополнительно пробуем полную загрузку и конвертацию,
                # так как verify() иногда пропускает частично битые файлы
                img2 = Image.open(filepath)
                img2.convert("RGB")

            except Exception as e:
                print(f"БИТЫЙ ФАЙЛ: {filepath}")
                print(f"   Ошибка: {e}")
                broken_files.append(filepath)

    print(f"\nПроверено файлов: {total}")
    print(f"Битых файлов найдено: {len(broken_files)}")

    if broken_files and AUTO_DELETE:
        for f in broken_files:
            os.remove(f)
        print(f"Удалено файлов: {len(broken_files)}")
    elif broken_files:
        print("\nЧтобы удалить их автоматически, поставь AUTO_DELETE = True в начале файла и запусти снова.")
        print("Либо удали вручную из папок файлы, перечисленные выше.")
    else:
        print("Все файлы в порядке, можно запускать обучение.")


if __name__ == "__main__":
    check_dataset()
