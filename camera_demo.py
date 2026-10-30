"""
Распознавание жестов с веб-камеры в реальном времени.

Запуск:  python camera_demo.py
Выход:   клавиша Q
"""

import cv2
import torch
import numpy as np
from torchvision import transforms

from gesture_net import GestureNet, IMG_SIZE, DEVICE

# ============================================================
# НАСТРОЙКИ
# ============================================================

MODEL_PATH = "gesture_net_best.pt"

# ВАЖНО: порядок классов должен совпадать с порядком папок в датасете.
# ImageFolder сортирует папки по алфавиту, поэтому список тоже
# должен быть отсортирован по алфавиту.
# Обучение (python gesture_net.py) печатает правильный порядок — скопируй оттуда.
CLASS_NAMES = ["catch", "down", "release", "stop", "up"]

# Порог уверенности: ниже него выводим "не распознано"
CONFIDENCE_THRESHOLD = 0.6

# Размер квадратной области захвата в центре кадра (пиксели)
ROI_SIZE = 300


# ============================================================
# ЗАГРУЗКА МОДЕЛИ
# ============================================================

def load_model():
    model = GestureNet(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()  # обязательно: переключает BatchNorm в режим инференса
    print(f"Модель загружена из {MODEL_PATH}, устройство: {DEVICE}")
    return model


# Преобразования — ТОЧНО такие же, как val_transform при обучении,
# без аугментации (никаких поворотов и зума)
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


def predict(model, roi_bgr):
    """Принимает вырезанную область кадра (BGR), возвращает (класс, уверенность)."""
    # OpenCV даёт BGR, а модель обучалась на RGB — конвертируем
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)

    tensor = preprocess(roi_rgb).unsqueeze(0).to(DEVICE)  # добавляем batch-размерность

    with torch.no_grad():
        probs = model.predict_proba(tensor)[0]

    idx = probs.argmax().item()
    return CLASS_NAMES[idx], probs[idx].item()


# ============================================================
# ОСНОВНОЙ ЦИКЛ
# ============================================================

def main():
    model = load_model()

    cap = cv2.VideoCapture(0)  # 0 — встроенная камера; попробуй 1, если не открывается
    if not cap.isOpened():
        print("Не удалось открыть камеру. Проверь, не занята ли она другой программой.")
        return

    print("Камера запущена. Показывай жест в зелёном квадрате. Выход — клавиша Q.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Не удалось получить кадр с камеры.")
            break

        frame = cv2.flip(frame, 1)  # зеркалим, чтобы было как в зеркале
        h, w = frame.shape[:2]

        # Вырезаем квадратную область в центре кадра
        x1 = w // 2 - ROI_SIZE // 2
        y1 = h // 2 - ROI_SIZE // 2
        x2 = x1 + ROI_SIZE
        y2 = y1 + ROI_SIZE
        roi = frame[y1:y2, x1:x2]

        # Предсказание
        label, confidence = predict(model, roi)

        # Рисуем рамку области захвата
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Подпись с результатом
        if confidence >= CONFIDENCE_THRESHOLD:
            text = f"{label}: {confidence * 100:.1f}%"
            color = (0, 255, 0)
        else:
            text = f"unknown ({confidence * 100:.1f}%)"
            color = (0, 165, 255)

        cv2.putText(frame, text, (x1, y1 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Gesture recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Работа завершена.")


if __name__ == "__main__":
    main()
