"""
Реализация на PyTorch нейросети для распознавания жестов,
воспроизводящая архитектуру и параметры обучения из среды NNTrack.

Архитектура (из схемы NNTrack):
    Вход 64x64x3
    -> [Conv2D(32, k=3, s=1, padding='valid') -> ReLU -> BatchNorm -> MaxPool2D] x3
    -> Flatten -> Dense(5) -> Softmax

Структура датасета (по 500 изображений на класс, всего 2500):
    dataset/
        catch/
        down/
        release/
        stop/
        up/
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# ============================================================
# ПАРАМЕТРЫ (в точности как в NNTrack-отчёте)
# ============================================================

IMG_SIZE = 64          # ширина/высота изображения
NUM_CLASSES = 5        # 5 видов жестов
BATCH_SIZE = 20        # размер партии
EPOCHS = 32            # количество эпох
LEARNING_RATE = 0.001  # скорость обучения (Adam)
VAL_SPLIT = 0.2        # разбиение данных: 80% train / 20% val

# Параметры early stopping (предотвращение переобучения)
ES_MIN_DELTA = 0.05
ES_PATIENCE = 2

DATA_DIR = "dataset"   # путь к папке с датасетом
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# АРХИТЕКТУРА СЕТИ
# ============================================================

class GestureNet(nn.Module):
    """CNN для классификации жестов — точная копия схемы из NNTrack."""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        # Блок 1: вход 3 канала (RGB) -> 32 фильтра
        # padding=0 соответствует padding='valid' в NNTrack
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Блок 2
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Блок 3
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Расчёт размерности после свёрток (для входа 64x64):
        #   64 -> conv(3, valid) -> 62 -> pool(2) -> 31
        #   31 -> conv(3, valid) -> 29 -> pool(2) -> 14
        #   14 -> conv(3, valid) -> 12 -> pool(2) -> 6
        #   итого: 6 * 6 * 32 = 1152 признака
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6 * 6 * 32, num_classes),
        )
        # Softmax НЕ добавляем сюда: nn.CrossEntropyLoss применяет его внутри.
        # Для инференса используем метод predict_proba ниже.

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)

    @torch.no_grad()
    def predict_proba(self, x):
        """Возвращает вероятности классов (аналог блока Softmax в NNTrack)."""
        return torch.softmax(self.forward(x), dim=1)


# ============================================================
# ДАННЫЕ И АУГМЕНТАЦИЯ
# ============================================================

# Аугментация из NNTrack: random rotation = 0.05, random zoom = 0.1
# В Keras rotation=0.05 означает долю от полного оборота: 0.05 * 360 = 18 градусов
# zoom = 0.1 означает масштабирование в диапазоне +-10%
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(degrees=18),
    transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
    transforms.ToTensor(),
])

# Для валидации аугментация не применяется
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


def get_dataloaders(data_dir=DATA_DIR):
    """Загружает датасет из папок и делит на train/val."""
    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)

    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size

    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)

    # Валидационной части назначаем трансформации без аугментации
    val_subset.dataset = datasets.ImageFolder(root=data_dir, transform=val_transform)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Классы: {full_dataset.classes}")
    print(f"Train: {train_size} изображений, Val: {val_size} изображений")

    return train_loader, val_loader, full_dataset.classes


# ============================================================
# ОБУЧЕНИЕ
# ============================================================

def train():
    train_loader, val_loader, class_names = get_dataloaders()

    model = GestureNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss()          # CategoricalCrossentropyLoss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # AdamOptimizer

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        # --- обучение ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

        # --- валидация ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        train_loss /= train_total
        val_loss /= val_total
        train_acc = train_correct / train_total    # CategoricalAccuracyMetric
        val_acc = val_correct / val_total

        print(f"Эпоха {epoch:2d}/{EPOCHS} | "
              f"loss: {train_loss:.4f} acc: {train_acc:.4f} | "
              f"val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}")

        # --- early stopping (monitor=val_loss, min_delta=0.05, patience=2) ---
        if val_loss < best_val_loss - ES_MIN_DELTA:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "gesture_net_best.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= ES_PATIENCE:
                print(f"Early stopping на эпохе {epoch} "
                      f"(val_loss не улучшался {ES_PATIENCE} эпох подряд)")
                break

    print("Обучение завершено. Лучшая модель: gesture_net_best.pt")
    return model, class_names


# ============================================================
# ПРОВЕРКА АРХИТЕКТУРЫ
# ============================================================

def check_architecture():
    """Прогоняет случайный тензор и печатает размерности после каждого блока."""
    model = GestureNet()
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    print(f"Вход:          {tuple(x.shape)}")
    x = model.block1(x); print(f"После блока 1: {tuple(x.shape)}")
    x = model.block2(x); print(f"После блока 2: {tuple(x.shape)}")
    x = model.block3(x); print(f"После блока 3: {tuple(x.shape)}")
    x = model.classifier(x); print(f"Выход:         {tuple(x.shape)}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nВсего параметров: {total_params:,}")


if __name__ == "__main__":
    # check_architecture()   # раскомментировать, чтобы проверить размерности слоёв
    train()
