# Домашнее задание: PyTorch Lightning

Заполните ноутбук `pytorch_lightning_classification.ipynb` так, чтобы он работал.

Необходимо реализовать методы, помеченные `# TODO`:

1. **WineDataModule** - класс для управления данными
2. **NeuralNetLightning** - модель на PyTorch Lightning с методами:
   - `__init__` - инициализация модели и параметров
   - `forward` - прямой проход
   - `training_step` - шаг обучения
   - `validation_step` - шаг валидации
   - `test_step` - шаг тестирования
   - `configure_optimizers` - настройка оптимизатора

Ориентируйтесь на реализацию из семинара (`seminar/pytorch_classification.ipynb`).

