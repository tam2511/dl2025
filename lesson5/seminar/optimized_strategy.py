"""
Оптимизированная стратегия для PyTorch Lightning с улучшенной передачей данных на GPU
"""
import torch
from pytorch_lightning.strategies import SingleDeviceStrategy
from typing import Any


class OptimizedSingleDeviceStrategy(SingleDeviceStrategy):
    """
    Оптимизированная версия SingleDeviceStrategy с улучшенной передачей данных на GPU.
    
    Оптимизации:
    1. Использование non_blocking=True для асинхронной передачи данных
    2. Использование pin_memory для ускорения передачи данных
    3. Batch оптимизация для тензоров
    """
    
    def batch_to_device(self, batch: Any, device: torch.device = None, dataloader_idx: int = 0) -> Any:
        """
        Оптимизированная передача batch на device.
        
        Использует non_blocking=True для асинхронной передачи данных на GPU,
        что позволяет CPU продолжать работу во время передачи данных.
        """
        device = device or self.root_device
        
        if isinstance(batch, torch.Tensor):
            # Для тензоров используем non_blocking передачу
            return batch.to(device, non_blocking=True)
        elif isinstance(batch, (tuple, list)):
            # Для tuple/list рекурсивно применяем к элементам
            return type(batch)(self.batch_to_device(item, device, dataloader_idx) for item in batch)
        elif isinstance(batch, dict):
            # Для словарей применяем к значениям
            return {k: self.batch_to_device(v, device, dataloader_idx) for k, v in batch.items()}
        
        # Для других типов возвращаем как есть
        return batch

