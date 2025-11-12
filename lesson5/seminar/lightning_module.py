"""
PyTorch Lightning module for text classification
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics


class TextClassificationModule(pl.LightningModule):
    """
    Lightning модуль для классификации текстов
    
    Args:
        model: PyTorch модель
        lr: learning rate
        optimizer_type: тип оптимизатора ('adam', 'adamw', 'sgd')
        weight_decay: weight decay для регуляризации
        num_classes: количество классов
    """
    
    def __init__(
        self,
        model,
        lr=1e-3,
        optimizer_type='adam',
        weight_decay=0.0,
        num_classes=4,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        
        self.model = model
        self.lr = lr
        self.optimizer_type = optimizer_type
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        
        # Loss
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.val_f1 = torchmetrics.F1Score(task='multiclass', num_classes=num_classes, average='macro')
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Metrics
        self.train_acc(logits, y)
        
        # Logging
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Metrics
        self.val_acc(logits, y)
        self.val_f1(logits, y)
        
        # Logging
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        if self.optimizer_type == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")
        
        return optimizer

