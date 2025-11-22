# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class CSVDataset(Dataset):
    def __init__(self, csv_path, has_target=True):
        self.data = pd.read_csv(csv_path)
        self.has_target = has_target
        
        if has_target:
            self.X = self.data.drop('target', axis=1).values.astype(np.float32)
            self.y = self.data['target'].values.astype(np.int64)
        else:
            self.X = self.data.values.astype(np.float32)
            self.y = None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.X[idx])
        if self.has_target:
            y = torch.LongTensor([self.y[idx]])[0]
            return x, y
        else:
            return x


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        train_labeled_csv='train_labeled.csv',
        train_unlabeled_csv='train_unlabeled.csv',
        test_csv='test.csv',
        batch_size=128,
        num_workers=4
    ):
        super().__init__()
        self.data_dir = data_dir
        self.train_labeled_csv = os.path.join(data_dir, train_labeled_csv)
        self.train_unlabeled_csv = os.path.join(data_dir, train_unlabeled_csv)
        self.test_csv = os.path.join(data_dir, test_csv)
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.train_labeled_dataset = None
        self.test_dataset = None
        
        self.input_dim = None
        self.n_classes = None
    
    def setup(self, stage=None):
        self.train_labeled_dataset = CSVDataset(self.train_labeled_csv, has_target=True)
        self.test_dataset = CSVDataset(self.test_csv, has_target=True)
        
        sample_x, _ = self.train_labeled_dataset[0]
        self.input_dim = sample_x.shape[0]
        
        all_labels = self.train_labeled_dataset.y
        self.n_classes = len(np.unique(all_labels))
        
        print(f'Input dimension: {self.input_dim}')
        print(f'Number of classes: {self.n_classes}')
        print(f'Labeled train samples: {len(self.train_labeled_dataset)}')
        print(f'Test samples: {len(self.test_dataset)}')
    
    def train_dataloader(self):
        return DataLoader(
            self.train_labeled_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
