import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import pytorch_lightning as pl


class WineQualityDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        
        if isinstance(y, pd.Series):
            y = y.values
        
        if isinstance(y, np.ndarray):
            self.y = torch.LongTensor(y)
        elif isinstance(y, torch.Tensor):
            self.y = y
        else:
            self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class WineQualityDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64, val_split=0.2, random_state=42):
        super().__init__()
        self.batch_size = batch_size
        self.val_split = val_split
        self.random_state = random_state
        self.preprocessor = None
        
    def prepare_data(self):
        fetch_openml('wine-quality-red', version=1, as_frame=True, parser='pandas')
    
    def setup(self, stage=None):
        wine = fetch_openml('wine-quality-red', version=1, as_frame=True, parser='pandas')
        X = wine.data
        y = wine.target.astype(int)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.val_split, random_state=self.random_state, stratify=y
        )
        
        y_train_arr = y_train.values if isinstance(y_train, pd.Series) else y_train
        y_val_arr = y_val.values if isinstance(y_val, pd.Series) else y_val
        
        unique_classes = np.unique(np.concatenate([y_train_arr, y_val_arr]))
        class_mapping = {old_class: new_class for new_class, old_class in enumerate(unique_classes)}
        y_train = np.array([class_mapping[c] for c in y_train_arr])
        y_val = np.array([class_mapping[c] for c in y_val_arr])
        
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        self.preprocessor = ColumnTransformer(
            [('num', StandardScaler(), numeric_cols)],
            remainder='drop'
        )
        self.preprocessor.fit(X_train)
        
        X_train_proc = self.preprocessor.transform(X_train)
        X_val_proc = self.preprocessor.transform(X_val)
        
        if stage == 'fit' or stage is None:
            self.train_dataset = WineQualityDataset(X_train_proc, y_train)
            self.val_dataset = WineQualityDataset(X_val_proc, y_val)
        
        self.input_dim = X_train_proc.shape[1]
        self.n_classes = len(unique_classes)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=0
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=0
        )
