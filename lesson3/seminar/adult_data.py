import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pytorch_lightning as pl


class AdultDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        
        if isinstance(y, pd.Series):
            y = y.values
        
        if isinstance(y, np.ndarray):
            self.y = torch.FloatTensor(y.astype(np.float32))
        elif isinstance(y, torch.Tensor):
            self.y = y.float()
        else:
            self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class AdultDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64, val_split=0.2, random_state=42):
        super().__init__()
        self.batch_size = batch_size
        self.val_split = val_split
        self.random_state = random_state
        self.preprocessor = None
        
    def prepare_data(self):
        fetch_openml('adult', version=1, as_frame=True, parser='pandas')
    
    def setup(self, stage=None):
        adult = fetch_openml('adult', version=1, as_frame=True, parser='pandas')
        X = adult.data
        y = (adult.target == '>50K').astype(int)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.val_split, random_state=self.random_state, stratify=y
        )
        
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        
        transformers = []
        if numeric_cols:
            transformers.append(('num', StandardScaler(), numeric_cols))
        if categorical_cols:
            transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols))
        
        self.preprocessor = ColumnTransformer(transformers, remainder='drop')
        self.preprocessor.fit(X_train)
        
        X_train_proc = self.preprocessor.transform(X_train)
        X_val_proc = self.preprocessor.transform(X_val)
        
        if stage == 'fit' or stage is None:
            self.train_dataset = AdultDataset(X_train_proc, y_train)
            self.val_dataset = AdultDataset(X_val_proc, y_val)
        
        self.input_dim = X_train_proc.shape[1]
        self.n_classes = 2
        
        n_pos = (y_train == 1).sum()
        n_neg = (y_train == 0).sum()
        self.pos_weight = torch.tensor([n_neg / n_pos]) if n_pos > 0 else torch.tensor([1.0])
    
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
