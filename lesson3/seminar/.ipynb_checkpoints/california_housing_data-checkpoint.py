import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import pytorch_lightning as pl


class CaliforniaHousingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CaliforniaHousingDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64, val_split=0.2, random_state=42):
        super().__init__()
        self.batch_size = batch_size
        self.val_split = val_split
        self.random_state = random_state
        self.preprocessor = None
        
    def prepare_data(self):
        fetch_california_housing()
    
    def setup(self, stage=None):
        california_housing = fetch_california_housing()
        X = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
        y = california_housing.target
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.val_split, random_state=self.random_state
        )
        
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        self.preprocessor = ColumnTransformer(
            [('num', StandardScaler(), numeric_cols)],
            remainder='drop'
        )
        self.preprocessor.fit(X_train)
        
        X_train_proc = self.preprocessor.transform(X_train)
        X_val_proc = self.preprocessor.transform(X_val)
        
        if stage == 'fit' or stage is None:
            self.train_dataset = CaliforniaHousingDataset(X_train_proc, y_train)
            self.val_dataset = CaliforniaHousingDataset(X_val_proc, y_val)
        
        self.input_dim = X_train_proc.shape[1]
    
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
