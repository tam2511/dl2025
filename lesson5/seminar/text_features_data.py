"""
Text Features Dataset Module

Загружает большой текстовый датасет (AG News - 120k текстов)
и строит векторы фичей большой размерности для демонстрации
техник оптимизации обучения на классическом табличном представлении.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import pytorch_lightning as pl
from datasets import load_dataset
from pathlib import Path
import pickle
from tqdm import tqdm


class TextFeaturesDataset(Dataset):
    """
    Dataset для работы с предобработанными текстовыми фичами.
    
    Использует memory-mapped файлы для экономии RAM.
    Данные хранятся на диске и загружаются по требованию.
    
    Args:
        X_path: путь к .npy файлу с фичами
        y_path: путь к .npy файлу с метками
    """
    
    def __init__(self, X_path, y_path):
        # Загружаем данные в mmap режиме
        self.X = np.load(X_path, mmap_mode='r')  # read-only mmap
        self.y = np.load(y_path, mmap_mode='r')
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # Конвертируем в torch тензор на лету
        x = torch.FloatTensor(self.X[idx])
        y = torch.LongTensor([self.y[idx]])[0]
        return x, y


class TextFeaturesDataModule(pl.LightningDataModule):
    """
    DataModule для работы с текстовым датасетом AG News.
    
    Датасет: AG News (120k новостных текстов, 4 класса)
    - Business
    - Sci/Tech
    - Sports
    - World
    
    Фичи: TF-IDF векторы большой размерности + статистические фичи
    - TF-IDF uni-grams и bi-grams (настраиваемый размер словаря)
    - Длина текста, количество слов, средняя длина слова и т.д.
    
    Args:
        max_features: максимальный размер словаря для TF-IDF (по умолчанию 20000)
        batch_size: размер батча
        val_split: доля валидационной выборки
        use_bigrams: использовать ли биграммы
        cache_dir: директория для кеширования предобработанных данных
        random_state: random seed
    """
    
    def __init__(
        self, 
        max_features: int = 20000,
        batch_size: int = 128,
        val_split: float = 0.15,
        use_bigrams: bool = True,
        cache_dir: str = "./cache",
        random_state: int = 42,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.max_features = max_features
        self.batch_size = batch_size
        self.val_split = val_split
        self.use_bigrams = use_bigrams
        self.cache_dir = Path(cache_dir)
        self.random_state = random_state
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        self.vectorizer = None
        self.scaler = None
        self.input_dim = None
        self.n_classes = 4  # AG News has 4 classes
        
        # Создаем директорию для кеша
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_npy_paths(self, split: str):
        """Пути к .npy файлам для mmap режима"""
        ngram = "bigrams" if self.use_bigrams else "unigrams"
        base_name = f"ag_news_{split}_{ngram}_{self.max_features}"
        paths = {
            'X': self.cache_dir / f"{base_name}_X.npy",
            'y': self.cache_dir / f"{base_name}_y.npy",
        }
        # Метаданные хранятся только для train
        if split == 'train':
            paths['meta'] = self.cache_dir / f"{base_name}_meta.pkl"
        return paths
    
    def _extract_statistical_features(self, texts):
        """Извлекает статистические фичи из текстов"""
        features = []
        for text in tqdm(texts, desc="Statistical features"):
            words = text.split()
            features.append([
                len(text),                          # длина текста в символах
                len(words),                         # количество слов
                np.mean([len(w) for w in words]) if words else 0,   # средняя длина слова
                np.std([len(w) for w in words]) if words else 0,    # std длины слова
                len(set(words)) / len(words) if words else 0,       # отношение уникальных слов
            ])
        return np.array(features)
    
    def prepare_data(self):
        """Загружает датасет AG News"""
        # Загружаем датасет из HuggingFace
        load_dataset("ag_news")
    
    def setup(self, stage=None):
        """Подготавливает данные для обучения"""
        
        # Проверяем наличие кеша
        npy_train = self._get_npy_paths("train")
        npy_val = self._get_npy_paths("val")
        
        # Проверяем что все нужные файлы существуют
        cache_exists = (
            npy_train['X'].exists() and 
            npy_train['y'].exists() and 
            npy_train['meta'].exists() and
            npy_val['X'].exists() and 
            npy_val['y'].exists()
        )
        
        if cache_exists:
            print("Loading from cache...")
            
            # Загружаем метаданные
            with open(npy_train['meta'], 'rb') as f:
                meta = pickle.load(f)
                self.vectorizer = meta['vectorizer']
                self.scaler = meta['scaler']
                self.input_dim = meta['input_dim']
            
            # Создаем датасеты
            if stage == 'fit' or stage is None:
                self.train_dataset = TextFeaturesDataset(str(npy_train['X']), str(npy_train['y']))
                self.val_dataset = TextFeaturesDataset(str(npy_val['X']), str(npy_val['y']))
            
            print(f"Dataset loaded: classes={self.n_classes}, features={self.input_dim}, "
                  f"train={len(self.train_dataset)}, val={len(self.val_dataset)}")
            
            return
        
        # Если кеша нет - обрабатываем данные
        print("Preparing AG News dataset...")
        
        # Загружаем датасет
        dataset = load_dataset("ag_news")
        train_data = dataset['train']
        test_data = dataset['test']
        
        # Объединяем train и test для создания нового split
        # Конвертируем в списки для объединения
        all_texts = list(tqdm(train_data['text'], desc="Loading train")) + \
                    list(tqdm(test_data['text'], desc="Loading test"))
        all_labels = list(train_data['label']) + list(test_data['label'])
        
        print(f"Total texts: {len(all_texts)}")
        
        # Разбиваем на train и val
        texts_train, texts_val, y_train, y_val = train_test_split(
            all_texts, all_labels,
            test_size=self.val_split,
            random_state=self.random_state,
            stratify=all_labels
        )
        print(f"Split: train={len(texts_train)}, val={len(texts_val)}")
        
        # Создаем TF-IDF векторизатор
        ngram_range = (1, 2) if self.use_bigrams else (1, 1)
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            strip_accents='unicode',
            lowercase=True,
            stop_words='english'
        )
        
        # Извлекаем TF-IDF фичи
        print(f"Extracting TF-IDF features (max_features={self.max_features}, ngrams={ngram_range})...")
        tfidf_train = self.vectorizer.fit_transform(tqdm(texts_train, desc="TF-IDF train")).toarray()
        tfidf_val = self.vectorizer.transform(tqdm(texts_val, desc="TF-IDF val")).toarray()
        print(f"TF-IDF dimension: {tfidf_train.shape[1]}")
        
        # Извлекаем статистические фичи
        print("Extracting statistical features...")
        stat_train = self._extract_statistical_features(texts_train)
        stat_val = self._extract_statistical_features(texts_val)
        
        # Объединяем все фичи
        print(f"Concatenating train features ({tfidf_train.shape[0]}x{tfidf_train.shape[1]} + {stat_train.shape[1]})...")
        X_train = np.concatenate([tfidf_train, stat_train], axis=1)
        print(f"Concatenating val features ({tfidf_val.shape[0]}x{tfidf_val.shape[1]} + {stat_val.shape[1]})...")
        X_val = np.concatenate([tfidf_val, stat_val], axis=1)
        print(f"Total feature dimension: {X_train.shape[1]}")
        
        # Нормализуем только статистические фичи (TF-IDF уже нормализован)
        print("Normalizing statistical features...")
        self.scaler = StandardScaler()
        X_train[:, -stat_train.shape[1]:] = self.scaler.fit_transform(
            X_train[:, -stat_train.shape[1]:]
        )
        X_val[:, -stat_val.shape[1]:] = self.scaler.transform(
            X_val[:, -stat_val.shape[1]:]
        )
        print("Normalization done")
        
        y_train = np.array(y_train)
        y_val = np.array(y_val)
        
        # Сохраняем в кеш
        print("Saving to cache...")
        npy_train = self._get_npy_paths("train")
        npy_val = self._get_npy_paths("val")
        
        print(f"Saving train X ({X_train.shape[0]}x{X_train.shape[1]}, {X_train.nbytes / 1024**2:.1f} MB)...")
        np.save(npy_train['X'], X_train)
        print(f"Saving train y...")
        np.save(npy_train['y'], y_train)
        print(f"Saving val X ({X_val.shape[0]}x{X_val.shape[1]}, {X_val.nbytes / 1024**2:.1f} MB)...")
        np.save(npy_val['X'], X_val)
        print(f"Saving val y...")
        np.save(npy_val['y'], y_val)
        
        # Сохраняем метаданные
        with open(npy_train['meta'], 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'scaler': self.scaler,
                'input_dim': X_train.shape[1],
            }, f)
        
        print(f"Saved to {self.cache_dir}")
        
        # Создаем PyTorch датасеты
        if stage == 'fit' or stage is None:
            npy_train = self._get_npy_paths("train")
            npy_val = self._get_npy_paths("val")
            self.train_dataset = TextFeaturesDataset(str(npy_train['X']), str(npy_train['y']))
            self.val_dataset = TextFeaturesDataset(str(npy_val['X']), str(npy_val['y']))
        
        self.input_dim = X_train.shape[1]
        
        print(f"Dataset ready: classes={self.n_classes}, features={self.input_dim}, "
              f"train={len(X_train)}, val={len(X_val)}")
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=self.pin_memory,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=self.pin_memory,
        )


def get_example_datamodule(max_features: int = 20000, batch_size: int = 128):
    """
    Вспомогательная функция для быстрого создания DataModule
    
    Args:
        max_features: размер словаря TF-IDF
        batch_size: размер батча
    
    Returns:
        TextFeaturesDataModule
    """
    return TextFeaturesDataModule(
        max_features=max_features,
        batch_size=batch_size,
        use_bigrams=True,
    )
