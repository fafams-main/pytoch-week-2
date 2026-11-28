import pandas as pd
import numpy as np
import torch
import os
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader
from .data import load_data , data_encode, split_data, impute_and_standardize, preprocess_data

torch.manual_seed(42)

x, y = load_data()
y = data_encode(y)
x_train, y_train, x_val, y_val = split_data(x, y)
x_train, x_val, impute, standardize = impute_and_standardize(x_train, x_val)
x_train, x_val, y_train, y_val = preprocess_data(x_train, x_val, y_train, y_val)

class APSDataset(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def create_datasets(x_train, y_train, x_val, y_val):
    train_data = APSDataset(x_train, y_train)
    val_data = APSDataset(x_val, y_val)
    return train_data, val_data

train_data, val_data = create_datasets(x_train, y_train, x_val, y_val)

def create_dataloaders(train_data, val_data, batch_size=64):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

train_loader, val_loader = create_dataloaders(train_data, val_data)

x_batch, y_batch = next(iter(train_loader))



print(x_batch.shape)
print(y_batch.shape)
print(x_batch.dtype)
print(y_batch.dtype)