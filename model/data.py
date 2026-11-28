import pandas as pd
import numpy as np
import torch
import os
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

torch.manual_seed(42)


def load_data():

    train_data = pd.read_csv("data/aps_failure_training_set.csv", skiprows=20,na_values = ["na"])


    x = train_data.drop(columns=['class'])
    y = train_data['class']

    print(f"Shape of x: {x.shape}")
    print(f"Shape of y: {y.shape}")

    return x,y
x, y = load_data()

def data_encode(y):
    y = y.map({'neg': 0, 'pos': 1})
    return y
y = data_encode(y)

def split_data(x, y):
    test_size = 0.25
    random_state = 42

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=y)
    

    return x_train, y_train, x_val, y_val

x_train, y_train, x_val, y_val = split_data(x, y)



def impute_and_standardize(x_train, x_val):
    impute = SimpleImputer(strategy='median')
    standardize = StandardScaler()

    x_train = impute.fit_transform(x_train)
    x_val = impute.transform(x_val)

    x_train = standardize.fit_transform(x_train)
    x_val = standardize.transform(x_val)

    print(np.isnan(x_train).sum())
    print(np.isnan(x_val).sum())

    return x_train, x_val , impute, standardize

x_train, x_val, impute, standardize = impute_and_standardize(x_train, x_val)

def preprocess_data(x_train, x_val, y_train, y_val):
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_val = torch.tensor(x_val, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.long)
    y_val = torch.tensor(y_val.values, dtype=torch.long)

    return x_train, x_val, y_train, y_val

def load_test_data():
    test_data = pd.read_csv("data/aps_failure_test_set.csv", skiprows=20,na_values = ["na"])

    x_test = test_data.drop(columns=['class'])
    y_test = test_data['class']

    return x_test, y_test

x_test, y_test = load_test_data()

def data_encode_test(y_test):
    y_test = y_test.map({'neg': 0, 'pos': 1})
    return y_test

y_test = data_encode_test(y_test)


def preprocess_test_data(x_test, y_test, impute, standardize):
    x_test = impute.transform(x_test)
    x_test = standardize.transform(x_test)
    y_test = torch.tensor(y_test.values, dtype=torch.long)

    return x_test, y_test
x_test, y_test = preprocess_test_data(x_test, y_test, impute, standardize)


print(x_test.dtype)
print(x_val.dtype)
print(y_train.dtype)
print(y_val.dtype)
print(x_train.shape)
print(x_val.shape)
print(y_train.shape)
print(y_val.shape)


if __name__ == "__main__":
    x, y = load_data()
    y = data_encode(y)
    x_train, y_train, x_val, y_val = split_data(x, y)
    x_train, x_val, impute, standardize = impute_and_standardize(x_train, x_val)
    x_train, x_val, y_train, y_val = preprocess_data(x_train, x_val, y_train, y_val)
    x_test, y_test = load_test_data()
    y_test = data_encode_test(y_test)
    x_test, y_test = preprocess_test_data(x_test, y_test, impute, standardize)



