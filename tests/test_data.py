import numpy as np
import pytest
import torch

from model.data import load_data, data_encode, split_data, impute_and_standardize, preprocess_data

x, y = load_data()
y = data_encode(y)
x_train, y_train, x_val, y_val = split_data(x, y)
x_train, x_val, impute, standardize = impute_and_standardize(x_train, x_val)
x_train, x_val, y_train, y_val = preprocess_data(x_train, x_val, y_train, y_val)

def test_data_shape():
    assert x.shape == (60000, 170)
    assert y.shape == (60000,)
    assert x_train.shape == (45000, 170)
    assert y_train.shape == (45000,)
    assert x_val.shape == (15000, 170)
    assert y_val.shape == (15000,)

def test_data_types():
    assert (x_train.dtype == torch.float32)
    assert (y_train.dtype == torch.long)
    assert (x_val.dtype == torch.float32)
    assert (y_val.dtype == torch.long)

