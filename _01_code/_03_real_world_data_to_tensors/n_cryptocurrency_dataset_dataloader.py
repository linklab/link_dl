# https://towardsdatascience.com/cryptocurrency-price-prediction-using-deep-learning-70cfca50dd3a
import json
import requests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import seaborn as sns
from sklearn.metrics import mean_absolute_error
from datetime import datetime

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from _01_code._03_real_world_data_to_tensors.h_cryptocurrency_data import get_hist, split_df, prepare_data

BASE_PATH = str(Path(__file__).resolve().parent.parent.parent) # BASE_PATH: /Users/yhhan/git/link_dl
import sys
sys.path.append(BASE_PATH)


class CryptoCurrencyDataset(Dataset):
  def __init__(self, X_dataset, y_dataset):
    self.X_dataset = X_dataset
    self.y_dataset = y_dataset

    assert len(self.X_dataset) == len(self.y_dataset)

  def __len__(self):
    return len(self.X_dataset)

  def __getitem__(self, idx):
    X = self.X_dataset[idx]
    y = self.y_dataset[idx]
    return {'input': X, 'target': y}

  def __str__(self):
    str = "Data Size: {0}, Input Shape: {1}, Target Shape: {2}".format(
      len(self.X_dataset), self.X_dataset.shape, self.y_dataset.shape
    )
    return str


if __name__ == "__main__":
  target_col = 'close'
  crypto_currency_df = get_hist()

  train_data, validation_data, test_data = split_df(crypto_currency_df, validation_size=50, test_size=1)

  X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_data(
    train_data=train_data, validation_data=validation_data, test_data=test_data, target_col="close", window_len=10
  )

  print(X_train.shape, y_train.shape, "!!!!!!!")
  print(X_validation.shape, y_validation.shape, "!!!!!!!")
  print(X_test.shape, y_test.shape, "!!!!!!!")

  crypto_currency_train_dataset = CryptoCurrencyDataset(X_train, y_train)
  print(crypto_currency_train_dataset)

  crypto_currency_validation_dataset = CryptoCurrencyDataset(X_validation, y_validation)
  print(crypto_currency_validation_dataset)

  crypto_currency_test_dataset = CryptoCurrencyDataset(X_test, y_test)
  print(crypto_currency_test_dataset)
