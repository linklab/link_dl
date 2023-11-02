# https://towardsdatascience.com/cryptocurrency-price-prediction-using-deep-learning-70cfca50dd3a
import json
import requests
import matplotlib.pyplot as plt
import pandas as pd

import os
from pathlib import Path

import numpy as np

BASE_PATH = str(Path(__file__).resolve().parent.parent.parent) # BASE_PATH: /Users/yhhan/git/link_dl
import sys
sys.path.append(BASE_PATH)


def get_hist():
  endpoint = 'https://min-api.cryptocompare.com/data/histoday?fsym=BTC&tsym=USD&limit=500'
  res = requests.get(endpoint)
  hist = pd.DataFrame(json.loads(res.content)['Data'])
  columns = hist.columns
  print([column for column in columns])

  hist = hist.drop(columns=['conversionType', 'conversionSymbol'])

  hist = hist.set_index('time')
  hist.index = pd.to_datetime(hist.index, unit='s')
  return hist


def split_df(df, validation_size=50, test_size=1):
  validation_split_row_idx = len(df) - (validation_size + test_size)
  test_split_row_idx = len(df) - test_size

  train_data = df.iloc[:validation_split_row_idx]
  validation_data = df.iloc[validation_split_row_idx:test_split_row_idx]
  test_data = df.iloc[test_split_row_idx:]

  return train_data, validation_data, test_data


def line_plot(line1, line2, line3, label1=None, label2=None, label3=None, title='', lw=2):
  fig, ax = plt.subplots(1, figsize=(13, 7))
  ax.plot(line1, label=label1, linewidth=lw)
  ax.plot(line2, label=label2, linewidth=lw)
  ax.plot(line3, label=label3, linewidth=lw)
  ax.set_ylabel('Bitcoin [USD]', fontsize=14)
  ax.set_title(title, fontsize=16)
  ax.legend(loc='best', fontsize=16)
  plt.show()


def normalise_min_max(df):
  return (df - df.min()) / (df.max() - df.min())


def extract_window_data(df, window_len=5):
  window_data = []
  for idx in range(len(df) - window_len):
    tmp = df[idx: (idx + window_len)].copy().to_numpy()
    tmp = normalise_min_max(tmp)
    window_data.append(tmp)
  return np.array(window_data)


def prepare_data(train_data, validation_data, test_data, target_col, window_len=10):
  X_train = extract_window_data(train_data, window_len)
  X_validation = extract_window_data(validation_data, window_len)
  X_test = extract_window_data(test_data, window_len)

  y_train = train_data[target_col][window_len:]
  y_validation = validation_data[target_col][window_len]
  y_test = test_data[target_col][window_len:]

  print(X_train.shape, X_validation.shape, X_test.shape, "!!!")
  print(y_train.shape, y_validation.shape, y_test.shape, "!!!")
  return X_train, X_validation, X_test, y_train, y_validation, y_test


if __name__ == "__main__":
  target_col = 'close'
  crypto_currency_df = get_hist()
  print(len(crypto_currency_df))

  train_data, validation_data, test_data = split_df(crypto_currency_df, validation_size=50, test_size=1)
  print(len(train_data), len(validation_data), len(test_data))
  line_plot(
    train_data[target_col], validation_data[target_col], test_data[target_col], 'train', 'validation', 'test', title=''
  )

  X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_data(
    train_data=train_data, validation_data=validation_data, test_data=test_data, target_col="close", window_len=10
  )