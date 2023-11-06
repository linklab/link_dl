import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from datetime import datetime
import os
import wandb
from pathlib import Path

from _01_code._10_rnn.g_bikes_train_and_test_rnn import get_model

BASE_PATH = str(Path(__file__).resolve().parent.parent.parent) # BASE_PATH: /Users/yhhan/git/link_dl
import sys
sys.path.append(BASE_PATH)

CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_FILE_PATH = os.path.join(CURRENT_FILE_PATH, "../_10_rnn/checkpoints")
if not os.path.isdir(CHECKPOINT_FILE_PATH):
  os.makedirs(os.path.join(CURRENT_FILE_PATH, "../_10_rnn/checkpoints"))

from _01_code._03_real_world_data_to_tensors.m_time_series_dataset_dataloader import BikesDataset

def get_trained_model():
  model = get_model()

  project_name = "rnn_bikes"
  latest_file_path = os.path.join(
    CHECKPOINT_FILE_PATH, f"{project_name}_checkpoint_latest.pt"
  )
  print("MODEL FILE: {0}".format(latest_file_path))
  model.load_state_dict(torch.load(latest_file_path, map_location=torch.device('cpu')))

  model.eval()

  return model


def predict_all(model):
  bikes_dataset = BikesDataset()
  print(bikes_dataset)

  with torch.no_grad():
    for idx in range(3):
      for bikes_data in bikes_dataset:
        input, target = bikes_data
        output = model(input)
        print(output.shape, target.shape)


if __name__ == "__main__":
  model = get_trained_model()
  predict_all(model)