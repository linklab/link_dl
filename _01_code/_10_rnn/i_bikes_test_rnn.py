import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
import os
from pathlib import Path

BASE_PATH = str(Path(__file__).resolve().parent.parent.parent) # BASE_PATH: /Users/yhhan/git/link_dl
import sys
sys.path.append(BASE_PATH)

CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_FILE_PATH = os.path.join(CURRENT_FILE_PATH, "checkpoints")
if not os.path.isdir(CHECKPOINT_FILE_PATH):
  os.makedirs(os.path.join(CURRENT_FILE_PATH, "checkpoints"))

from _01_code._03_real_world_data_to_tensors.o_hourly_bikes_sharing_dataset_dataloader import get_hourly_bikes_data, HourlyBikesDataset
from _01_code._10_rnn.h_bikes_train_rnn import get_model


def test_main(test_model):
  _, _, X_test, _, _, y_test = get_hourly_bikes_data(
    sequence_size=24, validation_size=96, test_size=24, y_normalizer=100
  )

  test_hourly_bikes_dataset = HourlyBikesDataset(X=X_test, y=y_test)

  test_data_loader = DataLoader(
    dataset=test_hourly_bikes_dataset, batch_size=len(test_hourly_bikes_dataset)
  )

  test_model.eval()

  y_normalizer = 100

  print("[TEST DATA]")
  with torch.no_grad():
    for test_batch in test_data_loader:
      input_test, target_test = test_batch

      output_test = test_model(input_test)

    for idx, (output, target) in enumerate(zip(output_test, target_test)):
      output = round(output.item() * y_normalizer)
      target = target.item() * y_normalizer

      print("{0:2}: {1:6,.2f} <--> {2:6,.2f} (Loss: {3:>13,.2f})".format(
        idx, output, target, abs(output - target)
      ))


def predict_all(test_model):
  y_normalizer = 100

  X_train, X_validation, X_test, y_train, y_validation, y_test = get_hourly_bikes_data(
      sequence_size=24, validation_size=96, test_size=24, y_normalizer=100
  )

  train_hourly_bikes_dataset = HourlyBikesDataset(X=X_train, y=y_train)
  validation_hourly_bikes_dataset = HourlyBikesDataset(X=X_validation, y=y_validation)
  test_hourly_bikes_dataset = HourlyBikesDataset(X=X_test, y=y_test)

  dataset_list = [
    train_hourly_bikes_dataset, validation_hourly_bikes_dataset, test_hourly_bikes_dataset
  ]
  dataset_labels = [
    "train", "validation", "test"
  ]
  num = 0
  fig, axs = plt.subplots(3, 1, figsize=(6, 9))

  for i in range(3):
    X = []
    TARGET_Y = []
    PREDICTION_Y = []
    for data in dataset_list[i]:
      input, target = data
      prediction = test_model(input.unsqueeze(0)).squeeze(-1).squeeze(-1)

      X.append(num)
      TARGET_Y.append(target.item() * y_normalizer)
      PREDICTION_Y.append(round(prediction.item() * y_normalizer))

      num += 1

    axs[i].plot(X, TARGET_Y, label='target')
    axs[i].plot(X, PREDICTION_Y, label='prediction')
    axs[i].set_title(dataset_labels[i])
    axs[i].legend()

  plt.tight_layout()
  plt.show()


if __name__ == "__main__":
  test_model = get_model()

  project_name = "rnn_bikes"
  latest_file_path = os.path.join(
    CHECKPOINT_FILE_PATH, f"{project_name}_checkpoint_latest.pt"
  )
  print("MODEL FILE: {0}".format(latest_file_path))
  test_model.load_state_dict(torch.load(latest_file_path, map_location=torch.device('cpu')))

  test_main(test_model)
  predict_all(test_model)
