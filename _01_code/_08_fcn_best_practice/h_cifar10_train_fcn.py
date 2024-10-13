import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from datetime import datetime
import os
import wandb
from pathlib import Path

BASE_PATH = str(Path(__file__).resolve().parent.parent.parent) # BASE_PATH: /Users/yhhan/git/link_dl
import sys
sys.path.append(BASE_PATH)

CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_FILE_PATH = os.path.join(CURRENT_FILE_PATH, "checkpoints")
if not os.path.isdir(CHECKPOINT_FILE_PATH):
  os.makedirs(os.path.join(CURRENT_FILE_PATH, "checkpoints"))

from _01_code._99_common_utils.utils import is_linux, is_windows, get_num_cpu_cores
from _01_code._08_fcn_best_practice.c_trainer import ClassificationTrainer
from _01_code._08_fcn_best_practice.e_arg_parser import get_parser


def get_cifar10_data(flatten=False):
  data_path = os.path.join(BASE_PATH, "_00_data", "i_cifar10")

  print("DATA PATH: {0}".format(data_path))

  cifar10_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transforms.ToTensor())

  cifar10_train, cifar10_validation = random_split(cifar10_train, [45_000, 5_000])

  print("Num Train Samples: ", len(cifar10_train))
  print("Num Validation Samples: ", len(cifar10_validation))

  num_data_loading_workers = get_num_cpu_cores() if is_linux() or is_windows() else 0
  print("Number of Data Loading Workers:", num_data_loading_workers)

  train_data_loader = DataLoader(
    dataset=cifar10_train, batch_size=wandb.config.batch_size, shuffle=True,
    pin_memory=True, num_workers=num_data_loading_workers
  )

  validation_data_loader = DataLoader(
    dataset=cifar10_validation, batch_size=wandb.config.batch_size,
    pin_memory=True, num_workers=num_data_loading_workers
  )

  cifar10_transforms = nn.Sequential(
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=(0.4915, 0.4823, 0.4468), std=(0.2470, 0.2435, 0.2616)),
  )

  if flatten:
    cifar10_transforms.append(
      nn.Flatten()
    )

  return train_data_loader, validation_data_loader, cifar10_transforms


def get_model():
  class MyModel(nn.Module):
    def __init__(self, n_input, n_output):
      super().__init__()

      self.model = nn.Sequential(
        nn.Linear(n_input, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, n_output),
      )

    def forward(self, x):
      x = self.model(x)
      return x

  # 3 * 32 * 32 = 3072
  my_model = MyModel(n_input=3_072, n_output=10)

  return my_model


def main(args):
  run_time_str = datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S')

  config = {
    'epochs': args.epochs,
    'batch_size': args.batch_size,
    'validation_intervals': args.validation_intervals,
    'learning_rate': args.learning_rate,
    'early_stop_patience': args.early_stop_patience,
    'early_stop_delta': args.early_stop_delta
  }

  project_name = "fcn_cifar10"
  wandb.init(
    mode="online" if args.wandb else "disabled",
    project=project_name,
    notes="cifar10 experiment with fcn",
    tags=["fcc", "cifar10"],
    name=run_time_str,
    config=config
  )
  print(args)
  print(wandb.config)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(f"Training on device {device}.")

  train_data_loader, validation_data_loader, cifar10_transforms = get_cifar10_data(flatten=True)
  model = get_model()
  model.to(device)
  wandb.watch(model)

  optimizer = optim.SGD(model.parameters(), lr=wandb.config.learning_rate)

  classification_trainer = ClassificationTrainer(
    project_name, model, optimizer, train_data_loader, validation_data_loader, cifar10_transforms,
    run_time_str, wandb, device, CHECKPOINT_FILE_PATH
  )
  classification_trainer.train_loop()

  wandb.finish()


if __name__ == "__main__":
  parser = get_parser()
  args = parser.parse_args()
  main(args)
  # python _01_code/_06_fcn_best_practice/h_cifar10_train_fcn.py --wandb -b 2048 -r 1e-3 -v 10
  # python _01_code/_06_fcn_best_practice/h_cifar10_train_fcn.py --no-wandb -b 2048 -r 1e-3 -v 10
