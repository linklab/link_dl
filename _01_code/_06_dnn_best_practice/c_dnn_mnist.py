import argparse

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
from torchvision import datasets, transforms
from datetime import datetime
import os
import wandb

from _01_code._06_dnn_best_practice.a_trainer import ClassificationTrainer
from _01_code._99_common_utils.utils import is_linux, get_num_cpu_cores

def get_ready():
  current_path = os.path.dirname(os.path.abspath(__file__))
  if not os.path.isdir(os.path.join(current_path, "checkpoints")):
    os.makedirs(os.path.join(current_path, "checkpoints"))

def get_data_flattened():
  data_path = os.path.join(os.path.pardir, os.path.pardir, "_00_data", "i_mnist")

  transformed_mnist_train = datasets.MNIST(
    data_path, train=True, download=False, transform=transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=0.1307, std=0.3081),
      T.Lambda(lambda x: torch.flatten(x))
    ])
  )

  transformed_mnist_train, transformed_mnist_test = random_split(transformed_mnist_train, [59000, 1000])

  transformed_mnist_validation = datasets.MNIST(
    data_path, train=False, download=False, transform=transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=0.1307, std=0.3081),
      T.Lambda(lambda x: torch.flatten(x))
    ])
  )

  print("Num Train Samples: ", len(transformed_mnist_train))
  print("Num Validation Samples: ", len(transformed_mnist_validation))
  print("Num Test Samples: ", len(transformed_mnist_test))

  num_data_loading_workers = get_num_cpu_cores() if is_linux() else 0
  print("Number of Data Loading Workers:", num_data_loading_workers)

  train_data_loader = DataLoader(
    dataset=transformed_mnist_train, batch_size=wandb.config.batch_size, shuffle=True,
    pin_memory=True, num_workers=num_data_loading_workers
  )
  validation_data_loader = DataLoader(
    dataset=transformed_mnist_validation, batch_size=wandb.config.batch_size,
    pin_memory=True, num_workers=num_data_loading_workers
  )

  return train_data_loader, validation_data_loader


def get_model_and_optimizer():
  class MyModel(nn.Module):
    def __init__(self, n_input, n_output):
      super().__init__()

      self.model = nn.Sequential(
        nn.Linear(n_input, wandb.config.n_hidden_unit_list[0]),
        nn.Sigmoid(),
        nn.Linear(wandb.config.n_hidden_unit_list[0], wandb.config.n_hidden_unit_list[1]),
        nn.Sigmoid(),
        nn.Linear(wandb.config.n_hidden_unit_list[1], n_output),
      )

    def forward(self, x):
      x = self.model(x)
      return x

  # 1 * 28 * 28 = 784
  my_model = MyModel(n_input=784, n_output=10)
  optimizer = optim.SGD(my_model.parameters(), lr=wandb.config.learning_rate)

  return my_model, optimizer


def main(args):
  get_ready()

  run_time_str = datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S')

  config = {
    'epochs': args.epochs,
    'batch_size': args.batch_size,
    'learning_rate': 1e-3,
    'n_hidden_unit_list': [128, 128],
  }

  wandb.init(
    mode="online" if args.use_wandb else "disabled",
    project="dnn_mnist",
    notes="mnist experiment",
    tags=["dnn", "mnist"],
    name=run_time_str,
    config=config
  )
  print(args)
  print(wandb.config)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(f"Training on device {device}.")

  train_data_loader, validation_data_loader = get_data_flattened()
  model, optimizer = get_model_and_optimizer()
  model.to(device)
  wandb.watch(model)

  classification_trainer = ClassificationTrainer(
    "mnist", model, optimizer, train_data_loader, validation_data_loader,
    run_time_str, wandb, device
  )
  classification_trainer.train_loop()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument(
    "-w", "--use_wandb", type=bool, default=False, help="True or False"
  )

  parser.add_argument(
    "-b", "--batch_size", type=int, default=128, help="Batch size (int)"
  )

  parser.add_argument(
    "-e", "--epochs", type=int, default=10_000, help="Number of training epochs (int)"
  )

  args = parser.parse_args()

  main(args)
