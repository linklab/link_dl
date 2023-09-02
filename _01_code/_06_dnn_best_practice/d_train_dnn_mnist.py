import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from datetime import datetime
import os
import wandb

from pathlib import Path
BASE_PATH = str(Path(__file__).resolve().parent.parent.parent)
print("BASE_PATH", BASE_PATH)
CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))

import sys
sys.path.append(BASE_PATH)

from a_trainer import ClassificationTrainer
from b_tester import ClassificationTester
from _01_code._99_common_utils.utils import is_linux, get_num_cpu_cores


def get_ready():
  if not os.path.isdir(os.path.join(CURRENT_FILE_PATH, "checkpoints")):
    os.makedirs(os.path.join(CURRENT_FILE_PATH, "checkpoints"))


def get_data():
  data_path = os.path.join(os.path.pardir, os.path.pardir, "_00_data", "i_mnist")

  mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transforms.ToTensor())
  mnist_train, mnist_test = random_split(mnist_train, [59_000, 1_000])
  mnist_validation = datasets.MNIST(data_path, train=False, download=True, transform=transforms.ToTensor())

  print("Num Train Samples: ", len(mnist_train))
  print("Num Validation Samples: ", len(mnist_validation))
  print("Num Test Samples: ", len(mnist_test))

  num_data_loading_workers = get_num_cpu_cores() if is_linux() else 0
  print("Number of Data Loading Workers:", num_data_loading_workers)

  train_data_loader = DataLoader(
    dataset=mnist_train, batch_size=wandb.config.batch_size, shuffle=True,
    pin_memory=True, num_workers=num_data_loading_workers
  )

  validation_data_loader = DataLoader(
    dataset=mnist_validation, batch_size=wandb.config.batch_size,
    pin_memory=True, num_workers=num_data_loading_workers
  )

  test_data_loader = DataLoader(
    dataset=mnist_test, batch_size=len(mnist_test),
    pin_memory=True, num_workers=num_data_loading_workers
  )

  mnist_transforms = nn.Sequential(
    transforms.Normalize(mean=0.1307, std=0.3081),
    nn.Flatten(),
  )

  return train_data_loader, validation_data_loader, test_data_loader, mnist_transforms


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
    'validation_intervals': args.validation_intervals,
    'learning_rate': args.learning_rate,
    'n_hidden_unit_list': [256, 256],
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

  train_data_loader, validation_data_loader, test_data_loader, mnist_transforms = get_data()
  model, optimizer = get_model_and_optimizer()
  model.to(device)
  wandb.watch(model)

  classification_trainer = ClassificationTrainer(
    "mnist", model, optimizer, train_data_loader, validation_data_loader, mnist_transforms,
    run_time_str, wandb, device
  )
  classification_trainer.train_loop()

  test_model, _ = get_model_and_optimizer()
  classification_tester = ClassificationTester("mnist", test_model, test_data_loader, mnist_transforms)
  classification_tester.test()

  wandb.finish()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument(
    "-w", "--use_wandb", type=bool, default=False, help="True or False"
  )

  parser.add_argument(
    "-e", "--epochs", type=int, default=10_000,
    help="Number of training epochs (int, default: 10,000)"
  )

  parser.add_argument(
    "-b", "--batch_size", type=int, default=2_048,
    help="Batch size (int, default: 256)"
  )

  parser.add_argument(
    "-r", "--learning_rate", type=float, default=1e-3,
    help="Learning rate (float, default: 1e-3)"
  )

  parser.add_argument(
    "-v", "--validation_intervals", type=int, default=10,
    help="Number of training epochs between validations (int, default: 10)"
  )

  args = parser.parse_args()

  main(args)
