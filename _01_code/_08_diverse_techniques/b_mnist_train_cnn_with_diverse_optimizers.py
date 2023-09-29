import torch
from torch import nn, optim
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

import sys
sys.path.append(BASE_PATH)

from _01_code._06_fcn_best_practice.f_mnist_train_fcn import get_data
from _01_code._06_fcn_best_practice.e_arg_parser import get_parser
from _01_code._08_diverse_techniques.a_trainer_with_diverse_optimizers import ClassificationTrainerWithDiverseOptimizers


def get_cnn_model():
  class MyModel(nn.Module):
    def __init__(self, in_channels, n_output):
      super().__init__()

      self.model = nn.Sequential(
        # 1 x 28 x 28 --> 6 x (28 - 5 + 1) x (28 - 5 + 1) = 6 x 24 x 24
        nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=(5, 5), stride=(1, 1)),
        # 6 x 24 x 24 --> 6 x 12 x 12
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.ReLU(),
        # 6 x 12 x 12 --> 16 x (12 - 5 + 1) x (12 - 5 + 1) = 16 x 8 x 8
        nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1)),
        # 16 x 8 x 8 --> 16 x 4 x 4
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(256, 84),
        nn.ReLU(),
        nn.Linear(84, n_output),
      )

    def forward(self, x):
      x = self.model(x)
      return x

  # 1 * 28 * 28
  my_model = MyModel(in_channels=1, n_output=10)

  return my_model


def main(args):
  run_time_str = datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S')

  config = {
    'epochs': args.epochs,
    'batch_size': args.batch_size,
    'validation_intervals': args.validation_intervals,
    'print_epochs': args.print_epochs,
    'learning_rate': args.learning_rate,
  }

  wandb.init(
    mode="online" if args.wandb else "disabled",
    project="cnn_mnist_with_diverse_optimizers",
    notes="mnist experiment with cnn and diverse optimziers",
    tags=["cnn", "mnist", "diverse_optimizers"],
    name=run_time_str,
    config=config
  )
  print(args)
  print(wandb.config)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(f"Training on device {device}.")

  train_data_loader, validation_data_loader, mnist_transforms = get_data(flatten=False)

  optimizer_names = ["SGD", "Momentum", "RMSProp", "Adam"]
  models = []

  for _ in optimizer_names:
    model = get_cnn_model()
    model.to(device)
    models.append(model)

  optimizers = [
    optim.SGD(models[0].parameters(), lr=wandb.config.learning_rate),
    optim.SGD(models[1].parameters(), lr=wandb.config.learning_rate, momentum=0.9),
    optim.RMSprop(models[2].parameters(), lr=wandb.config.learning_rate),
    optim.Adam(models[3].parameters(), lr=wandb.config.learning_rate)
  ]

  classification_trainer = ClassificationTrainerWithDiverseOptimizers(
    "cnn_mnist_with_diverse_optimizers", models, optimizers, optimizer_names, train_data_loader, validation_data_loader, mnist_transforms,
    run_time_str, wandb, device, CHECKPOINT_FILE_PATH
  )
  classification_trainer.train_loop()

  wandb.finish()


if __name__ == "__main__":
  parser = get_parser()
  args = parser.parse_args()
  main(args)
  # python _01_code/_08_diverse_techniques/b_mnist_train_cnn_with_diverse_optimizers.py --wandb -v 1