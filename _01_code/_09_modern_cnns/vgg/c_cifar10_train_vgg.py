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

from _01_code._06_fcn_best_practice.c_trainer import ClassificationTrainer
from _01_code._06_fcn_best_practice.h_cifar10_train_fcn import get_data
from _01_code._06_fcn_best_practice.e_arg_parser import get_parser


def get_vgg_model():
  def vgg_block(num_convs, out_channels):
    layers = []
    for _ in range(num_convs):
      layers.append(nn.LazyConv2d(out_channels, kernel_size=3, padding=1))
      layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

    block = nn.Sequential(*layers)
    return block

  class VGG(nn.Module):
    def __init__(self, in_channels, n_output):
      super().__init__()

      self.model = nn.Sequential(
        # B x 3 x 32 x 32 --> B x 6 x (32 - 5 + 1) x (32 - 5 + 1) = B x 6 x 28 x 28
        nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=(5, 5), stride=(1, 1)),
        # B x 6 x 28 x 28 --> B x 6 x 14 x 14
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.ReLU(),
        # B x 6 x 14 x 14 --> B x 16 x (14 - 5 + 1) x (14 - 5 + 1) = B x 16 x 10 x 10
        nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1)),
        # B x 16 x 10 x 10 --> B x 16 x 5 x 5
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(400, 128),
        nn.ReLU(),
        nn.Linear(128, n_output),
      )

    def forward(self, x):
      x = self.model(x)
      # print(x.shape, "!!!")
      return x

  # 3 * 32 * 32
  my_model = MyModel(in_channels=3, n_output=10)

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

  project_name = "cnn_cifar10"
  wandb.init(
    mode="online" if args.wandb else "disabled",
    project=project_name,
    notes="cifar10 experiment with cnn",
    tags=["cnn", "cifar10"],
    name=run_time_str,
    config=config
  )
  print(args)
  print(wandb.config)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(f"Training on device {device}.")

  train_data_loader, validation_data_loader, cifar10_transforms = get_data(flatten=False)
  model = get_cnn_model()
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
  # python _01_code/_07_cnn/c_cifar10_train_cnn.py --wandb -b 2048 -r 1e-3 -v 10
  # python _01_code/_07_cnn/c_cifar10_train_cnn.py --no-wandb -b 2048 -r 1e-3 -v 10