import torch
from torch import nn, optim
from datetime import datetime
import os
import wandb
from pathlib import Path

BASE_PATH = str(Path(__file__).resolve().parent.parent.parent)  # BASE_PATH: /Users/yhhan/git/link_dl
import sys
sys.path.append(BASE_PATH)

CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_FILE_PATH = os.path.join(CURRENT_FILE_PATH, "checkpoints")
if not os.path.isdir(CHECKPOINT_FILE_PATH):
  os.makedirs(os.path.join(CURRENT_FILE_PATH, "checkpoints"))

import sys
sys.path.append(BASE_PATH)

from _01_code._08_fcn_best_practice.c_trainer import ClassificationTrainer
from _01_code._08_fcn_best_practice.e_arg_parser import get_parser
from _01_code._08_fcn_best_practice.h_cifar10_train_fcn import get_cifar10_data


def get_alexnet_model():
  class AlexNet(nn.Module):
    def __init__(self, in_channels=3, n_output=10):
        """
        Define and allocate layers for this neural net.
        """
        super().__init__()
        # The image in the original paper states that width and height are 224 pixels, but
        # the correct input size should be : (B x 3 x 227 x 227)
        self.cnn = nn.Sequential(
            # B x 3 x 32 x 32 --> B x 64 x (32 - 3 + 1) x (32 - 3 + 1) = B x 64 x 30 x 30
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.LocalResponseNorm(size=3, alpha=0.0001, beta=0.75, k=2),
            # B x 64 x 30 x 30 --> B x 64 x ((30 - 2) / 2 + 1) x ((30 - 2) / 2 + 1) = B x 64 x 15 x 15
            nn.MaxPool2d(kernel_size=2, stride=2),

            # B x 64 x 15 x 15 --> B x 64 x (15 - 3 + 2 + 1) x (15 - 3 + 2 + 1) = B x 192 x 15 x 15
            nn.Conv2d(64, 192, (3, 3), (1, 1), padding=1),
            nn.ReLU(),
            nn.LocalResponseNorm(size=3, alpha=0.0001, beta=0.75, k=2),
            # B x 192 x 15 x 15 --> B x 192 x ((15 - 3) / 2 + 1) x ((15 - 3) / 2 + 1) = B x 192 x 7 x 7
            nn.MaxPool2d(kernel_size=3, stride=2),

            # B x 192 x 7 x 7 --> B x 256 x ((7 - 3 + 2) / 1 + 1) x ((13 - 3 + 2) / 1 + 1) = B x 256 x 7 x 7
            nn.Conv2d(192, 256, (3, 3), (1, 1), padding=1),
            nn.ReLU(),

            # B x 256 x 7 x 7 --> B x 256 x ((7 - 3 + 2) / 1 + 1) x ((13 - 3 + 2) / 1 + 1) = B x 256 x 7 x 7
            nn.Conv2d(256, 256, (3, 3), (1, 1), padding=1),
            nn.ReLU(),

            # B x 256 x 7 x 7 --> B x 192 x ((7 - 2) / 1 + 1) x ((7 - 2) / 1 + 1) = B x 192 x 6 x 6
            nn.Conv2d(256, 192, (2, 2), (1, 1)),
            nn.ReLU(),
            # B x 192 x 6 x 6 --> B x 192 x ((6 - 2) / 2 + 1) x ((6 - 2) / 2 + 1) = B x 192 x 3 x 3
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # classifier is just a name for linear layers
        self.fcn = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=192 * 3 * 3, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=n_output),
        )

    def forward(self, x):
        """
        Pass the input through the net.
        """
        x = self.cnn(x)
        x = x.view(-1, 192 * 3 * 3)  # reduce the dimensions for linear layer input
        return self.fcn(x)

  my_model = AlexNet(in_channels=3, n_output=10)

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

  project_name = "alexnet_cifar10"
  wandb.init(
    mode="online" if args.wandb else "disabled",
    project=project_name,
    notes="cifar10 experiment with alexnet",
    tags=["alexnet", "cifar10"],
    name=run_time_str,
    config=config
  )
  print(args)
  print(wandb.config)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(f"Training on device {device}.")

  train_data_loader, validation_data_loader, mnist_transforms = get_cifar10_data()
  model = get_alexnet_model()
  model.to(device)
  wandb.watch(model)

  from torchinfo import summary
  summary(
    model=model, input_size=(1, 3, 32, 32),
    col_names=["kernel_size", "input_size", "output_size", "num_params", "mult_adds"]
  )

  optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

  classification_trainer = ClassificationTrainer(
    project_name, model, optimizer, train_data_loader, validation_data_loader, mnist_transforms,
    run_time_str, wandb, device, CHECKPOINT_FILE_PATH
  )
  classification_trainer.train_loop()

  wandb.finish()


if __name__ == "__main__":
  parser = get_parser()
  args = parser.parse_args()
  main(args)
  # python _01_code/_07_cnn/a_mnist_train_cnn.py --wandb -b 2048 -r 1e-3 -v 10
  # python _01_code/_07_cnn/a_mnist_train_cnn.py --no-wandb -b 2048 -r 1e-3 -v 10