import torch
from torch import nn, optim
from datetime import datetime
import os
import wandb
from pathlib import Path

BASE_PATH = str(Path(__file__).resolve().parent.parent.parent.parent) # BASE_PATH: /Users/yhhan/git/link_dl
import sys
sys.path.append(BASE_PATH)

CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_FILE_PATH = os.path.join(CURRENT_FILE_PATH, "checkpoints")
if not os.path.isdir(CHECKPOINT_FILE_PATH):
  os.makedirs(os.path.join(CURRENT_FILE_PATH, "checkpoints"))

import sys
sys.path.append(BASE_PATH)

from _01_code._06_fcn_best_practice.c_trainer import ClassificationTrainer
from _01_code._06_fcn_best_practice.h_cifar10_train_fcn import get_cifar10_data
from _01_code._09_modern_cnns.a_arg_parser import get_parser


def get_vgg_model():
  def vgg_block(num_conv_layers, out_channels):
    layers = []

    for _ in range(num_conv_layers):
      layers.append(nn.LazyConv2d(out_channels=out_channels, kernel_size=3, padding=1))
      layers.append(nn.ReLU())

    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

    block = nn.Sequential(*layers)
    return block

  class VGG(nn.Module):
    def __init__(self, block_info, n_output=10):
      super().__init__()

      conv_blocks = []
      for (num_conv_layers, out_channels) in block_info:
        conv_blocks.append(vgg_block(num_conv_layers, out_channels))

      self.model = nn.Sequential(
        *conv_blocks,
        nn.Flatten(),
        nn.LazyLinear(out_features=512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.LazyLinear(out_features=512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.LazyLinear(n_output)
      )

    def forward(self, x):
      x = self.model(x)
      return x

  my_model = VGG(
    block_info=((1, 64), (1, 128), (2, 256), (2, 512), (2, 512)),
    n_output=10
  )

  return my_model


def main(args):
  run_time_str = datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S')

  config = {
    'epochs': args.epochs,
    'batch_size': args.batch_size,
    'validation_intervals': args.validation_intervals,
    'learning_rate': args.learning_rate,
    'early_stop_patience': args.early_stop_patience,
  }

  project_name = "modern_cifar10"
  name = "vgg_{0}".format(run_time_str)
  wandb.init(
    mode="online" if args.wandb else "disabled",
    project=project_name,
    notes="cifar10 experiment with vgg",
    tags=["vgg", "cifar10"],
    name=name,
    config=config
  )
  print(args)
  print(wandb.config)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(f"Training on device {device}.")

  train_data_loader, validation_data_loader, cifar10_transforms = get_cifar10_data(flatten=False)
  model = get_vgg_model()
  model.to(device)
  #wandb.watch(model)

  from torchinfo import summary
  summary(
    model=model, input_size=(1, 3, 32, 32),
    col_names=["kernel_size", "input_size", "output_size", "num_params", "mult_adds"]
  )

  optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

  classification_trainer = ClassificationTrainer(
    project_name + "_vgg", model, optimizer, train_data_loader, validation_data_loader, cifar10_transforms,
    run_time_str, wandb, device, CHECKPOINT_FILE_PATH
  )
  classification_trainer.train_loop()

  wandb.finish()


if __name__ == "__main__":
  parser = get_parser()
  args = parser.parse_args()
  main(args)
  # python _01_code/_09_modern_cnns/_01_vgg/a_cifar10_train_vgg.py --wandb -v 10

