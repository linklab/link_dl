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


def get_resnet_model():
  class Residual(nn.Module):
    """The Residual block of ResNet models."""
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.LazyConv2d(out_channels=num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.LazyConv2d(out_channels=num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.LazyConv2d(out_channels=num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()

    def forward(self, X):
        Y = torch.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return torch.relu(Y)


  class ResNet(nn.Module):
    def __init__(self, arch, n_outputs=10):
      super(ResNet, self).__init__()
      self.model = nn.Sequential(
        nn.Sequential(
          nn.LazyConv2d(out_channels=64, kernel_size=7, stride=2, padding=3),
          nn.LazyBatchNorm2d(),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
      )

      for i, b in enumerate(arch):
        self.model.add_module(
          name=f'b{i + 2}', module=self.block(*b, first_block=(i == 0))
        )

      self.model.add_module(
        name='last',
        module=nn.Sequential(
          nn.AdaptiveAvgPool2d((1, 1)),
          nn.Flatten(),
          nn.LazyLinear(n_outputs)
        )
      )

    def block(self, num_residuals, num_channels, first_block=False):
      blk = []
      for i in range(num_residuals):
        if i == 0 and not first_block:
          blk.append(Residual(num_channels=num_channels, use_1x1conv=True, strides=2))
        else:
          blk.append(Residual(num_channels=num_channels))
      return nn.Sequential(*blk)

    def forward(self, x):
      x = self.model(x)
      return x

  my_model = ResNet(arch=((2, 64), (2, 128), (2, 256), (2, 512)), n_outputs=10)

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
  name = "resnet_{0}".format(run_time_str)
  wandb.init(
    mode="online" if args.wandb else "disabled",
    project=project_name,
    notes="cifar10 experiment with resnet",
    tags=["resnet", "cifar10"],
    name=name,
    config=config
  )
  print(args)
  print(wandb.config)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(f"Training on device {device}.")

  train_data_loader, validation_data_loader, cifar10_transforms = get_cifar10_data(flatten=False)
  model = get_resnet_model()
  model.to(device)
  #wandb.watch(model)

  from torchinfo import summary
  summary(
    model=model, input_size=(1, 3, 32, 32),
    col_names=["kernel_size", "input_size", "output_size", "num_params", "mult_adds"]
  )

  optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

  classification_trainer = ClassificationTrainer(
    project_name + "_googlenet", model, optimizer, train_data_loader, validation_data_loader, cifar10_transforms,
    run_time_str, wandb, device, CHECKPOINT_FILE_PATH
  )
  classification_trainer.train_loop()

  wandb.finish()


if __name__ == "__main__":
  parser = get_parser()
  args = parser.parse_args()
  main(args)
  # python _01_code/_09_modern_cnns/_02_googlenet/a_cifar10_train_googlenet.py --wandb -v 10

