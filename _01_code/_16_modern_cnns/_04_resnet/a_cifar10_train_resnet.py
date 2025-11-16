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

from _01_code._09_fcn_best_practice.c_trainer import ClassificationTrainer
from _01_code._09_fcn_best_practice.h_cifar10_train_fcn import get_cifar10_data
from _01_code._16_modern_cnns.a_arg_parser import get_parser

import torchvision

USE_PYTORCH_MODEL = False

def get_resnet_model(num_classes=10):
  class ResnetBlock(nn.Module):

    def __init__(self, out_channels, stride=1, downsample=None):
      """
      in_channels는 LazyConv2d가 자동으로 추론함!
      """
      super(ResnetBlock, self).__init__()

      self.conv1 = nn.LazyConv2d(
        out_channels, kernel_size=3, stride=stride, padding=1, bias=False
      )
      self.bn1 = nn.LazyBatchNorm2d()

      self.conv2 = nn.LazyConv2d(
        out_channels, kernel_size=3, stride=1, padding=1, bias=False
      )
      self.bn2 = nn.LazyBatchNorm2d()

      self.relu = nn.ReLU(inplace=True)
      self.downsample = downsample

    def forward(self, x):
      identity = x

      out = self.conv1(x)
      out = self.bn1(out)
      out = self.relu(out)

      out = self.conv2(out)
      out = self.bn2(out)

      if self.downsample is not None:
        identity = self.downsample(x)

      out += identity
      out = self.relu(out)

      return out

  # ------------------------------------
  # ResNet-18 using Lazy Modules
  # ------------------------------------
  class ResNet18(nn.Module):
    def __init__(self):
      super(ResNet18, self).__init__()

      # 처음 stem 부분 → LazyConv 사용
      self.conv1 = nn.LazyConv2d(
        64, kernel_size=7, stride=2, padding=3, bias=False
      )
      self.bn1 = nn.LazyBatchNorm2d()
      self.relu = nn.ReLU(inplace=True)
      self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

      # ResNet stages (2 blocks × 4 layers)
      self.layer1 = self._make_layer(out_channels=64, blocks=2, stride=1)
      self.layer2 = self._make_layer(out_channels=128, blocks=2, stride=2)
      self.layer3 = self._make_layer(out_channels=256, blocks=2, stride=2)
      self.layer4 = self._make_layer(out_channels=512, blocks=2, stride=2)

      self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
      self.fc = nn.LazyLinear(num_classes)

    def _make_layer(self, out_channels, blocks, stride):
      downsample = None

      # downsample 로직 → LazyConv/BatchNorm 활용
      if stride != 1:
        downsample = nn.Sequential(
          nn.LazyConv2d(out_channels, kernel_size=1, stride=stride, bias=False),
          nn.LazyBatchNorm2d()
        )

      layers = []
      layers.append(ResnetBlock(out_channels, stride=stride, downsample=downsample))

      for _ in range(1, blocks):
        layers.append(ResnetBlock(out_channels))

      return nn.Sequential(*layers)

    def forward(self, x):
      x = self.conv1(x)
      x = self.bn1(x)
      x = self.relu(x)
      x = self.maxpool(x)

      x = self.layer1(x)
      x = self.layer2(x)
      x = self.layer3(x)
      x = self.layer4(x)

      x = self.avgpool(x)
      x = torch.flatten(x, 1)
      x = self.fc(x)
      return x

  my_model = ResNet18()
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
  model = torchvision.models.resnet18(num_classes=10) if USE_PYTORCH_MODEL else get_resnet_model(num_classes=10)
  model.to(device)

  from torchinfo import summary
  summary(
    model=model, input_size=(1, 3, 32, 32),
    col_names=["kernel_size", "input_size", "output_size", "num_params", "mult_adds"]
  )

  optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

  classification_trainer = ClassificationTrainer(
    project_name + "_resnet", model, optimizer, train_data_loader, validation_data_loader, cifar10_transforms,
    run_time_str, wandb, device, CHECKPOINT_FILE_PATH
  )
  classification_trainer.train_loop()

  wandb.finish()


if __name__ == "__main__":
  parser = get_parser()
  args = parser.parse_args()
  main(args)
  # python _01_code/_09_modern_cnns/_02_googlenet/a_cifar10_train_googlenet.py --wandb -v 10

