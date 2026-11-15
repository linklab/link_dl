import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import torchvision
import wandb
from torch import nn, optim
from torchinfo import summary

BASE_PATH = str(Path(__file__).resolve().parent.parent.parent.parent)  # BASE_PATH: /Users/yhhan/git/link_dl
sys.path.append(BASE_PATH)

CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_FILE_PATH = os.path.join(CURRENT_FILE_PATH, "checkpoints")
if not os.path.isdir(CHECKPOINT_FILE_PATH):
  os.makedirs(CHECKPOINT_FILE_PATH)

from _01_code._09_fcn_best_practice.c_trainer import ClassificationTrainer
from _01_code._09_fcn_best_practice.h_cifar10_train_fcn import get_cifar10_data
from _01_code._16_modern_cnns.a_arg_parser import get_parser

# True: torchvision이 제공하는 ResNet-18 사용, False: 아래에서 직접 만든 ResNet-18 사용
USE_PYTORCH_MODEL = False


class ResnetBlock(nn.Module):
  """
  ResNet의 기본 블록 (Basic Block)

  입력 x ──> conv1 ─> bn1 ─> relu ─> conv2 ─> bn2 ──> (+) ─> relu ─> 출력
    │                                                  ↑
    └──────────────── (필요하면 downsample) ────────────┘   <- 이 지름길이 "skip connection"

  핵심 아이디어: 블록이 "입력 전체"를 배우는 대신, 입력과의 "차이(잔차, residual)"만 배운다.
  """

  def __init__(self, out_channels, stride=1):
    super().__init__()

    # LazyConv2d: 입력 채널 수(in_channels)는 첫 forward 때 입력을 보고 자동으로 정해진다.
    # 첫 번째 conv: stride가 2이면 이미지의 가로/세로 크기가 절반으로 줄어든다.
    self.conv1 = nn.LazyConv2d(out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(out_channels)

    # 두 번째 conv: 크기와 채널 수를 그대로 유지한다.
    self.conv2 = nn.LazyConv2d(out_channels, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(out_channels)

    self.relu = nn.ReLU(inplace=True)

    # 지름길(skip connection)의 입력과 conv 경로의 출력은 크기가 같아야 더할 수 있다.
    # stride가 1이 아니면 크기가 달라지므로, 1x1 conv로 지름길의 크기를 맞춰준다.
    if stride != 1:
      self.downsample = nn.Sequential(
        nn.LazyConv2d(out_channels, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_channels)
      )
    else:
      self.downsample = None

  def forward(self, x):
    # 1) 지름길로 보낼 입력을 따로 보관한다.
    identity = x
    if self.downsample is not None:
      identity = self.downsample(x)

    # 2) conv 경로: conv -> bn -> relu -> conv -> bn
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)

    # 3) 두 경로를 더한 뒤 relu를 적용한다.
    out = out + identity
    out = self.relu(out)

    return out


class ResNet18(nn.Module):
  """
  ResNet-18 = 시작부분(stem) + 4개의 stage(각 stage는 블록 2개) + 분류부분

  입력 (3, 32, 32)
    stem   : conv(7x7, stride 2) -> bn -> relu -> maxpool(stride 2)  => (64, 8, 8)
    layer1 : 블록 2개, 채널  64, 크기 유지                           => (64, 8, 8)
    layer2 : 블록 2개, 채널 128, 크기 절반                           => (128, 4, 4)
    layer3 : 블록 2개, 채널 256, 크기 절반                           => (256, 2, 2)
    layer4 : 블록 2개, 채널 512, 크기 절반                           => (512, 1, 1)
    분류   : 평균 풀링 -> 펼치기 -> fc                               => (num_classes,)
  """

  def __init__(self, num_classes=10):
    super().__init__()

    # 시작부분 (stem)
    self.conv1 = nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    # 4개의 stage: 채널 수는 점점 늘리고, 이미지 크기는 점점 줄인다.
    self.layer1 = self._make_layer(out_channels=64, stride=1)
    self.layer2 = self._make_layer(out_channels=128, stride=2)
    self.layer3 = self._make_layer(out_channels=256, stride=2)
    self.layer4 = self._make_layer(out_channels=512, stride=2)

    # 분류부분
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 어떤 크기든 (1, 1)로 평균
    self.fc = nn.Linear(512, num_classes)

  def _make_layer(self, out_channels, stride):
    # 한 stage는 블록 2개로 이루어진다.
    # 첫 번째 블록만 채널 수/크기를 바꾸고(stride 적용), 두 번째 블록은 그대로 유지한다.
    first_block = ResnetBlock(out_channels, stride=stride)
    second_block = ResnetBlock(out_channels, stride=1)
    return nn.Sequential(first_block, second_block)

  def forward(self, x):
    # 시작부분
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    # 4개의 stage
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    # 분류부분
    x = self.avgpool(x)
    x = torch.flatten(x, 1)  # (batch, 512, 1, 1) -> (batch, 512)
    x = self.fc(x)
    return x


def get_resnet_model(num_classes=10):
  return ResNet18(num_classes=num_classes)


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
  with wandb.init(
    mode="online" if args.wandb else "disabled",
    project=project_name,
    notes="cifar10 experiment with resnet",
    tags=["resnet", "cifar10"],
    name=name,
    config=config
  ) as run:
    print(args)
    print(run.config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on device {device}.")

    train_data_loader, validation_data_loader, cifar10_transforms = get_cifar10_data(run, flatten=False)

    if USE_PYTORCH_MODEL:
      model = torchvision.models.resnet18(num_classes=10)
    else:
      model = get_resnet_model(num_classes=10)
    model.to(device)

    summary(
      model=model, input_size=(1, 3, 32, 32),
      col_names=["kernel_size", "input_size", "output_size", "num_params", "mult_adds"]
    )

    optimizer = optim.Adam(model.parameters(), lr=run.config.learning_rate)

    classification_trainer = ClassificationTrainer(
      project_name + "_resnet", model, optimizer, train_data_loader, validation_data_loader, cifar10_transforms,
      run_time_str, run, device, CHECKPOINT_FILE_PATH
    )
    classification_trainer.train_loop()


if __name__ == "__main__":
  parser = get_parser()
  args = parser.parse_args()
  main(args)
  # python _01_code/_16_modern_cnns/_04_resnet/a_cifar10_train_resnet.py --wandb -v 10
