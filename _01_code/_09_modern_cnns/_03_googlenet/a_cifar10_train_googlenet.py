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

from _01_code._06_fcn_best_practice.h_cifar10_train_fcn import get_cifar10_data
from _01_code._09_modern_cnns.a_arg_parser import get_parser
from _01_code._09_modern_cnns._03_googlenet.b_googlenet_trainer import GoogLeNetClassificationTrainer


def get_googlenet_model():
  class Inception(nn.Module):
    # c1--c4 are the number of output channels for each branch
    def __init__(self, c1, c2, c3, c4, **kwargs):
      super(Inception, self).__init__(**kwargs)
      # Branch 1
      self.b1_1 = nn.LazyConv2d(out_channels=c1, kernel_size=1)
      # Branch 2
      self.b2_1 = nn.LazyConv2d(out_channels=c2[0], kernel_size=1)
      self.b2_2 = nn.LazyConv2d(out_channels=c2[1], kernel_size=3, padding=1)
      # Branch 3
      self.b3_1 = nn.LazyConv2d(out_channels=c3[0], kernel_size=1)
      self.b3_2 = nn.LazyConv2d(out_channels=c3[1], kernel_size=5, padding=2)
      # Branch 4
      self.b4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
      self.b4_2 = nn.LazyConv2d(out_channels=c4, kernel_size=1)

    def forward(self, x):
      b1 = torch.relu(self.b1_1(x))
      b2 = torch.relu(self.b2_2(torch.relu(self.b2_1(x))))
      b3 = torch.relu(self.b3_2(torch.relu(self.b3_1(x))))
      b4 = torch.relu(self.b4_2(self.b4_1(x)))
      return torch.cat((b1, b2, b3, b4), dim=1)

  class InceptionAux(nn.Module):
    def __init__(self, n_outputs, **kwargs):
      super(InceptionAux, self).__init__(**kwargs)

      self.conv = nn.Sequential(
        #nn.AvgPool2d(kernel_size=5, stride=3),
        nn.LazyConv2d(out_channels=128, kernel_size=1),
      )

      self.fc = nn.Sequential(
        nn.LazyLinear(out_features=1024),
        nn.ReLU(),
        nn.Dropout(),
        nn.LazyLinear(out_features=n_outputs),
      )

    def forward(self, x):
      x = self.conv(x)
      x = x.view(x.shape[0], -1)
      x = self.fc(x)
      return x

  class GoogleNet(nn.Module):
    def __init__(self, n_outputs=10):
      super(GoogleNet, self).__init__()
      self.conv_block = nn.Sequential(self.conv_blk_1(), self.conv_blk_2())
      self.inception_block_1 = self.inception_blk_1()
      self.inception_block_2 = self.inception_blk_2()
      self.inception_block_3 = self.inception_blk_3()
      self.aux_1 = InceptionAux(n_outputs)
      self.aux_2 = InceptionAux(n_outputs)

    def conv_blk_1(self):
      # LocalRespNorm is ignored
      return nn.Sequential(
        nn.LazyConv2d(out_channels=64, kernel_size=7, stride=2, padding=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
      )

    def conv_blk_2(self):
      # LocalRespNorm is ignored
      return nn.Sequential(
        nn.LazyConv2d(out_channels=64, kernel_size=1),
        nn.ReLU(),
        nn.LazyConv2d(out_channels=192, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
      )

    def inception_blk_1(self):
      return nn.Sequential(
        Inception(c1=64, c2=(96, 128), c3=(16, 32), c4=32),
        Inception(c1=128, c2=(128, 192), c3=(32, 96), c4=64),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        Inception(c1=192, c2=(96, 208), c3=(16, 48), c4=64),
      )

    def inception_blk_2(self):
      return nn.Sequential(
        Inception(c1=160, c2=(112, 224), c3=(24, 64), c4=64),
        Inception(c1=128, c2=(128, 256), c3=(24, 64), c4=64),
        Inception(c1=112, c2=(144, 288), c3=(32, 64), c4=64),
      )

    def inception_blk_3(self):
      return nn.Sequential(
        Inception(c1=256, c2=(160, 320), c3=(32, 128), c4=128),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        Inception(c1=256, c2=(160, 320), c3=(32, 128), c4=128),
        Inception(c1=384, c2=(192, 384), c3=(48, 128), c4=128),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()
      )

    def forward(self, x):
      x = self.conv_block(x)
      inception_out_1 = self.inception_block_1(x)
      aux_out_1 = self.aux_1(inception_out_1)
      inception_out_2 = self.inception_block_2(inception_out_1)
      aux_out_2 = self.aux_2(inception_out_2)
      inception_out_3 = self.inception_block_3(inception_out_2)
      return inception_out_3, aux_out_1, aux_out_2

  my_model = GoogleNet()

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
  name = "googlenet_{0}".format(run_time_str)
  wandb.init(
    mode="online" if args.wandb else "disabled",
    project=project_name,
    notes="cifar10 experiment with googlenet",
    tags=["googlenet", "cifar10"],
    name=name,
    config=config
  )
  print(args)
  print(wandb.config)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(f"Training on device {device}.")

  train_data_loader, validation_data_loader, cifar10_transforms = get_cifar10_data(flatten=False)
  model = get_googlenet_model()
  model.to(device)
  #wandb.watch(model)

  from torchinfo import summary
  summary(
    model=model, input_size=(1, 3, 32, 32),
    col_names=["kernel_size", "input_size", "output_size", "num_params", "mult_adds"]
  )

  optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

  classification_trainer = GoogLeNetClassificationTrainer(
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

