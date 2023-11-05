import torch
from torch import optim, nn
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
from _01_code._06_fcn_best_practice.h_cifar10_train_fcn import get_cifar10_data
from _01_code._08_diverse_techniques.e_cifar10_train_cnn_with_dropout import get_cnn_model_with_dropout
from _01_code._08_diverse_techniques.a_arg_parser import get_parser


def get_cnn_model_with_dropout_and_batch_normalization():
  class MyModel(nn.Module):
    def __init__(self, in_channels, n_output):
      super().__init__()

      self.model = nn.Sequential(
        # 3 x 32 x 32 --> 6 x (32 - 5 + 1) x (32 - 5 + 1) = 6 x 28 x 28
        nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=(5, 5), stride=(1, 1)),
        # 6 x 28 x 28 --> 6 x 14 x 14
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.BatchNorm2d(num_features=6),
        nn.ReLU(),
        # 6 x 14 x 14 --> 16 x (14 - 5 + 1) x (14 - 5 + 1) = 16 x 10 x 10
        nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1)),
        # 16 x 10 x 10 --> 16 x 5 x 5
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.BatchNorm2d(num_features=16),
        nn.ReLU(),
        nn.Flatten(),
        nn.Dropout(p=0.5),      # p: dropout probability
        nn.Linear(400, 128),
        nn.BatchNorm1d(num_features=128),
        nn.ReLU(),
        nn.Dropout(p=0.5),      # p: dropout probability
        nn.Linear(128, n_output),
      )

    def forward(self, x):
      x = self.model(x)
      return x

  # 3 * 32 * 32
  my_model = MyModel(in_channels=3, n_output=10)

  return my_model


def get_cnn_model_with_dropout_and_layer_normalization():
  class MyModel(nn.Module):
    def __init__(self, in_channels, n_output):
      super().__init__()

      self.model = nn.Sequential(
        # 3 x 32 x 32 --> 6 x (32 - 5 + 1) x (32 - 5 + 1) = 6 x 28 x 28
        nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=(5, 5), stride=(1, 1)),
        # 6 x 28 x 28 --> 6 x 14 x 14
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.LayerNorm(normalized_shape=[6, 14, 14]),
        nn.ReLU(),
        # 6 x 14 x 14 --> 16 x (14 - 5 + 1) x (14 - 5 + 1) = 16 x 10 x 10
        nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1)),
        # 16 x 10 x 10 --> 16 x 5 x 5
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.LayerNorm(normalized_shape=[16, 5, 5]),
        nn.ReLU(),
        nn.Flatten(),
        nn.Dropout(p=0.5),      # p: dropout probability
        nn.Linear(400, 128),
        nn.LayerNorm(normalized_shape=[128]),
        nn.ReLU(),
        nn.Dropout(p=0.5),      # p: dropout probability
        nn.Linear(128, n_output),
      )

    def forward(self, x):
      x = self.model(x)
      return x

  # 3 * 32 * 32
  my_model = MyModel(in_channels=3, n_output=10)

  return my_model


def main(args):
  config = {
    'epochs': args.epochs,
    'batch_size': args.batch_size,
    'validation_intervals': args.validation_intervals,
    'learning_rate': args.learning_rate,
    'early_stop_patience': args.early_stop_patience,
    'weight_decay': args.weight_decay,
    'dropout': args.dropout,
    'normalization': args.normalization
  }

  normalization_names = ["no_normalization", "batch_norm", "layer_norm"]
  technique_name = "{0}".format(normalization_names[args.normalization])
  run_time_str = datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S')
  name = "{0}_{1}".format(technique_name, run_time_str)

  project_name = "cnn_cifar10_with_normalization"
  wandb.init(
    mode="online" if args.wandb else "disabled",
    project=project_name,
    notes="cifar10 experiment with cnn and normalization",
    tags=["cnn", "cifar10", "normalization"],
    name=name,
    config=config
  )
  print(args)
  print(wandb.config)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(f"Training on device {device}.")

  train_data_loader, validation_data_loader, cifar10_transforms = get_cifar10_data(flatten=False)

  if args.normalization == 0:
    model = get_cnn_model_with_dropout()
  elif args.normalization == 1:
    model = get_cnn_model_with_dropout_and_batch_normalization()
  elif args.normalization == 2:
    model = get_cnn_model_with_dropout_and_layer_normalization()
  else:
    raise ValueError()

  model.to(device)
  wandb.watch(model)

  optimizers = [
    optim.SGD(model.parameters(), lr=wandb.config.learning_rate, weight_decay=args.weight_decay),
    optim.SGD(model.parameters(), lr=wandb.config.learning_rate, momentum=0.9, weight_decay=args.weight_decay),
    optim.RMSprop(model.parameters(), lr=wandb.config.learning_rate, weight_decay=args.weight_decay),
    optim.Adam(model.parameters(), lr=wandb.config.learning_rate, weight_decay=args.weight_decay)
  ]

  print("Optimizer:", optimizers[args.optimizer])

  classification_trainer = ClassificationTrainer(
    project_name, model, optimizers[args.optimizer],
    train_data_loader, validation_data_loader, cifar10_transforms,
    run_time_str, wandb, device, CHECKPOINT_FILE_PATH
  )
  classification_trainer.train_loop()

  wandb.finish()


if __name__ == "__main__":
  parser = get_parser()
  args = parser.parse_args()
  main(args)
  # python _01_code/_08_diverse_techniques/f_cifar10_train_cnn_with_normalization.py --wandb -v 1 -o 3 -w 0.002 --dropout -n 0
  # python _01_code/_08_diverse_techniques/f_cifar10_train_cnn_with_normalization.py --wandb -v 1 -o 3 -w 0.002 --dropout -n 1
  # python _01_code/_08_diverse_techniques/f_cifar10_train_cnn_with_normalization.py --wandb -v 1 -o 3 -w 0.002 --dropout -n 2
