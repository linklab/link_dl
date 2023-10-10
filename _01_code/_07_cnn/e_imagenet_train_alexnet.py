import torch
from torch import nn, optim
from datetime import datetime
import os
import wandb
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

BASE_PATH = str(Path(__file__).resolve().parent.parent.parent)  # BASE_PATH: /Users/yhhan/git/link_dl
import sys
sys.path.append(BASE_PATH)

CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_FILE_PATH = os.path.join(CURRENT_FILE_PATH, "checkpoints")
if not os.path.isdir(CHECKPOINT_FILE_PATH):
  os.makedirs(os.path.join(CURRENT_FILE_PATH, "checkpoints"))

import sys
sys.path.append(BASE_PATH)

from _01_code._99_common_utils.utils import get_num_cpu_cores, is_linux, is_windows
from _01_code._06_fcn_best_practice.c_trainer import ClassificationTrainer
from _01_code._06_fcn_best_practice.e_arg_parser import get_parser


def get_imagenet_data():
  """
  Before using this function, it is required to download ImageNet 2012 dataset and place
  the files 'ILSVRC2012_devkit_t12.tar.gz' and 'ILSVRC2012_img_train.tar' or 'ILSVRC2012_img_val.tar'
  based on split in the root directory.
  Refer to https://image-net.org/download-images and https://on-ai.tistory.com/8
  """
  data_path = os.path.join(os.path.pardir, os.path.pardir, "_00_data", "j_imagenet")

  imagenet_train = datasets.ImageNet(
    data_path, split="train", transform=transforms.ToTensor()
  )

  imagenet_validation = datasets.ImageNet(
    data_path, split="val", transform=transforms.ToTensor()
  )

  print("Num Train Samples: ", len(imagenet_train))
  print("Num Validation Samples: ", len(imagenet_validation))

  num_data_loading_workers = get_num_cpu_cores() if is_linux() or is_windows() else 0
  print("Number of Data Loading Workers:", num_data_loading_workers)

  train_data_loader = DataLoader(
    dataset=imagenet_train, batch_size=wandb.config.batch_size, shuffle=True,
    pin_memory=True, num_workers=num_data_loading_workers
  )

  validation_data_loader = DataLoader(
    dataset=imagenet_validation, batch_size=wandb.config.batch_size,
    pin_memory=True, num_workers=num_data_loading_workers
  )

  mnist_transforms = nn.Sequential(
    transforms.ConvertImageDtype(torch.float)
  )

  return train_data_loader, validation_data_loader, mnist_transforms


def get_alexnet_model():
  class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        """
        Define and allocate layers for this neural net.

        Args:
            num_classes (int): number of classes to predict with this model
        """
        super().__init__()
        # The image in the original paper states that width and height are 224 pixels, but
        # the correct input size should be: B x 3 x 227 x 227
        self.cnn = nn.Sequential(
            # B x 3 x 227 x 227 --> B x 96 x ((227 - 11) / 4 + 1) x ((227 - 11) / 4 + 1) = B x 96 x 55 x 55
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(11, 11), stride=(4, 4)),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            # B x 96 x 55 x 55 --> B x 96 x ((55 - 3) / 2 + 1) x ((30 - 2) / 2 + 1) = B x 96 x 27 x 27
            nn.MaxPool2d(kernel_size=3, stride=2),

            # B x 96 x 27 x 27 --> B x 256 x ((27 - 5 + 4) / 1 + 1) x ((27 - 5 + 4) / 1 + 1) = B x 256 x 27 x 27
            nn.Conv2d(96, 256, (5, 5), (1, 1), padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            # B x 256 x 27 x 27 --> B x 256 x ((27 - 3) / 2 + 1) x ((27 - 3) / 2 + 1) = B x 256 x 13 x 13
            nn.MaxPool2d(kernel_size=3, stride=2),

            # B x 256 x 13 x 13 --> B x 384 x ((13 - 3 + 2) / 1 + 1) x ((13 - 3 + 2) / 1 + 1) = B x 384 x 13 x 13
            nn.Conv2d(256, 384, (3, 3), (1, 1), padding=1),
            nn.ReLU(),

            # B x 384 x 13 x 13 --> B x 384 x ((13 - 3 + 2) / 1 + 1) x ((13 - 3 + 2) / 1 + 1) = B x 384 x 13 x 13
            nn.Conv2d(384, 384, (3, 3), (1, 1), padding=1),
            nn.ReLU(),

            # B x 384 x 13 x 13 --> B x 256 x ((13 - 3 + 2) / 1 + 1) x ((13 - 3 + 2) / 1 + 1) = B x 256 x 13 x 13
            nn.Conv2d(384, 256, (3, 3), (1, 1), padding=1),
            nn.ReLU(),
            # B x 256 x 13 x 13 --> B x 256 x ((13 - 3) / 2 + 1) x ((13 - 3) / 2 + 1) = B x 256 x 6 x 6)
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # classifier is just a name for linear layers
        self.fcn = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes),
        )

    def forward(self, x):
        """
        Pass the input through the net.
        """
        x = self.cnn(x)
        x = x.view(-1, 256 * 6 * 6)  # reduce the dimensions for linear layer input
        return self.fcn(x)

  my_model = AlexNet()

  return my_model


def main(args):
  run_time_str = datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S')

  config = {
    'epochs': args.epochs,
    'batch_size': args.batch_size,
    'validation_intervals': args.validation_intervals,
    'learning_rate': args.learning_rate,
    'early_stop_patience': args.early_stop_patience
  }

  project_name = "alexnet_imagenet"
  wandb.init(
    mode="online" if args.wandb else "disabled",
    project=project_name,
    notes="imagenet experiment with alexnet",
    tags=["alexnet", "imagenet"],
    name=run_time_str,
    config=config
  )
  print(args)
  print(wandb.config)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(f"Training on device {device}.")

  train_data_loader, validation_data_loader, mnist_transforms = get_imagenet_data()
  model = get_alexnet_model()
  model.to(device)
  wandb.watch(model)

  from torchinfo import summary
  summary(model=model, input_size=(1, 3, 227, 227))

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