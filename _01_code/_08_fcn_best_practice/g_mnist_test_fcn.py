import numpy as np
import torch
import os

from matplotlib import pyplot as plt
from torch import nn
from torchvision import transforms, datasets
from pathlib import Path

from torch.utils.data import DataLoader

BASE_PATH = str(Path(__file__).resolve().parent.parent.parent) # BASE_PATH: /Users/yhhan/git/link_dl
CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_FILE_PATH = os.path.join(CURRENT_FILE_PATH, "checkpoints")

import sys
sys.path.append(BASE_PATH)

from _01_code._08_fcn_best_practice.f_mnist_train_fcn import get_model
from _01_code._08_fcn_best_practice.d_tester import ClassificationTester


def get_mnist_test_data(flatten=False):
  data_path = os.path.join(BASE_PATH, "_00_data", "h_mnist")

  mnist_test_images = datasets.MNIST(data_path, train=False, download=True)

  mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transforms.ToTensor())
  test_data_loader = DataLoader(dataset=mnist_test, batch_size=len(mnist_test))

  mnist_transforms = nn.Sequential(
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=0.1307, std=0.3081),
  )

  if flatten:
    mnist_transforms.append(
      nn.Flatten()
    )

  return mnist_test_images, test_data_loader, mnist_transforms


def main():
  mnist_test_images, test_data_loader, mnist_transforms = get_mnist_test_data(flatten=True)

  test_model = get_model()

  project_name = "fcn_mnist"
  classification_tester = ClassificationTester(
    project_name, test_model, test_data_loader, mnist_transforms, CHECKPOINT_FILE_PATH
  )
  classification_tester.test()

  print()

  img, label = mnist_test_images[0]
  print("     LABEL:", label)
  plt.imshow(img)
  plt.show()

  # torch.tensor(np.array(mnist_test_images[0][0])).unsqueeze(dim=0).unsqueeze(dim=0).shape: (1, 1, 28, 28)
  output = classification_tester.test_single(
    torch.tensor(np.array(mnist_test_images[0][0])).unsqueeze(dim=0).unsqueeze(dim=0)
  )
  print("PREDICTION:", output)


if __name__ == "__main__":
  main()
