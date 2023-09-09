import numpy as np
import torch
import os

from matplotlib import pyplot as plt
from torch import nn
from torchvision import transforms, datasets
from pathlib import Path

from torch.utils.data import DataLoader

BASE_PATH = str(Path(__file__).resolve().parent.parent.parent)
print("BASE_PATH", BASE_PATH)
CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))

import sys
sys.path.append(BASE_PATH)

from g_cifar10_train_fcn import get_model
from d_tester import ClassificationTester


def get_data():
  data_path = os.path.join(os.path.pardir, os.path.pardir, "_00_data", "j_cifar10")

  cifar10_test_images = datasets.CIFAR10(data_path, train=True, download=True)

  cifar10_test = datasets.CIFAR10(data_path, train=True, download=False, transform=transforms.ToTensor())
  test_data_loader = DataLoader(dataset=cifar10_test, batch_size=len(cifar10_test))

  cifar10_transforms = nn.Sequential(
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=0.1307, std=0.3081),
    nn.Flatten(),
  )

  return cifar10_test_images, test_data_loader, cifar10_transforms


def main():
  cifar10_test_images, test_data_loader, cifar10_transforms = get_data()

  test_model = get_model()
  classification_tester = ClassificationTester("cifar10", test_model, test_data_loader, cifar10_transforms)
  classification_tester.test()

  print()

  img, label = cifar10_test_images[0]
  print("     LABEL:", label)
  plt.imshow(img)
  plt.show()

  output = classification_tester.test_single(
    torch.tensor(np.array(cifar10_test_images[0][0])).unsqueeze(dim=0)
  )
  print("PREDICTION:", output)


if __name__ == "__main__":
  main()
