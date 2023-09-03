import torch
import os
from torch import nn
from torchvision import transforms
from pathlib import Path

from torch.utils.data import DataLoader

BASE_PATH = str(Path(__file__).resolve().parent.parent.parent)
print("BASE_PATH", BASE_PATH)
CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))

import sys
sys.path.append(BASE_PATH)

from g_cifar10_train_dnn import get_model
from b_tester import ClassificationTester


def main():
  cifar10_test = torch.load(os.path.join(CURRENT_FILE_PATH, "checkpoints", "cifar10_test_dataset.pt"))
  print("Num Test Samples: ", len(cifar10_test))

  test_data_loader = DataLoader(dataset=cifar10_test, batch_size=len(cifar10_test))

  cifar10_transforms = nn.Sequential(
    transforms.Normalize(mean=(0.4915, 0.4823, 0.4468), std=(0.2470, 0.2435, 0.2616)),
    nn.Flatten(),
  )

  test_model = get_model()
  classification_tester = ClassificationTester("cifar10", test_model, test_data_loader, cifar10_transforms)
  classification_tester.test()


if __name__ == "__main__":
  main()
