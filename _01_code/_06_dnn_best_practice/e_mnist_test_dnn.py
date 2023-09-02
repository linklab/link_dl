import torch
import os
import wandb
from torch import nn
from torchvision import transforms
from pathlib import Path

from torch.utils.data import DataLoader

BASE_PATH = str(Path(__file__).resolve().parent.parent.parent)
print("BASE_PATH", BASE_PATH)
CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))

import sys
sys.path.append(BASE_PATH)

from d_mnist_train_dnn import get_model_and_optimizer
from b_tester import ClassificationTester

def main():
  mnist_test = torch.load(os.path.join(CURRENT_FILE_PATH, "checkpoints", "mnist_test_dataset.pt"))
  print("Num Test Samples: ", len(mnist_test))

  test_data_loader = DataLoader(dataset=mnist_test, batch_size=len(mnist_test))

  mnist_transforms = nn.Sequential(
    transforms.Normalize(mean=0.1307, std=0.3081),
    nn.Flatten(),
  )

  test_model, _ = get_model_and_optimizer()
  classification_tester = ClassificationTester("mnist", test_model, test_data_loader, mnist_transforms)
  classification_tester.test()

  wandb.finish()


if __name__ == "__main__":
  main()
