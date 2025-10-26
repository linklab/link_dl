import numpy as np
import torch
import os

from matplotlib import pyplot as plt
from pathlib import Path

BASE_PATH = str(Path(__file__).resolve().parent.parent.parent.parent)  # BASE_PATH: /Users/yhhan/git/link_dl
CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_FILE_PATH = os.path.join(CURRENT_FILE_PATH, "checkpoints")

import sys
sys.path.append(BASE_PATH)

from _01_code._06_fcn_best_practice.d_tester import ClassificationTester
from _01_code._06_fcn_best_practice.i_cifar10_test_fcn import get_cifar10_test_data
from _01_code._09_modern_cnns._01_vgg.a_cifar10_train_vgg import get_vgg_model


def main():
  cifar10_test_images, test_data_loader, cifar10_transforms = get_cifar10_test_data(flatten=False)

  test_model = get_vgg_model()

  project_name = "modern_cifar10"
  classification_tester = ClassificationTester(
    project_name + "_vgg", test_model, test_data_loader, cifar10_transforms, CHECKPOINT_FILE_PATH
  )
  classification_tester.test()

  print()

  img, label = cifar10_test_images[0]
  print("     LABEL:", label)
  plt.imshow(img)
  plt.show()

  # torch.tensor(np.array(cifar10_test_images[0][0])).permute(2, 0, 1).unsqueeze(dim=0).shape: (1, 3, 32, 32)
  output = classification_tester.test_single(
    torch.tensor(np.array(cifar10_test_images[0][0])).permute(2, 0, 1).unsqueeze(dim=0)
  )
  print("PREDICTION:", output)


if __name__ == "__main__":
  main()
