import numpy as np
import torch
import os

from matplotlib import pyplot as plt
from pathlib import Path

BASE_PATH = str(Path(__file__).resolve().parent.parent.parent)  # BASE_PATH: /Users/yhhan/git/link_dl
CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))

import sys
sys.path.append(BASE_PATH)

from _01_code._07_cnn.a_mnist_train_cnn import get_cnn_model
from _01_code._06_fcn_best_practice.d_tester import ClassificationTester
from _01_code._06_fcn_best_practice.i_cifar10_test_fcn import get_test_data


def main():
  mnist_test_images, test_data_loader, cifar10_transforms = get_test_data(flatten=False)

  test_model = get_cnn_model()
  classification_tester = ClassificationTester("cifar10", test_model, test_data_loader, mnist_transforms)
  classification_tester.test()

  print()

  img, label = mnist_test_images[0]
  print("     LABEL:", label)
  plt.imshow(img)
  plt.show()

  output = classification_tester.test_single(
    torch.tensor(np.array(mnist_test_images[0][0])).unsqueeze(dim=0)
  )
  print("PREDICTION:", output)


if __name__ == "__main__":
  main()
