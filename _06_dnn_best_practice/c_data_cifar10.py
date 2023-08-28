import numpy as np
from matplotlib import pyplot as plt
import torch

from torchvision import datasets
data_path = '../_00_data/j_cifar10/'
cifar10_train = datasets.CIFAR10(data_path, train=True, download=True)
cifar10_validation = datasets.CIFAR10(data_path, train=False, download=True)

print(len(cifar10_train), len(cifar10_validation))  # >>> 50000 10000

class_names = [
  'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
]

fig = plt.figure(figsize=(8, 3))
num_classes = 10
for i in range(num_classes):
    ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
    ax.set_title(class_names[i])
    img = next((img for img, label in cifar10_train if label == i))
    plt.imshow(img)
plt.show()

print("#" * 50, 1)

img, label = cifar10_train[0]
print(type(img))                  # >>> <class 'PIL.Image.Image'>
print(label, class_names[label])  # >>> 1 automobile
plt.imshow(img)
plt.show()

img = np.array(img)
print(type(img))                  # >>> <class 'numpy.ndarray'>

print("#" * 50, 2)

from torchvision import transforms
tensor_cifar10_train = datasets.CIFAR10(data_path, train=True, download=False, transform=transforms.ToTensor())

img_t, _ = tensor_cifar10_train[0]
print(type(img_t))
print(img_t.shape)
print(img_t.min(), img_t.max())

print("#" * 50, 3)

imgs = torch.stack([img_t for img_t, _ in tensor_cifar10_train], dim=3)
print(imgs.shape)

print(imgs.view(3, -1).mean(dim=-1))
# >>> tensor([0.4914, 0.4822, 0.4465])

print(imgs.view(3, -1).std(dim=-1))
# >>> tensor([0.2470, 0.2435, 0.2616])

import torchvision.transforms as T

transformed_cifar10_train = datasets.CIFAR10(
  data_path, train=True, download=False, transform=transforms.Compose([
    transforms.ToTensor(), transforms.Normalize(
      mean=(0.4915, 0.4823, 0.4468), std=(0.2470, 0.2435, 0.2616)
    ),
    T.Lambda(lambda x: torch.flatten(x))
  ])
)

transformed_cifar10_validation = datasets.CIFAR10(
  data_path, train=False, download=False, transform=transforms.Compose([
    transforms.ToTensor(), transforms.Normalize(
      mean=(0.4915, 0.4823, 0.4468), std=(0.2470, 0.2435, 0.2616)
    ),
    T.Lambda(lambda x: torch.flatten(x))
  ])
)

img_t, _ = transformed_cifar10_train[0]
print(img_t.shape)

print(len(transformed_cifar10_train))
print(len(transformed_cifar10_validation))
