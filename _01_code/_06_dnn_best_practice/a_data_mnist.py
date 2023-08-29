import numpy as np
from matplotlib import pyplot as plt
import torch
import os
from torchvision import datasets

data_path = os.path.join(os.path.pardir, os.path.pardir, "_00_data", "i_mnist")
mnist_train = datasets.MNIST(data_path, train=True, download=True)
mnist_validation = datasets.MNIST(data_path, train=False, download=True)

print(len(mnist_train), len(mnist_validation))  # >>> 60000 10000

fig = plt.figure(figsize=(8, 3))
num_classes = 10
for i in range(num_classes):
    ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
    ax.set_title(i)
    img = next((img for img, label in mnist_train if label == i))
    plt.imshow(img)
plt.show()

print("#" * 50, 1)

img, label = mnist_train[0]
print(type(img))                  # >>> <class 'PIL.Image.Image'>
print(label)  # >>> 1 automobile
plt.imshow(img)
plt.show()

img = np.array(img)
print(type(img))                  # >>> <class 'numpy.ndarray'>

print("#" * 50, 2)

from torchvision import transforms
mnist_train = datasets.MNIST(data_path, train=True, download=False, transform=transforms.ToTensor())

img_t, _ = mnist_train[0]
print(type(img_t))
print(img_t.shape)
print(img_t.min(), img_t.max())

print("#" * 50, 3)

imgs = torch.stack([img_t for img_t, _ in mnist_train], dim=3)
print(imgs.shape)
# >>> torch.Size([1, 28, 28, 60000])

print(imgs.view(1, -1).mean(dim=-1))
# >>> tensor([0.1307])

print(imgs.view(1, -1).std(dim=-1))
# >>> tensor([0.3081])

import torchvision.transforms as T

transformed_mnist_train = datasets.MNIST(
  data_path, train=True, download=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=0.1307, std=0.3081),
    T.Lambda(lambda x: torch.flatten(x))
  ])
)

transformed_mnist_validation = datasets.MNIST(
  data_path, train=False, download=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=0.1307, std=0.3081),
    T.Lambda(lambda x: torch.flatten(x))
  ])
)
img_t, _ = transformed_mnist_train[0]
print(img_t.shape)

print(len(transformed_mnist_train))
print(len(transformed_mnist_validation))

