import numpy as np
from matplotlib import pyplot as plt
import torch
import os

from torch import nn
from torch.utils.data import random_split, DataLoader
from torchvision import datasets

data_path = os.path.join(os.path.pardir, os.path.pardir, "_00_data", "h_mnist")
# If train=True, creates dataset from train-images-idx3-ubyte,
# otherwise from t10k-images-idx3-ubyte.
mnist_train_images = datasets.MNIST(data_path, train=True, download=True)
mnist_test_images = datasets.MNIST(data_path, train=False, download=True)

print(len(mnist_train_images), len(mnist_test_images))  # >>> 60000 10000

fig = plt.figure(figsize=(8, 3))
num_classes = 10
for i in range(num_classes):
    ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
    ax.set_title(i)
    img = next((img for img, label in mnist_train_images if label == i))
    plt.imshow(img)
plt.show()

print("#" * 50, 1)

img, label = mnist_train_images[0]
print(type(img))                  # >>> <class 'PIL.Image.Image'>
print(label)  # >>> 5
plt.imshow(img)
plt.show()

img = np.array(img)
print(type(img))
print(img.shape)

print("#" * 50, 2)

from torchvision import transforms
mnist_train = datasets.MNIST(data_path, train=True, download=False, transform=transforms.ToTensor())
mnist_train, mnist_validation = random_split(mnist_train, [55_000, 5_000])
mnist_test = datasets.MNIST(data_path, train=False, download=False, transform=transforms.ToTensor())

print(len(mnist_train), len(mnist_validation), len(mnist_test))  # >>> 55000 5000 10000

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

print("#" * 50, 4)

# >>> 60_000

# input.shape: torch.Size([-1, 1, 28, 28]) --> torch.Size([-1, 784])
mnist_transforms = nn.Sequential(
  transforms.ConvertImageDtype(torch.float),
  transforms.Normalize(mean=0.1307, std=0.3081),
  nn.Flatten(),
)

train_data_loader = DataLoader(dataset=mnist_train, batch_size=32, shuffle=True)

for idx, train_batch in enumerate(train_data_loader):
    input, target = train_batch
    transformed_input = mnist_transforms(input)
    print(input.shape, transformed_input.shape, target.shape)
    print(target)