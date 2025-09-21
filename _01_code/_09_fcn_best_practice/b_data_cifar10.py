import numpy as np
from matplotlib import pyplot as plt
import torch
import os

from torch import nn
from torch.utils.data import random_split, DataLoader
from torchvision import datasets

data_path = os.path.join(os.path.pardir, os.path.pardir, "_00_data", "i_cifar10")
cifar10_train_images = datasets.CIFAR10(data_path, train=True, download=True)
cifar10_test_images = datasets.CIFAR10(data_path, train=False, download=True)

print(len(cifar10_train_images), len(cifar10_test_images))  # >>> 50000 10000

class_names = [
  'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
]

fig = plt.figure(figsize=(8, 3))
num_classes = 10
for i in range(num_classes):
    ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
    ax.set_title(class_names[i])
    img = next((img for img, label in cifar10_train_images if label == i))
    plt.imshow(img)
plt.show()

print("#" * 50, 1)

img, label = cifar10_train_images[0]
print(type(img))                  # >>> <class 'PIL.Image.Image'>
print(label, class_names[label])  # >>> 1 automobile
plt.imshow(img)
plt.show()

img = np.array(img)
print(type(img))                  # >>> <class 'numpy.ndarray'>
print(img.shape)

print("#" * 50, 2)

from torchvision import transforms
cifar10_train = datasets.CIFAR10(data_path, train=True, download=False, transform=transforms.ToTensor())
cifar10_train, cifar10_validation = random_split(cifar10_train, [45_000, 5_000])
cifar10_test = datasets.CIFAR10(data_path, train=False, download=False, transform=transforms.ToTensor())

print(len(cifar10_train), len(cifar10_validation), len(cifar10_test))  # >>> 45000 5000 10000

img_t, _ = cifar10_train[0]
print(type(img_t))
print(img_t.shape)
print(img_t.min(), img_t.max())

print("#" * 50, 3)

imgs = torch.stack([img_t for img_t, _ in cifar10_train], dim=3)
print(imgs.shape)

print(imgs.view(3, -1).mean(dim=-1))
# >>> tensor([0.4914, 0.4822, 0.4465])

print(imgs.view(3, -1).std(dim=-1))
# >>> tensor([0.2470, 0.2435, 0.2616])

print("#" * 50, 4)

# input.shape: torch.Size([-1, 3, 32, 32]) --> torch.Size([-1, 3072])
cifar10_transforms = nn.Sequential(
  transforms.ConvertImageDtype(torch.float),
  transforms.Normalize(mean=(0.4915, 0.4823, 0.4468), std=(0.2470, 0.2435, 0.2616)),
  nn.Flatten()
)

train_data_loader = DataLoader(dataset=cifar10_train, batch_size=32, shuffle=True)

for idx, train_batch in enumerate(train_data_loader):
    input, target = train_batch
    transformed_input = cifar10_transforms(input)
    print(input.shape, transformed_input.shape, target.shape)
    print(target)