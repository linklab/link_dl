import os

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


class DogCat2DImageDataset(Dataset):
  def __init__(self):
    self.image_transforms = transforms.Compose([
      transforms.Resize(size=(256, 256)),
      transforms.ToTensor()
    ])

    dogs_dir = os.path.join(os.path.pardir, os.path.pardir, "_00_data", "a_image-dog")
    cats_dir = os.path.join(os.path.pardir, os.path.pardir, "_00_data", "b_image-cats")

    image_lst = [
      Image.open(os.path.join(dogs_dir, "bobby.jpg")),  # (1280, 720, 3)
      Image.open(os.path.join(cats_dir, "cat1.png")),  # (256, 256, 3)
      Image.open(os.path.join(cats_dir, "cat2.png")),  # (256, 256, 3)
      Image.open(os.path.join(cats_dir, "cat3.png"))  # (256, 256, 3)
    ]

    image_lst = [self.image_transforms(img) for img in image_lst]
    self.images = torch.stack(image_lst, dim=0)

    # 0: "dog", 1: "cat"
    self.image_labels = torch.tensor([[0], [1], [1], [1]])

  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    return self.images[idx], self.image_labels[idx]

  def __str__(self):
    str = "Data Size: {0}, Input Shape: {1}, Target Shape: {2}".format(
      len(self.images), self.images.shape, self.image_labels.shape
    )
    return str


if __name__ == "__main__":
  dog_cat_2d_image_dataset = DogCat2DImageDataset()

  print(dog_cat_2d_image_dataset)

  print("#" * 50, 1)

  for idx, sample in enumerate(dog_cat_2d_image_dataset):
    input, target = sample
    print("{0} - {1}: {2}".format(idx, input.shape, target))

  train_dataset, test_dataset = random_split(dog_cat_2d_image_dataset, [0.7, 0.3])

  print("#" * 50, 2)

  print(len(train_dataset), len(test_dataset))

  print("#" * 50, 3)

  train_data_loader = DataLoader(
    dataset=train_dataset,
    batch_size=2,
    shuffle=True
  )

  for idx, batch in enumerate(train_data_loader):
    input, target = batch
    print("{0} - {1}: {2}".format(idx, input.shape, target))
