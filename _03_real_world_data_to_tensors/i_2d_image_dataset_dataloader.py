import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class DogCat2DImageDataset(Dataset):
    def __init__(self):
        self.image_transforms = transforms.Compose([
            transforms.Resize(size=(256, 256)),
            transforms.ToTensor()
        ])

        dogs_dir = os.path.join(os.path.pardir, "_00_data", "a_image-dog")
        cats_dir = os.path.join(os.path.pardir, "_00_data", "b_image-cats")

        self.image_array = [
            Image.open(os.path.join(dogs_dir, "bobby.jpg")),  # (1280, 720, 3)
            Image.open(os.path.join(cats_dir, "cat1.png")),  # (256, 256, 3)
            Image.open(os.path.join(cats_dir, "cat2.png")),  # (256, 256, 3)
            Image.open(os.path.join(cats_dir, "cat3.png"))   # (256, 256, 3)
        ]

        # 0: "dog", 1: "cat"
        self.image_labels = [
            0, 1, 1, 1
        ]

    def __len__(self):
        return len(self.image_array)

    def __getitem__(self, idx):
        image = self.image_transforms(self.image_array[idx])
        label = self.image_labels[idx]
        return {'input': image, 'target': label}


if __name__ == "__main__":
    dog_cat_2d_image_dataset = DogCat2DImageDataset()

    for idx, sample in enumerate(dog_cat_2d_image_dataset):
        print("{0} - {1}: {2}".format(idx, sample['input'].shape, sample['target']))

    dataloader = DataLoader(
        dataset=dog_cat_2d_image_dataset,
        batch_size=2,
        shuffle=True
    )

    print()

    for idx, batch in enumerate(dataloader):
        print("{0} - {1}: {2}".format(idx, batch['input'].shape, batch['target']))

