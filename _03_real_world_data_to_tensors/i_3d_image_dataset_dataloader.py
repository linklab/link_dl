import os

import imageio
from torch.utils.data import Dataset, DataLoader


class Dicom3DImageDataset(Dataset):
    def __init__(self):
        dir_path = os.path.join(os.path.pardir, "_00_data", "c_volumetric-dicom", "2-LUNG_3.0_B70f-04083")
        self.vol_array = imageio.volread(dir_path, format='DICOM')

    def __len__(self):
        return len(self.vol_array)

    def __getitem__(self, idx):
        return self.vol_array[idx]


if __name__ == "__main__":
    dicom_3d_image_dataset = Dicom3DImageDataset()

    for image in dicom_3d_image_dataset:
        print("{0}".format(image.shape))

    # cat_dog_2d_image_dataloader = DataLoader(
    #     dataset=cat_dog_2d_image_dataset,
    #     batch_size=2,
    #     shuffle=True
    # )
    #
    # print()
    #
    # for images, labels in cat_dog_2d_image_dataloader:
    #     print("{0}: {1}".format(images.shape, labels))

