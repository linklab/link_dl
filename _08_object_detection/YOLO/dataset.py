"""
Creates a Pytorch dataset to load the Pascal VOC dataset
"""

import torch
import os
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import Compose


class VOCDataset(torch.utils.data.Dataset):
    def __init__(
        self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None,
    ):
        self.annotations_csv = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations_csv)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations_csv.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]

                boxes.append([class_label, x, y, width, height])

        img_path = os.path.join(self.img_dir, self.annotations_csv.iloc[index, 0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        if self.transform:
            # image = self.transform(image)
            image = self.transform(image)

        # Convert To Cells
        label_matrix = torch.zeros(size=(self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            # i,j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i

            """
            Calculating the width and height of cell of bounding box,
            relative to the cell is done by the following, with
            width as the example:
            
            width_pixels = (width*self.image_width)
            cell_pixels = (self.image_width)
            
            Then to find the width relative to the cell is simply:
            width_pixels/cell_pixels, simplification leads to the
            formulas below.
            """
            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )

            # If no object already found for specific cell i,j
            # Note: This means we restrict to ONE object per cell!
            if label_matrix[i, j, 20] == 0:
                # Set that there exists an object
                label_matrix[i, j, 20] = 1

                # Box coordinates
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, 21:25] = box_coordinates

                # Set one hot encoding for class_label
                label_matrix[i, j, class_label] = 1

        return image, label_matrix


if __name__ == "__main__":
    import torchvision.transforms as transforms
    transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(), ])

    IMG_DIR = os.path.join(os.path.pardir, os.path.pardir, "_00_data", "h_yolo_voc_data", "images")
    LABEL_DIR = os.path.join(os.path.pardir, os.path.pardir, "_00_data", "h_yolo_voc_data", "labels")
    CSV_FILE = os.path.join(os.path.pardir, os.path.pardir, "_00_data", "h_yolo_voc_data", "100examples.csv")

    train_dataset = VOCDataset(
        csv_file=CSV_FILE,
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=1,
        num_workers=1,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )

    for batch_idx, (x, labels) in enumerate(train_loader):
        print(batch_idx, x.shape, labels.shape, labels[0, 3, 3])

