"""
Main file for training Yolo model on Pascal VOC dataset

"""

import sys
import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision.transforms import Compose
import time
import os
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import VOCDataset
from utils import (
  non_max_suppression,
  cellboxes_to_boxes,
  plot_image,
  load_model,
)
from loss import YoloLoss

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(CURRENT_PATH, os.pardir, os.pardir))
if PROJECT_HOME not in sys.path:
  sys.path.append(PROJECT_HOME)

seed = 123
torch.manual_seed(seed)

# Hyperparameters etc. 
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LOAD_MODEL_FILE = os.path.join(
  PROJECT_HOME, "_08_object_detection", "YOLO", "yolo_v1.tar"
)
IMG_DIR = os.path.join(os.path.pardir, os.path.pardir, os.path.pardir, "_00_data", "h_yolo_voc_data", "images")
LABEL_DIR = os.path.join(os.path.pardir, os.path.pardir, os.path.pardir, "_00_data", "h_yolo_voc_data", "labels")
TEST_DATASET_CSV = os.path.join(os.path.pardir, os.path.pardir, os.path.pardir, "_00_data", "h_yolo_voc_data", "test.csv")

transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(), ])


def main():
  model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)

  load_model(torch.load(LOAD_MODEL_FILE), model)

  test_dataset = VOCDataset(
    csv_file=TEST_DATASET_CSV, transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR,
  )

  test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=1,
    num_workers=1,
    shuffle=True,
    drop_last=True,
  )

  for x, labels in test_loader:
    x = x.to(DEVICE)
    bboxes = cellboxes_to_boxes(model(x))
    for idx in range(len(bboxes)):
      bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
      plot_image(x[0].permute(1, 2, 0).to("cpu"), bboxes)


if __name__ == "__main__":
  main()
