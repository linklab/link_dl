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
  mean_average_precision,
  get_bboxes,
  save_checkpoint,
  load_checkpoint,
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

BATCH_SIZE = 32  # 64 in original paper but I don't have that much vram, grad accum?
WEIGHT_DECAY = 0
EPOCHS = 1000
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False

LOAD_MODEL_FILE = os.path.join(
  PROJECT_HOME, "_08_object_detection", "YOLO", "yolo_v1.tar"
)
IMG_DIR = os.path.join(os.path.pardir, os.path.pardir, "_00_data", "h_yolo_voc_data", "images")
LABEL_DIR = os.path.join(os.path.pardir, os.path.pardir, "_00_data", "h_yolo_voc_data", "labels")
TRAIN_DATASET_CSV = os.path.join(os.path.pardir, os.path.pardir, "_00_data", "h_yolo_voc_data", "train.csv")
# TRAIN_DATASET_CSV = os.path.join(os.path.pardir, os.path.pardir, "_00_data", "h_yolo_voc_data", "train_8examples.csv")
# TRAIN_DATASET_CSV = os.path.join(os.path.pardir, os.path.pardir, "_00_data", "h_yolo_voc_data", "train_100examples.csv")

transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(), ])


def train_fn(train_loader, model, optimizer, loss_fn):
  mean_loss = []

  for batch_idx, (x, y) in enumerate(train_loader):
    x, y = x.to(DEVICE), y.to(DEVICE)
    out = model(x)
    loss = loss_fn(out, y)
    mean_loss.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  print(f"Mean loss was {sum(mean_loss) / len(mean_loss)}")
  print()


def main():
  model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
  optimizer = optim.Adam(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
  )
  loss_fn = YoloLoss()

  if LOAD_MODEL:
    load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

  train_dataset = VOCDataset(
    csv_file=TRAIN_DATASET_CSV, transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR,
  )

  train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    shuffle=True,
    drop_last=True,
  )

  for epoch in range(EPOCHS):
    pred_boxes, target_boxes = get_bboxes(
      train_loader, model, iou_threshold=0.5, threshold=0.4, device=DEVICE
    )

    mean_avg_prec = mean_average_precision(
      pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
    )
    print(f"[EPOCH: {epoch}] Train mAP: {mean_avg_prec}")

    if mean_avg_prec > 0.9:
      checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
      }
      save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
      time.sleep(3)

    train_fn(train_loader, model, optimizer, loss_fn)


if __name__ == "__main__":
  main()
