# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

import torch
import os
from pathlib import Path
from PIL import Image

BASE_PATH = str(Path(__file__).resolve().parent.parent.parent) # BASE_PATH: /Users/yhhan/git/link_dl
import sys
sys.path.append(BASE_PATH)

CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_FILE_PATH = os.path.join(CURRENT_FILE_PATH, "checkpoints")

from _01_code._12_transfer_learning.a_train_transfer_learning import get_model, imshow, get_new_data, data_transforms


def visualize_model_prediction_single(data_transforms, class_names, model, img_path, img_label):
    model.eval()

    img = Image.open(img_path)
    img = data_transforms['val'](img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)

        imshow(
            img.cpu().data[0],
            label="Prediction: {0} - Label: {1}".format(
                class_names[preds.item()], img_label
            )
        )


def main(method):
    device = torch.device("cpu")

    image_datasets, dataset_sizes, dataloaders, class_names = get_new_data()

    test_model = get_model(method, device)

    best_model_params_path = os.path.join(CHECKPOINT_FILE_PATH, 'best_model_params.pt')

    print("MODEL FILE: {0}".format(best_model_params_path))
    test_model.load_state_dict(torch.load(best_model_params_path, map_location=torch.device('cpu')))

    visualize_model_prediction_single(
        data_transforms, class_names, test_model,
        img_path=os.path.join(BASE_PATH, "_00_data", "l_transfer_learning_data", "val", "bees", "26589803_5ba7000313.jpg"),
        img_label="bees"
    )


if __name__ == "__main__":
    method_idx = 0
    methods = [
        "frozen_and_train_new_classifier",
        "fine_tune_the_whole_model"
    ]
    main(methods[method_idx])
