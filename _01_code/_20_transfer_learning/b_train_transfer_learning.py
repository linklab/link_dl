# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchinfo import summary
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from pathlib import Path

BASE_PATH = str(Path(__file__).resolve().parent.parent.parent)  # BASE_PATH: /Users/yhhan/git/link_dl
import sys
sys.path.append(BASE_PATH)

CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_FILE_PATH = os.path.join(CURRENT_FILE_PATH, "checkpoints")

if not os.path.isdir(CHECKPOINT_FILE_PATH):
  os.makedirs(os.path.join(CURRENT_FILE_PATH, "checkpoints"))

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # RandomResizedCrop(224): 매번 랜덤 위치/스케일로 잘라서 224×224로 구성
    # RandomHorizontalFlip(): 좌우 반전을 랜덤으로 넣어서 데이터 다양성을 늘림(특정 방향성에 과적합 방지)

    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # Resize(256) + CenterCrop(224): 항상 동일한 규칙으로 크기 맞추고 가운데를 자름
    # 랜덤성이 없어서 매번 같은 입력이 들어가므로 검증 정확도가 재현 가능하고, “모델 성능 비교”가 공정해짐.

}


def get_new_data():
    new_data_path = os.path.join(BASE_PATH, "_00_data", "l_transfer_learning_data")
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(new_data_path, x), data_transforms[x])
        for x in ['train', 'val']
    }

    dataset_sizes = {
        x: len(image_datasets[x]) for x in ['train', 'val']
    }

    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=0)
        for x in ['train', 'val']
    }

    class_names = image_datasets['train'].classes  # class_names <-- ['ants', 'bees']

    return image_datasets, dataset_sizes, dataloaders, class_names


def train_model(
    dataloaders, dataset_sizes, model, loss_fn, optimizer, scheduler, num_epochs=25, device=torch.device("cpu")
):
    since = time.time()

    best_model_params_path = os.path.join(CHECKPOINT_FILE_PATH, 'best_model_params.pt')

    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)  # inputs.shape: [4, 3, 224, 224], outputs.shape: [4, 2]
                    _, preds = torch.max(outputs, dim=1)
                    loss = loss_fn(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                print(f'TRAIN: Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, LR: {scheduler.get_last_lr()}')
            else:
                print(f'  VAL: Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_params_path)

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}\n')

    # load best model weights
    model.load_state_dict(torch.load(best_model_params_path))
    return model


def imshow(input, label=None):
    """Display image for Tensor."""
    input = input.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input = std * input + mean
    input = np.clip(input, 0, 1)
    plt.imshow(input)

    if label is not None:
        plt.title(label)

    plt.show()


def visualize_model(dataloaders, class_names, model, num_images=6):
    model.eval()

    images_so_far = 0

    for i, (inputs, labels) in enumerate(dataloaders['val']):
        outputs = model(inputs)
        _, preds = torch.max(outputs, dim=1)

        for j in range(inputs.size()[0]):
            imshow(
                inputs.data[j], label="Prediction: {0} - Label: {1}".format(
                    class_names[preds[j].item()], class_names[labels[j].item()]
                )
            )
            images_so_far += 1
            if images_so_far == num_images:
                return


def get_model(method, device=torch.device("cpu")):
    model_ft = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    if method == "frozen_and_train_new_classifier":
        for param in model_ft.parameters():
            param.requires_grad = False

    summary(model_ft, input_size=(1, 3, 224, 224))
    print("#" * 100)

    # Here the size of each output sample is set to 2.
    model_ft.fc = nn.Linear(in_features=model_ft.fc.in_features, out_features=2)

    summary(model_ft, input_size=(1, 3, 224, 224))
    print("#" * 100)

    model_ft = model_ft.to(device)

    return model_ft


def main(method):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image_datasets, dataset_sizes, dataloaders, class_names = get_new_data()

    model_ft = get_model(method, device)

    loss_fn = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(
        dataloaders, dataset_sizes, model_ft, loss_fn, optimizer_ft, exp_lr_scheduler,
        num_epochs=25, device=device
    )

    visualize_model(dataloaders, class_names, model_ft)


if __name__ == "__main__":
    method_idx = 0
    methods = [
        "frozen_and_train_new_classifier",
        "fine_tune_the_whole_model"
    ]
    main(methods[method_idx])