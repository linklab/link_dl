import torch
from torch import nn, optim
from torch.utils.data import random_split, DataLoader
from datetime import datetime
import wandb
import argparse

from pathlib import Path
BASE_PATH = str(Path(__file__).resolve().parent.parent.parent) # BASE_PATH: /Users/yhhan/git/link_dl
print("BASE_PATH:", BASE_PATH)

import sys
sys.path.append(BASE_PATH)

from _01_code._03_real_world_data_to_tensors.m_california_housing_dataset_dataloader \
  import CaliforniaHousingDataset


def get_data(config):
  california_housing_dataset = CaliforniaHousingDataset()
  print(california_housing_dataset)

  train_dataset, validation_dataset = random_split(california_housing_dataset, [0.8, 0.2])
  print(len(train_dataset), len(validation_dataset))

  train_data_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
  validation_data_loader = DataLoader(dataset=validation_dataset, batch_size=len(validation_dataset))

  return train_data_loader, validation_data_loader


class MyModel(nn.Module):
  def __init__(self, n_input, n_output, config):
    super().__init__()

    self.model = nn.Sequential(
      nn.Linear(n_input, config.n_hidden_unit_list[0]),
      nn.ReLU(),
      nn.Linear(config.n_hidden_unit_list[0], config.n_hidden_unit_list[1]),
      nn.ReLU(),
      nn.Linear(config.n_hidden_unit_list[1], n_output),
    )

  def forward(self, x):
    x = self.model(x)
    return x


def get_model_and_optimizer(config):
  my_model = MyModel(n_input=8, n_output=1, config=config)
  optimizer = optim.SGD(my_model.parameters(), lr=config.learning_rate)

  return my_model, optimizer


def training_loop(model, optimizer, train_data_loader, validation_data_loader, run):
  n_epochs = run.config.epochs
  loss_fn = nn.MSELoss()  # Use a built-in loss function
  next_print_epoch = 100

  for epoch in range(1, n_epochs + 1):
    loss_train = 0.0
    num_trains = 0
    for train_batch in train_data_loader:
      input, target = train_batch
      output_train = model(input)
      loss = loss_fn(output_train, target)
      loss_train += loss.item()
      num_trains += 1

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    loss_validation = 0.0
    num_validations = 0
    with torch.no_grad():
      for validation_batch in validation_data_loader:
        input, target = validation_batch
        output_validation = model(input)
        loss = loss_fn(output_validation, target)
        loss_validation += loss.item()
        num_validations += 1

    run.log({
      "Epoch": epoch,
      "Training loss": loss_train / num_trains,
      "Validation loss": loss_validation / num_validations
    })

    if epoch >= next_print_epoch:
      print(
        f"Epoch {epoch}, "
        f"Training loss {loss_train / num_trains:.4f}, "
        f"Validation loss {loss_validation / num_validations:.4f}"
      )
      next_print_epoch += 100


def main(args):
  current_time_str = datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S')

  config = {
    'epochs': args.epochs,
    'batch_size': args.batch_size,
    'learning_rate': 1e-3,
    'n_hidden_unit_list': [20, 20],
  }

  with wandb.init(
    mode="online" if args.wandb else "disabled",
    project="my_model_training",
    notes="My first wandb experiment",
    tags=["my_model", "california_housing"],
    name=current_time_str,
    config=config
  ) as run:
    print(args)
    print(run.config)

    # 가로축을 Epoch, 세로축을 Training loss로 그려줘.”
    run.define_metric("Training loss", step_metric="Epoch")

    # 가로축을 Epoch, 세로축을 Validation loss로 그려줘, 실험 요약에는 가장 낮았던 값을 남겨줘.”
    run.define_metric("Validation loss", step_metric="Epoch", summary="min")

    train_data_loader, validation_data_loader = get_data(run.config)

    linear_model, optimizer = get_model_and_optimizer(run.config)

    print("#" * 50, 1)

    training_loop(
      model=linear_model,
      optimizer=optimizer,
      train_data_loader=train_data_loader,
      validation_data_loader=validation_data_loader,
      run=run
    )


# https://docs.wandb.ai/models/track/config
if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument(
    "--wandb", action=argparse.BooleanOptionalAction, default=True, help="True or False"
  )

  parser.add_argument(
    "-b", "--batch_size", type=int, default=512, help="Batch size (int, default: 512)"
  )

  parser.add_argument(
    "-e", "--epochs", type=int, default=1_000, help="Number of training epochs (int, default:1_000)"
  )

  args = parser.parse_args()

  main(args)
