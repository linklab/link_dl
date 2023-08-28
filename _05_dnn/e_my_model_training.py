import torch
from torch import nn, optim
from torch.utils.data import random_split, DataLoader
from _03_real_world_data_to_tensors.k_california_housing_dataset_dataloader import CaliforniaHousingDataset


def get_data():
  california_housing_dataset = CaliforniaHousingDataset()
  print(california_housing_dataset)

  train_dataset, validation_dataset = random_split(california_housing_dataset, [0.8, 0.2])
  print(len(train_dataset), len(validation_dataset))

  train_data_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
  validation_data_loader = DataLoader(dataset=validation_dataset, batch_size=len(validation_dataset))

  return train_data_loader, validation_data_loader


class MyModel(nn.Module):
  def __init__(self, n_input, n_hidden_unit_list, n_output):
    super().__init__()

    self.model = nn.Sequential(
      nn.Linear(n_input, n_hidden_unit_list[0]),
      nn.ReLU(),
      nn.Linear(n_hidden_unit_list[0], n_hidden_unit_list[1]),
      nn.ReLU(),
      nn.Linear(n_hidden_unit_list[1], n_output),
    )

  def forward(self, x):
    x = self.model(x)
    return x


def get_model_and_optimizer():
  my_model = MyModel(n_input=8, n_hidden_unit_list=[20, 20], n_output=1)
  optimizer = optim.SGD(my_model.parameters(), lr=1e-3)

  return my_model, optimizer


def training_loop(model, optimizer, train_data_loader, validation_data_loader):
  n_epochs = 10000
  loss_fn = nn.MSELoss()  # Use a built-in loss function

  for epoch in range(1, n_epochs + 1):
    loss_train = 0.0
    num_train_samples = 0
    for idx, train_batch in enumerate(train_data_loader):
      output_batch = model(train_batch['input'])
      loss = loss_fn(output_batch, train_batch['target'])
      loss_train += loss.item()
      num_train_samples += len(train_batch)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    loss_validation = 0.0
    num_validation_samples = 0
    with torch.no_grad():
      for idx, validation_batch in enumerate(validation_data_loader):
        output_batch = model(validation_batch['input'])
        loss_validation += loss_fn(output_batch, validation_batch['target']).item()
        num_validation_samples += len(validation_batch)

    if epoch == 1 or epoch % 10 == 0:
      print(
        f"Epoch {epoch}, "
        f"Training loss {loss_train / num_train_samples:.4f}, "
        f"Validation loss {loss_validation / num_validation_samples:.4f}"
      )


if __name__ == "__main__":
  train_data_loader, validation_data_loader = get_data()

  linear_model, optimizer = get_model_and_optimizer()

  print("#" * 50, 1)

  training_loop(
    model=linear_model,
    optimizer=optimizer,
    train_data_loader=train_data_loader,
    validation_data_loader=validation_data_loader
  )

