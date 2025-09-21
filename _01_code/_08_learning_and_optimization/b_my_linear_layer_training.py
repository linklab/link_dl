import torch
from torch import nn, optim
from torch.utils.data import random_split, DataLoader
from _01_code._03_real_world_data_to_tensors.j_linear_regression_dataset_dataloader import LinearRegressionDataset


def get_data():
  linear_regression_dataset = LinearRegressionDataset()
  print(linear_regression_dataset)

  train_dataset, validation_dataset = random_split(linear_regression_dataset, [0.8, 0.2])
  print(len(train_dataset), len(validation_dataset))

  train_data_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)
  validation_data_loader = DataLoader(dataset=validation_dataset, batch_size=len(validation_dataset))

  return train_data_loader, validation_data_loader


def get_model_and_optimizer():
  linear_model = nn.Linear(2, 1)

  print(linear_model.weight)
  print(linear_model.bias)

  print("#" * 50, 2)

  optimizer = optim.SGD(linear_model.parameters(), lr=1e-2)

  print(linear_model.parameters())

  for idx, parameter in enumerate(linear_model.parameters()):
    print(idx, parameter.data, parameter.data.shape, parameter.requires_grad)

  return linear_model, optimizer


def training_loop(model, optimizer, train_data_loader, validation_data_loader):
  n_epochs = 1500
  loss_fn = nn.MSELoss()  # Use a built-in loss function

  for epoch in range(1, n_epochs + 1):
    loss_train = 0.0
    num_trains = 0
    for idx, train_batch in enumerate(train_data_loader):
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
      for idx, validation_batch in enumerate(validation_data_loader):
        input, target = validation_batch
        output_validation = model(input)

        loss = loss_fn(output_validation, target)
        loss_validation += loss.item()
        num_validations += 1

    print(
      f"Epoch {epoch}, "
      f"Training loss {loss_train / num_trains:.4f}, "
      f"Validation loss {loss_validation / num_validations:.4f}"
    )


if __name__ == "__main__":
  train_data_loader, validation_data_loader = get_data()

  print("#" * 50, 1)

  linear_model, optimizer = get_model_and_optimizer()

  print("#" * 50, 3)

  training_loop(
    model=linear_model,
    optimizer=optimizer,
    train_data_loader=train_data_loader,
    validation_data_loader=validation_data_loader
  )
