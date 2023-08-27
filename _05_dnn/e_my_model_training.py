from torch import nn, optim
from torch.utils.data import random_split, DataLoader
from _03_real_world_data_to_tensors.j_tabular_dataset_dataloader import WineDataset


def get_data():
  wine_dataset = WineDataset()
  print(wine_dataset)

  train_dataset, validation_dataset = random_split(wine_dataset, [0.8, 0.2])
  print(len(train_dataset), len(validation_dataset))

  train_data_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
  validation_data_loader = DataLoader(dataset=validation_dataset, batch_size=len(validation_dataset))

  return train_data_loader, validation_data_loader


class MyModel(nn.Module):
  def __init__(self, n_input, n_hidden_unit_lst, n_output):
    super().__init__()

    self.model = nn.Sequential(
      nn.Linear(n_input, n_hidden_unit_lst[0]),
      nn.Sigmoid(),
      nn.Linear(n_hidden_unit_lst[0], n_hidden_unit_lst[1]),
      nn.Sigmoid(),
      nn.Linear(n_hidden_unit_lst[1], n_output),
      nn.Softmax(dim=1)
    )

  def forward(self, x):
    x = self.model(x)
    return x


def get_model_and_optimizer():
  my_model = MyModel(n_input=11, n_hidden_unit_lst=[30, 30], n_output=10)
  optimizer = optim.SGD(my_model.parameters(), lr=1e-2)

  return my_model, optimizer


def training_loop(model, optimizer, train_data_loader, validation_data_loader):
  n_epochs = 10000
  loss_fn = nn.MSELoss()  # Use a built-in loss function

  for epoch in range(1, n_epochs + 1):
    loss_train = 0.0
    for idx, train_batch in enumerate(train_data_loader):
      output_batch = model(train_batch['input'])
      loss = loss_fn(output_batch, train_batch['target'])
      loss_train += loss.item()

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    loss_validation = 0.0
    for idx, validation_batch in enumerate(validation_data_loader):
      output_batch = model(validation_batch['input'])
      loss_validation += loss_fn(output_batch, validation_batch['target']).item()

    if epoch == 1 or epoch % 100 == 0:
      print(
        f"Epoch {epoch}, "
        f"Training loss {loss_train:.4f}, "
        f"Validation loss {loss_validation:.4f}"
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

