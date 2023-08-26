import torch
from torch import nn, optim


def get_data():
  data_in = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0]
  data_out = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4]

  # Adds the extra dimension at axis 1
  data_in = torch.tensor(data_in).unsqueeze(1)
  data_out = torch.tensor(data_out).unsqueeze(1)

  print(data_in.shape, data_out.shape)

  n_samples = data_out.shape[0]
  n_val = int(0.2 * n_samples)

  shuffled_indices = torch.randperm(n_samples)
  print(shuffled_indices)

  train_indices = shuffled_indices[:-n_val]
  val_indices = shuffled_indices[-n_val:]

  print(train_indices, val_indices)

  data_in_train = data_in[train_indices]
  data_out_train = data_out[train_indices]

  data_in_val = data_in[val_indices]
  data_out_val = data_out[val_indices]

  data_out_train = 0.1 * data_out_train
  data_out_val = 0.1 * data_out_val

  return data_in_train, data_out_train, data_in_val, data_out_val


def training_loop(model, optimizer, data_out_train, data_out_val, data_in_train, data_in_val):
  n_epochs = 3000
  loss_fn = nn.MSELoss()  # Use a built-in loss function

  for epoch in range(1, n_epochs + 1):
    t_p_train = model(data_out_train)
    loss_train = loss_fn(t_p_train, data_in_train)

    t_p_val = model(data_out_val)
    loss_val = loss_fn(t_p_val, data_in_val)

    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()

    if epoch == 1 or epoch % 100 == 0:
      print(
        f"Epoch {epoch}, "
        f"Training loss {loss_train.item():.4f},"
        f"Validation loss {loss_val.item():.4f}"
      )


def get_model_and_optimizer():
  linear_model = nn.Linear(1, 1)

  print(linear_model.weight)
  print(linear_model.bias)

  print("#" * 50, 2)

  optimizer = optim.SGD(linear_model.parameters(), lr=1e-2)

  print(linear_model.parameters())

  for idx, parameter in enumerate(linear_model.parameters()):
    print(idx, parameter.data, parameter.data.shape, parameter.requires_grad)

  return linear_model, optimizer


if __name__ == "__main__":
  data_in_train, data_out_train, data_in_val, data_out_val = get_data()
  print(data_in_train.shape, data_out_train.shape, data_in_val.shape, data_out_val.shape)

  print("#" * 50, 1)

  linear_model, optimizer = get_model_and_optimizer()

  print("#" * 50, 3)

  training_loop(
    model=linear_model,
    optimizer=optimizer,
    data_in_train=data_in_train,
    data_out_train=data_out_train,
    data_out_val=data_out_val,
    data_in_val=data_in_val
  )
