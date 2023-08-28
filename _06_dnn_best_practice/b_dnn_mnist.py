import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision import datasets, transforms
from datetime import datetime
import wandb

class MnistTrain:
  def __init__(self, use_wandb):
    current_time_str = datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S')

    config = {
      'epochs': 10_000,
      'learning_rate': 1e-3,
      'batch_size': 256,
      'n_hidden_unit_list': [128, 128],
    }

    wandb.init(
      mode="online" if use_wandb else "disabled",
      project="dnn_mnist",
      notes="mnist experiment",
      tags=["dnn", "mnist"],
      name=current_time_str,
      config=config
    )

    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on device {self.device}.")

    self.num_train_data, self.train_data_loader, self.num_validation_data, self.validation_data_loader \
      = self.get_data_flattened()

    self.model, self.optimizer = self.get_model_and_optimizer()
    wandb.watch(self.model)

  def get_data_flattened(self):
    data_path = '../_00_data/i_mnist/'

    # input.shape: torch.Size([-1, 1, 28, 28]) --> torch.Size([-1, 784])
    transformed_mnist_train = datasets.MNIST(
      data_path, train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.1307, std=0.3081),
        T.Lambda(lambda x: torch.flatten(x))
      ])
    )

    transformed_mnist_validation = datasets.MNIST(
      data_path, train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.1307, std=0.3081),
        T.Lambda(lambda x: torch.flatten(x))
      ])
    )

    train_data_loader = DataLoader(
      dataset=transformed_mnist_train, batch_size=wandb.config.batch_size, shuffle=True, pin_memory=True
    )
    validation_data_loader = DataLoader(
      dataset=transformed_mnist_validation, batch_size=wandb.config.batch_size, pin_memory=True
    )

    return len(transformed_mnist_train), train_data_loader, len(transformed_mnist_validation), validation_data_loader

  def get_model_and_optimizer(self):
    class MyModel(nn.Module):
      def __init__(self, n_input, n_output):
        super().__init__()

        self.model = nn.Sequential(
          nn.Linear(n_input, wandb.config.n_hidden_unit_list[0]),
          nn.Sigmoid(),
          nn.Linear(wandb.config.n_hidden_unit_list[0], wandb.config.n_hidden_unit_list[1]),
          nn.Sigmoid(),
          nn.Linear(wandb.config.n_hidden_unit_list[1], n_output),
        )

      def forward(self, x):
        x = self.model(x)
        return x

    # 1 * 28 * 28 = 784
    my_model = MyModel(n_input=784, n_output=10).to(self.device)
    optimizer = optim.SGD(my_model.parameters(), lr=wandb.config.learning_rate)

    return my_model, optimizer

  def do_train_data(self, loss_fn):
    loss_train = 0.0
    num_corrects_train = 0
    num_trains = 0
    for train_batch in self.train_data_loader:
      input_train, target_train = train_batch
      input_train = input_train.to(device=self.device)
      target_train = target_train.to(device=self.device)

      output_train = self.model(input_train)
      loss = loss_fn(output_train, target_train)
      loss_train += loss.item()

      predicted_train = torch.argmax(output_train, dim=1)
      num_corrects_train += torch.sum(predicted_train == target_train)

      num_trains += 1

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

    train_loss = loss_train / num_trains
    train_accuracy = num_corrects_train / self.num_train_data

    return train_loss, train_accuracy

  def do_validation_data(self, loss_fn):
    loss_validation = 0.0
    num_corrects_validation = 0
    num_validations = 0
    with torch.no_grad():
      for validation_batch in self.validation_data_loader:
        input_validation, target_validation = validation_batch
        input_validation = input_validation.to(device=self.device)
        target_validation = target_validation.to(device=self.device)

        output_validation = self.model(input_validation)
        loss_validation += loss_fn(output_validation, target_validation).item()

        predicted_validation = torch.argmax(output_validation, dim=1)
        num_corrects_validation += torch.sum(predicted_validation == target_validation)

        num_validations += 1

    validation_loss = loss_validation / num_validations
    validation_accuracy = num_corrects_validation / self.num_validation_data

    return validation_loss, validation_accuracy

  def train_loop(self):
    n_epochs = wandb.config.epochs
    loss_fn = nn.CrossEntropyLoss()  # Use a built-in loss function

    for epoch in range(1, n_epochs + 1):
      train_loss, train_accuracy = self.do_train_data(loss_fn)
      validation_loss, validation_accuracy = self.do_validation_data(loss_fn)

      if epoch == 1 or epoch % 10 == 0:
        print(
          f"[Epoch {epoch}] "
          f"Training loss: {train_loss:.4f}, "
          f"Training accuracy: {train_accuracy:.4f} | "
          f"Validation loss: {validation_loss:.4f}, "
          f"Validation accuracy: {validation_accuracy:.4f}"
        )

      wandb.log({
        "Epoch": epoch,
        "Training loss": train_loss,
        "Training accuracy": train_accuracy,
        "Validation loss": validation_loss,
        "Validation accuracy": validation_accuracy,
      })

    wandb.finish()


def main(use_wandb):
  mnist_train = MnistTrain(use_wandb)
  mnist_train.train_loop()


if __name__ == "__main__":
  main(use_wandb=True)
