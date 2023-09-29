from datetime import datetime
import os

import numpy as np
import torch
import wandb
from torch import nn

from _01_code._99_common_utils.utils import strfdelta


class ClassificationTrainerWithDiverseOptimizers:
  def __init__(
    self, project_name, models, optimizers, optimizer_names, train_data_loader, validation_data_loader, transforms,
    run_time_str, wandb, device, checkpoint_file_path
  ):
    self.project_name = project_name
    self.models = models
    self.optimizers = optimizers
    self.optimizer_names = optimizer_names
    self.train_data_loader = train_data_loader
    self.validation_data_loader = validation_data_loader
    self.transforms = transforms
    self.run_time_str = run_time_str
    self.wandb = wandb
    self.device = device
    self.checkpoint_file_path = checkpoint_file_path

    # Use a built-in loss function
    self.loss_fn = nn.CrossEntropyLoss()

  def do_train(self):
    loss_trains = np.zeros(shape=(len(self.models, )))
    num_corrects_trains = np.zeros(shape=(len(self.models),))
    num_trained_samples = 0
    num_trains = 0

    for train_batch in self.train_data_loader:
      input_train, target_train = train_batch
      input_train = input_train.to(device=self.device)
      target_train = target_train.to(device=self.device)

      input_train = self.transforms(input_train)

      for idx, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
        output_train = model(input_train)
        loss = self.loss_fn(output_train, target_train)
        loss_trains[idx] += loss.item()

        predicted_train = torch.argmax(output_train, dim=1)
        num_corrects_trains[idx] += torch.sum(torch.eq(predicted_train, target_train)).item()

        num_trained_samples += len(input_train)
        num_trains += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_losses = loss_trains / num_trains / len(self.models)
    train_accuracies = 100.0 * num_corrects_trains / num_trained_samples / len(self.models)

    return train_losses, train_accuracies

  def do_validation(self):
    loss_validations = np.zeros(shape=(len(self.models, )))
    num_corrects_validations = np.zeros(shape=(len(self.models),))
    num_validated_samples = 0
    num_validations = 0

    with torch.no_grad():
      for validation_batch in self.validation_data_loader:
        input_validation, target_validation = validation_batch
        input_validation = input_validation.to(device=self.device)
        target_validation = target_validation.to(device=self.device)

        input_validation = self.transforms(input_validation)

        for idx, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
          output_validation = model(input_validation)
          loss_validations[idx] += self.loss_fn(output_validation, target_validation).item()

          predicted_validation = torch.argmax(output_validation, dim=1)
          num_corrects_validations[idx] += torch.sum(torch.eq(predicted_validation, target_validation)).item()

          num_validated_samples += len(input_validation)
          num_validations += 1

    validation_losses = loss_validations / num_validations / len(self.models)
    validation_accuracies = 100.0 * num_corrects_validations / num_validated_samples / len(self.models)

    return validation_losses, validation_accuracies

  def train_loop(self):
    n_epochs = self.wandb.config.epochs
    training_start_time = datetime.now()

    x_series = []

    train_loss_series = [[] for _ in self.models]
    train_accuracy_series = [[] for _ in self.models]
    validation_loss_series = [[] for _ in self.models]
    validation_accuracy_series = [[] for _ in self.models]

    for epoch in range(1, n_epochs + 1):
      train_losses, train_accuracies = self.do_train()

      if epoch == 1 or epoch % self.wandb.config.validation_intervals == 0:
        validation_losses, validation_accuracies = self.do_validation()

        elapsed_time = datetime.now() - training_start_time
        epoch_per_second = epoch / elapsed_time.seconds

        train_losses_str = str(
          [f"{train_loss:6.3f}" for train_loss in train_losses]
        )
        train_accuracies_str = str(
          [f"{train_accuracy:6.3f}" for train_accuracy in train_accuracies]
        )
        validation_losses_str = str(
          [f"{validation_loss:6.3f}" for validation_loss in validation_losses]
        )
        validation_accuracies_str = str(
          [f"{validation_accuracy:6.3f}" for validation_accuracy in validation_accuracies]
        )

        print(
          f"[Epoch {epoch:>3}] "
          f"T_loss: {train_losses_str}, "
          f"T_accuracy: {train_accuracies_str} | "
          f"V_loss: {validation_losses_str}, "
          f"V_accuracy: {validation_accuracies_str} | "
          f"T_time: {strfdelta(elapsed_time, '%H:%M:%S')}, "
          f"T_speed: {epoch_per_second:4.2f}"
        )

        x_series.append(epoch)

        for idx in range(len(self.models)):
          train_loss_series[idx].append(train_losses[idx])
          train_accuracy_series[idx].append(train_accuracies[idx])
          validation_loss_series[idx].append(validation_losses[idx])
          validation_accuracy_series[idx].append(validation_accuracies[idx])

        self.wandb.log({
          "Epoch": epoch,
          "Training loss": wandb.plot.line_series(
            xs=x_series, ys=train_loss_series
          ),
          "Training accuracy (%)": wandb.plot.line_series(
            xs=x_series, ys=train_accuracy_series
          ),
          "Validation loss": wandb.plot.line_series(
            xs=x_series, ys=validation_loss_series
          ),
          "Validation accuracy (%)": wandb.plot.line_series(
            xs=x_series, ys=validation_accuracy_series
          ),
          "Training speed (epochs/sec.)": epoch_per_second,
        })

    elapsed_time = datetime.now() - training_start_time
    print(f"Final training time: {strfdelta(elapsed_time, '%H:%M:%S')}")
