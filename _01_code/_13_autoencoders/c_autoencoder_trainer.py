from datetime import datetime
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn


from _01_code._06_fcn_best_practice.c_trainer import EarlyStopping
from _01_code._99_common_utils.utils import strfdelta


class AutoencoderTrainer:
  def __init__(
    self, project_name, model, optimizer, train_data_loader, validation_data_loader, transforms,
    run_time_str, wandb, device, checkpoint_file_path, test_dataset, test_transforms, denoising=True
  ):
    self.project_name = project_name
    self.model = model
    self.optimizer = optimizer
    self.train_data_loader = train_data_loader
    self.validation_data_loader = validation_data_loader
    self.transforms = transforms
    self.run_time_str = run_time_str
    self.wandb = wandb
    self.device = device
    self.checkpoint_file_path = checkpoint_file_path

    self.test_dataset = test_dataset
    self.test_transforms = test_transforms
    self.denoising = denoising

    # Use a built-in loss function
    self.loss_fn = nn.MSELoss()

  def add_noise(self, inputs, noise_factor=0.1):
    noisy = inputs + torch.randn(inputs.size()) * noise_factor
    noisy = torch.clip(noisy, 0., 1.)
    return noisy

  def do_train(self):
    self.model.train()

    loss_train = 0.0
    num_trains = 0

    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for train_batch in self.train_data_loader:
      # with "_" we just ignore the target labels
      input_train, _ = train_batch

      if self.denoising is True:
        input_train = self.add_noise(input_train)

      input_train = input_train.to(device=self.device)

      if self.transforms:
        input_train = self.transforms(input_train)

      decoded_input_train = self.model(input_train)

      loss = self.loss_fn(decoded_input_train, input_train)

      loss_train += loss.item()

      num_trains += 1

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

    train_loss = loss_train / num_trains

    return train_loss

  def do_validation(self):
    self.model.eval()

    loss_validation = 0.0
    num_validations = 0

    with torch.no_grad():
      for validation_batch in self.validation_data_loader:
        input_validation, _ = validation_batch

        if self.denoising is True:
          input_validation = self.add_noise(input_validation)

        input_validation = input_validation.to(device=self.device)

        if self.transforms:
          input_validation = self.transforms(input_validation)

        decoded_input_validation = self.model(input_validation)

        loss_validation += self.loss_fn(decoded_input_validation, input_validation).item()

        num_validations += 1

    validation_loss = loss_validation / num_validations

    return validation_loss

  def plot_denoising_autoencoders_outputs(self, n=10, noise_factor=0.1):
    self.model.eval()

    plt.figure(figsize=(16, 4.5))
    targets = self.test_dataset.targets.numpy()
    t_idx = {i: np.where(targets == i)[0][0] for i in range(n)}
    for i in range(n):
      ax = plt.subplot(3, n, i + 1)
      img = self.test_dataset.data[t_idx[i]].unsqueeze(0).unsqueeze(0)

      if self.denoising is True:
        image_noisy = self.add_noise(img, noise_factor)
        image_noisy = image_noisy.to(self.device)
      else:
        img = img.type(torch.float)
        img = img.to(self.device)

      with torch.no_grad():
        decoded_img = self.model(image_noisy if self.denoising is True else img)

      plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
      if i == n // 2:
        ax.set_title('Original images')

      if self.denoising is True:
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(image_noisy.cpu().squeeze().numpy(), cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
          ax.set_title('Corrupted images')

      ax = plt.subplot(3, n, i + 1 + n + n)
      plt.imshow(decoded_img.cpu().squeeze().numpy(), cmap='gist_gray')
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
      if i == n // 2:
        ax.set_title('Reconstructed images')

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.7, top=0.9, wspace=0.3, hspace=0.3)
    plt.show(block=False)

  def train_loop(self):
    early_stopping = EarlyStopping(
      patience=self.wandb.config.early_stop_patience,
      delta=self.wandb.config.early_stop_delta,
      project_name=self.project_name,
      checkpoint_file_path=self.checkpoint_file_path,
      run_time_str=self.run_time_str
    )
    n_epochs = self.wandb.config.epochs
    training_start_time = datetime.now()

    for epoch in range(1, n_epochs + 1):
      train_loss = self.do_train()

      if epoch == 1 or epoch % self.wandb.config.validation_intervals == 0:
        validation_loss = self.do_validation()

        elapsed_time = datetime.now() - training_start_time
        epoch_per_second = 1000 * epoch / elapsed_time.microseconds

        message, early_stop = early_stopping.check_and_save(validation_loss, self.model)

        print(
          f"[Epoch {epoch:>3}] "
          f"T_loss: {train_loss:7.5f}, "
          f"V_loss: {validation_loss:7.5f}, "
          f"{message} | "
          f"T_time: {strfdelta(elapsed_time, '%H:%M:%S')}, "
          f"T_speed: {epoch_per_second:4.3f}"
        )

        self.wandb.log({
          "Epoch": epoch,
          "Training loss": train_loss,
          "Validation loss": validation_loss,
          "Training speed (epochs/sec.)": epoch_per_second,
        })

        self.plot_denoising_autoencoders_outputs(n=10, noise_factor=0.3)

        if early_stop:
          break

    elapsed_time = datetime.now() - training_start_time
    print(f"Final training time: {strfdelta(elapsed_time, '%H:%M:%S')}")
