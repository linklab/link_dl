import torch

from _01_code._06_fcn_best_practice.c_trainer import ClassificationTrainer


class ClassificationTrainerWithImageAugmentation(ClassificationTrainer):
  def __init__(
    self, project_name, model, optimizer, train_data_loader, validation_data_loader, transforms, transforms_train,
    run_time_str, wandb, device, checkpoint_file_path
  ):
    super().__init__(
      project_name, model, optimizer, train_data_loader, validation_data_loader, transforms,
      run_time_str, wandb, device, checkpoint_file_path
    )
    self.transforms_train = transforms_train

  def do_train(self):
    self.model.train()  # Explained at 'Diverse Techniques' section

    loss_train = 0.0
    num_corrects_train = 0
    num_trained_samples = 0
    num_trains = 0

    for train_batch in self.train_data_loader:
      input_train, target_train = train_batch
      input_train = input_train.to(device=self.device)
      target_train = target_train.to(device=self.device)

      if self.transforms:
        input_train = self.transforms(input_train)

      if self.transforms_train:
        input_train = self.transforms_train(input_train)

      output_train = self.model(input_train)

      loss = self.loss_fn(output_train, target_train)
      loss_train += loss.item()

      predicted_train = torch.argmax(output_train, dim=1)
      num_corrects_train += torch.sum(torch.eq(predicted_train, target_train)).item()

      num_trained_samples += len(input_train)
      num_trains += 1

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

    train_loss = loss_train / num_trains
    train_accuracy = 100.0 * num_corrects_train / num_trained_samples

    return train_loss, train_accuracy
