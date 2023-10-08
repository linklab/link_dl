import torch

from _01_code._06_fcn_best_practice.c_trainer import ClassificationTrainer


class GoogLeNetClassificationTrainer(ClassificationTrainer):
  def __init__(
    self, project_name, model, optimizer, train_data_loader, validation_data_loader, transforms,
    run_time_str, wandb, device, checkpoint_file_path
  ):
    super(GoogLeNetClassificationTrainer, self).__init__(
      project_name, model, optimizer, train_data_loader, validation_data_loader, transforms,
      run_time_str, wandb, device, checkpoint_file_path
    )

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

      input_train = self.transforms(input_train)

      output_train, output_train_ax_1, output_train_ax_2 = self.model(input_train)
      loss = self.loss_fn(output_train, target_train)
      loss_aux_1 = self.loss_fn(output_train_ax_1, target_train)
      loss_aux_2 = self.loss_fn(output_train_ax_2, target_train)
      loss += 0.3 * (loss_aux_1 + loss_aux_2)
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

  def do_validation(self):
    self.model.eval()   # Explained at 'Diverse Techniques' section

    loss_validation = 0.0
    num_corrects_validation = 0
    num_validated_samples = 0
    num_validations = 0

    with torch.no_grad():
      for validation_batch in self.validation_data_loader:
        input_validation, target_validation = validation_batch
        input_validation = input_validation.to(device=self.device)
        target_validation = target_validation.to(device=self.device)

        input_validation = self.transforms(input_validation)

        output_validation, output_validation_ax_1, output_validation_ax_2 = self.model(input_validation)
        loss_validation = self.loss_fn(output_validation, target_validation)
        loss_validation_aux_1 = self.loss_fn(output_validation_ax_1, target_validation)
        loss_validation_aux_2 = self.loss_fn(output_validation_ax_2, target_validation)
        loss_validation += 0.3 * (loss_validation_aux_1 + loss_validation_aux_2)
        loss_validation += loss_validation.item()

        predicted_validation = torch.argmax(output_validation, dim=1)
        num_corrects_validation += torch.sum(torch.eq(predicted_validation, target_validation)).item()

        num_validated_samples += len(input_validation)
        num_validations += 1

    validation_loss = loss_validation / num_validations
    validation_accuracy = 100.0 * num_corrects_validation / num_validated_samples

    return validation_loss, validation_accuracy