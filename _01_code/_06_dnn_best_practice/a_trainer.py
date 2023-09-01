import torch
from torch import nn


class ClassificationTrainer:
    def __init__(self, model, optimizer, train_data_loader, validation_data_loader, wandb, device):
        self.model = model
        self.optimizer = optimizer
        self.train_data_loader = train_data_loader
        self.validation_data_loader = validation_data_loader
        self.wandb = wandb
        self.device = device

        # Use a built-in loss function
        self.loss_fn = nn.CrossEntropyLoss()

    def do_train(self):
        loss_train = 0.0
        num_corrects_train = 0
        num_trained_samples = 0
        num_trains = 0

        for train_batch in self.train_data_loader:
            input_train, target_train = train_batch
            input_train = input_train.to(device=self.device)
            target_train = target_train.to(device=self.device)

            output_train = self.model(input_train)
            loss = self.loss_fn(output_train, target_train)
            loss_train += loss.item()

            predicted_train = torch.argmax(output_train, dim=1)
            num_corrects_train += torch.sum(torch.eq(predicted_train, target_train))

            num_trained_samples += len(train_batch)
            num_trains += 1

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        train_loss = loss_train / num_trains
        train_accuracy = num_corrects_train / num_trained_samples

        return train_loss, train_accuracy

    def do_validation(self):
        loss_validation = 0.0
        num_corrects_validation = 0
        num_validated_samples = 0
        num_validations = 0

        with torch.no_grad():
            for validation_batch in self.validation_data_loader:
                input_validation, target_validation = validation_batch
                input_validation = input_validation.to(device=self.device)
                target_validation = target_validation.to(device=self.device)

                output_validation = self.model(input_validation)
                loss_validation += self.loss_fn(output_validation, target_validation).item()

                predicted_validation = torch.argmax(output_validation, dim=1)
                num_corrects_validation += torch.sum(torch.eq(predicted_validation, target_validation))

                num_validated_samples += len(validation_batch)
                num_validations += 1

        validation_loss = loss_validation / num_validations
        validation_accuracy = num_corrects_validation / num_validated_samples

        return validation_loss, validation_accuracy

    def train_loop(self):
        n_epochs = self.wandb.config.epochs

        for epoch in range(1, n_epochs + 1):
            train_loss, train_accuracy = self.do_train()
            validation_loss, validation_accuracy = self.do_validation()

            if epoch == 1 or epoch % 10 == 0:
                print(
                    f"[Epoch {epoch}] "
                    f"Training loss: {train_loss:.4f}, "
                    f"Training accuracy: {train_accuracy:.4f} | "
                    f"Validation loss: {validation_loss:.4f}, "
                    f"Validation accuracy: {validation_accuracy:.4f}"
                )

            self.wandb.log({
                "Epoch": epoch,
                "Training loss": train_loss,
                "Training accuracy": train_accuracy,
                "Validation loss": validation_loss,
                "Validation accuracy": validation_accuracy,
            })

        self.wandb.finish()