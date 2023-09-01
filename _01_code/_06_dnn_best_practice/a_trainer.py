from datetime import datetime
import os
import torch
from torch import nn

from _01_code._99_common_utils.utils import strfdelta


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience, project_name, run_time_str):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            path (str): Path for the checkpoint to be saved to.
        """
        self.patience = patience
        self.counter = 0
        self.early_stop = False
        self.val_loss_min = None
        self.file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "checkpoints", f"{project_name}_checkpoint_{run_time_str}.pt"
        )
        self.latest_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "checkpoints", f"{project_name}_checkpoint_latest.pt"
        )

    def check_and_save(self, val_loss, model):
        if self.val_loss_min is None:
            self.val_loss_min = val_loss
        elif val_loss >= self.val_loss_min:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(val_loss, model)
            self.val_loss_min = val_loss
            self.counter = 0

        return self.early_stop

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.file_path)
        torch.save(model.state_dict(), self.latest_file_path)
        self.val_loss_min = val_loss


class ClassificationTrainer:
    def __init__(self, project_name, model, optimizer, train_data_loader, validation_data_loader, run_time_str, wandb, device):
        self.project_name = project_name
        self.model = model
        self.optimizer = optimizer
        self.train_data_loader = train_data_loader
        self.validation_data_loader = validation_data_loader
        self.run_time_str = run_time_str
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
        early_stopping = EarlyStopping(
            patience=7, project_name=f"{self.project_name}", run_time_str=f"{self.run_time_str}.pt"
        )
        n_epochs = self.wandb.config.epochs
        training_start_time = datetime.now()

        for epoch in range(1, n_epochs + 1):
            train_loss, train_accuracy = self.do_train()
            validation_loss, validation_accuracy = self.do_validation()

            elapsed_time = datetime.now() - training_start_time
            if epoch == 1 or epoch % 10 == 0:
                print(
                    f"[Epoch {epoch:>3}] "
                    f"Training loss: {train_loss:5.2f}, "
                    f"Training accuracy: {train_accuracy:5.2f} | "
                    f"Validation loss: {validation_loss:5.2f}, "
                    f"Validation accuracy: {validation_accuracy:5.2f} | "
                    f"Training time: {strfdelta(elapsed_time, '%H:%M:%S')} "
                )

            self.wandb.log({
                "Epoch": epoch,
                "Training loss": train_loss,
                "Training accuracy": train_accuracy,
                "Validation loss": validation_loss,
                "Validation accuracy": validation_accuracy,
            })

            early_stop = early_stopping.check_and_save(validation_loss, self.model)

            if early_stop:
                break

        elapsed_time = datetime.now() - training_start_time
        print(f"Final training time: {strfdelta(elapsed_time, '%H:%M:%S')}")

        self.wandb.finish()
