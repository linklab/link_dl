import torch
from torch import nn, optim
from datetime import datetime
import os
import wandb
from pathlib import Path
import torch.nn.functional as F

BASE_PATH = str(Path(__file__).resolve().parent.parent.parent) # BASE_PATH: /Users/yhhan/git/link_dl
import sys
sys.path.append(BASE_PATH)

CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_FILE_PATH = os.path.join(CURRENT_FILE_PATH, "checkpoints")

if not os.path.isdir(CHECKPOINT_FILE_PATH):
  os.makedirs(os.path.join(CURRENT_FILE_PATH, "checkpoints"))

from _01_code._13_autoencoders.b_fashion_mnist_data import get_fashion_mnist_data, get_fashion_mnist_test_data
from _01_code._13_autoencoders.a_arg_parser import get_parser
from _01_code._13_autoencoders.c_autoencoder_trainer import AutoencoderTrainer


def get_model(encoded_space_dim=8):
    class Encoder(nn.Module):
        def __init__(self):
            super(Encoder, self).__init__()

            ### Convolutional section
            self.encoder_cnn = nn.Sequential(
                nn.Conv2d(1, 8, 3, stride=2, padding=1),
                nn.ReLU(True),
                nn.Conv2d(8, 16, 3, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(True),
                nn.Conv2d(16, 32, 3, stride=2, padding=0),
                nn.ReLU(True)
            )

            ### Flatten layer
            self.flatten = nn.Flatten(start_dim=1)

            ### Linear section
            self.encoder_lin = nn.Sequential(
                nn.Linear(3 * 3 * 32, 128),
                nn.ReLU(True),
                nn.Linear(128, encoded_space_dim)
            )

        def forward(self, x):
            x = self.encoder_cnn(x)
            x = self.flatten(x)
            x = self.encoder_lin(x)
            return x

    class Decoder(nn.Module):
        def __init__(self):
            super(Decoder, self).__init__()
            self.decoder_lin = nn.Sequential(
                nn.Linear(encoded_space_dim, 128),
                nn.ReLU(True),
                nn.Linear(128, 3 * 3 * 32),
                nn.ReLU(True)
            )

            self.unflatten = nn.Unflatten(dim=1,
                                          unflattened_size=(32, 3, 3))

            self.decoder_conv = nn.Sequential(
                nn.ConvTranspose2d(32, 16, 3,
                                   stride=2, output_padding=0),
                nn.BatchNorm2d(16),
                nn.ReLU(True),
                nn.ConvTranspose2d(16, 8, 3, stride=2,
                                   padding=1, output_padding=1),
                nn.BatchNorm2d(8),
                nn.ReLU(True),
                nn.ConvTranspose2d(8, 1, 3, stride=2,
                                   padding=1, output_padding=1)
            )

        def forward(self, x):
            x = self.decoder_lin(x)
            x = self.unflatten(x)
            x = self.decoder_conv(x)
            x = torch.sigmoid(x)
            return x

    class Autoencoder(torch.nn.Module):
        def __init__(self):
            super(Autoencoder, self).__init__()
            self.encoder = Encoder()
            self.decoder = Decoder()

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x

    autoencoder = Autoencoder()
    return autoencoder


def main(args):
    run_time_str = datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S')

    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'validation_intervals': args.validation_intervals,
        'learning_rate': args.learning_rate,
        'early_stop_patience': args.early_stop_patience,
        'early_stop_delta': args.early_stop_delta,
    }

    project_name = "denoising_autoencoder"
    wandb.init(
        mode="online" if args.wandb else "disabled",
        project=project_name,
        notes="denoising autoencoder",
        tags=["denoising", "autoencoder", "fashion_mnist"],
        name=run_time_str,
        config=config
    )
    print(args)
    print(wandb.config)

    train_data_loader, validation_data_loader, f_mnist_transforms = get_fashion_mnist_data()
    f_mnist_test_images, test_data_loader, f_mnist_transforms = get_fashion_mnist_test_data()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on device {device}.")

    model = get_model()
    model.to(device)
    wandb.watch(model)

    optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

    regression_trainer = AutoencoderTrainer(
        project_name, model, optimizer, train_data_loader, validation_data_loader, f_mnist_transforms,
        run_time_str, wandb, device, CHECKPOINT_FILE_PATH,
        f_mnist_test_images, f_mnist_transforms,
        denoising=False,
    )
    regression_trainer.train_loop()

    wandb.finish()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)

    # python _01_code/_11_lstm_and_its_application/f_crypto_currency_regression_train_lstm.py --wandb
