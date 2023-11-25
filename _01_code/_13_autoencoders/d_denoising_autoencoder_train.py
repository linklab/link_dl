import torch
from torch import nn, optim
from datetime import datetime
import os
import wandb
from pathlib import Path

from _01_code._13_autoencoders.b_fashion_mnist_data import get_fashion_mnist_data, get_fashion_mnist_test_data

BASE_PATH = str(Path(__file__).resolve().parent.parent.parent) # BASE_PATH: /Users/yhhan/git/link_dl
import sys
sys.path.append(BASE_PATH)

CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_FILE_PATH = os.path.join(CURRENT_FILE_PATH, "checkpoints")

if not os.path.isdir(CHECKPOINT_FILE_PATH):
  os.makedirs(os.path.join(CURRENT_FILE_PATH, "checkpoints"))

from _01_code._13_autoencoders.a_arg_parser import get_parser
from _01_code._13_autoencoders.c_autoencoder_trainer import AutoencoderTrainer


def get_model(encoded_space_dim=4):
    class Encoder(nn.Module):
        def __init__(self, encoded_space_dim):
            super().__init__()

            ### Convolutional section
            self.encoder_cnn = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(num_features=16),
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=0),
                nn.ReLU()
            )

            ### Flatten layer
            self.flatten = nn.Flatten(start_dim=1)

            ### Linear section
            self.encoder_lin = nn.Sequential(
                nn.Linear(in_features=32 * 3 * 3, out_features=128),
                nn.ReLU(),
                nn.Linear(in_features=128, out_features=encoded_space_dim)
            )

        def forward(self, x):
            x = self.encoder_cnn(x)
            x = self.flatten(x)
            x = self.encoder_lin(x)
            return x

    class Decoder(nn.Module):
        def __init__(self, encoded_space_dim):
            super().__init__()
            self.decoder_lin = nn.Sequential(
                nn.Linear(in_features=encoded_space_dim, out_features=128),
                nn.ReLU(),
                nn.Linear(in_features=128, out_features=32 * 3 * 3),
                nn.ReLU()
            )

            self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))

            self.decoder_conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=0, output_padding=0),
                nn.BatchNorm2d(num_features=16),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.Sigmoid()
            )

        def forward(self, x):
            x = self.decoder_lin(x)
            x = self.unflatten(x)
            x = self.decoder_conv(x)
            return x

    encoder = Encoder(encoded_space_dim=encoded_space_dim)
    decoder = Decoder(encoded_space_dim=encoded_space_dim)

    return encoder, decoder


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

    encoder, decoder = get_model()
    encoder.to(device)
    decoder.to(device)

    params_to_optimize = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
    ]

    optimizer = optim.Adam(params_to_optimize, lr=wandb.config.learning_rate)

    regression_trainer = AutoencoderTrainer(
        project_name, encoder, decoder, optimizer, train_data_loader, validation_data_loader, f_mnist_transforms,
        run_time_str, wandb, device, CHECKPOINT_FILE_PATH,
        f_mnist_test_images, f_mnist_transforms,
        denoising=True,
    )
    regression_trainer.train_loop()

    wandb.finish()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)

    # python _01_code/_11_lstm_and_its_application/f_crypto_currency_regression_train_lstm.py --wandb
