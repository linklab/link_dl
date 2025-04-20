import os
import time
import torch
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from collections import defaultdict
from pathlib import Path

BASE_PATH = str(Path(__file__).resolve().parent.parent.parent) # BASE_PATH: /Users/yhhan/git/link_dl
import sys
sys.path.append(BASE_PATH)

from a_vae_models import VAE


def main(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.join(BASE_PATH, "_00_data"):
        os.mkdir(os.path.join(BASE_PATH, "_00_data"))
    if not os.path.join(BASE_PATH, "_00_data", "h_mnist"):
        os.mkdir(os.path.join(BASE_PATH, "_00_data", "h_mnist"))

    ts = time.time()
    if not os.path.exists(os.path.join(args.fig_root, str(ts))):
        if not (os.path.exists(os.path.join(args.fig_root))):
            os.mkdir(os.path.join(args.fig_root))
        os.mkdir(os.path.join(args.fig_root, str(ts)))

    dataset = MNIST(
        root=os.path.join(BASE_PATH, "_00_data", "h_mnist"),
        train=True, transform=transforms.ToTensor(), download=True
    )
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)

    def loss_fn(recon_x, x, mean, log_var):
        BCE = torch.nn.functional.binary_cross_entropy(
            recon_x.view(-1, 28 * 28), x.view(-1, 28 * 28), reduction='sum'
        )
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return (BCE + KLD) / x.size(0)

    vae = VAE(
        encoder_layer_sizes=args.encoder_layer_sizes,
        latent_size=args.latent_size,
        decoder_layer_sizes=args.decoder_layer_sizes,
        conditional=args.conditional,
        num_labels=10 if args.conditional else 0
    ).to(device)

    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)

    logs = defaultdict(list)

    for epoch in range(args.epochs):
        epoch_latent_vectors = defaultdict(lambda: defaultdict(dict))
        for iteration, (x, y) in enumerate(data_loader):
            # x.shape: (bs, 1, 28, 28)
            # y.shape: (bs,)
            x, y = x.to(device), y.to(device)

            # recon_x: (128, 784)
            # mean: (128, 2)
            # log_var: (128, 2)
            # z: (128, 2)
            if args.conditional:
                recon_x, mean, log_var, z = vae(x, y)
            else:
                recon_x, mean, log_var, z = vae(x)

            # epoch_latent_vectors =
            # {
            #   0: defaultdict(<class 'dict'>, {'x': 0.976950347423553, 'y': 1.8106703758239746, 'label': 7}),
            #   1: defaultdict(<class 'dict'>, {'x': 1.988161325454712, 'y': 0.5007053017616272, 'label': 9}),
            #   ...
            # }
            for i, yi in enumerate(y):
                id = len(epoch_latent_vectors)
                epoch_latent_vectors[id]['x'] = z[i, 0].item()
                epoch_latent_vectors[id]['y'] = z[i, 1].item()
                epoch_latent_vectors[id]['label'] = yi.item()

            loss = loss_fn(recon_x, x, mean, log_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logs['loss'].append(loss.item())

            if iteration % args.print_every_batch == 0 or iteration == len(data_loader)-1:
                print("[Epoch {:02d}/{:02d}] Batch {:04d}/{:d}, Loss {:9.4f}".format(
                    epoch + 1, args.epochs, iteration, len(data_loader)-1, loss.item()
                ))

        print("-" * 50)

        ##### Figures: Start #####
        if args.conditional:
            # c = tensor([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])
            # z = tensor([[ 0.8595,  0.5746],
            #             [-1.2975, -0.8429],
            #             [-0.0832, -1.6947],
            #             [ 0.1950, -2.3876],
            #             [-0.0788,  0.1448],
            #             [-0.1970,  0.4447],
            #             [ 0.8921, -0.3893],
            #             [-0.5294,  0.0805],
            #             [-0.2197, -0.3926],
            #             [-0.9872,  2.2230]])
            c = torch.arange(0, 10).long().unsqueeze(1).to(device)
            z = torch.randn([c.size(0), args.latent_size]).to(device)
            x = vae.inference(z, c=c)
        else:
            # z = tensor([[ 0.8595,  0.5746],
            #             [-1.2975, -0.8429],
            #             [-0.0832, -1.6947],
            #             [ 0.1950, -2.3876],
            #             [-0.0788,  0.1448],
            #             [-0.1970,  0.4447],
            #             [ 0.8921, -0.3893],
            #             [-0.5294,  0.0805],
            #             [-0.2197, -0.3926],
            #             [-0.9872,  2.2230]])
            z = torch.randn([10, args.latent_size]).to(device)
            x = vae.inference(z)

        plt.figure()
        plt.figure(figsize=(5, 10))
        plt.title("Conditional VAE" if args.conditional else "VAE", fontsize=16, y=1.02)
        plt.axis('off')

        for p in range(10):
            plt.subplot(5, 2, p+1)
            if args.conditional:
                plt.text(
                    0, 0, "c={:d}".format(c[p].item()), color='black', backgroundcolor='white', fontsize=12
                )
            plt.imshow(x[p].view(28, 28).cpu().data.numpy())
            plt.axis('off')

        plt.savefig(
            os.path.join(args.fig_root, str(ts), "Epoch_{:03d}.png".format(epoch)),
            dpi=300
        )

        plt.clf()
        plt.close('all')

        df = pd.DataFrame.from_dict(epoch_latent_vectors, orient='index')
        g = sns.lmplot(
            x='x', y='y', hue='label', data=df.groupby('label').head(100), fit_reg=False, legend=True
        )
        g.savefig(
            os.path.join(args.fig_root, str(ts), "Epoch_{:03d}-dist.png".format(epoch)),
            dpi=300
        )
        ##### Figures: End #####

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--encoder_layer_sizes", type=list, default=[784, 256])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[256, 784])
    parser.add_argument("--latent_size", type=int, default=2)
    parser.add_argument("--print_every_batch", type=int, default=100)
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--conditional", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()

    main(args)