"""
VAE and Conditional VAE implementation
Operates on 1 x 28 x 28 images normalized to 0-1 values 
"""

import torch
from torch import nn


class VAE(nn.Module):
    def __init__(self, input_size, latent_size=15):
        super().__init__()

        self.input_size = input_size  # H x W
        self.latent_size = latent_size  # Z

        self.hidden_dim = 400

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_size, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )

        self.mu_layer = nn.Linear(self.hidden_dim, self.latent_size)
        self.logvar_layer = nn.Linear(self.hidden_dim, self.latent_size)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_size),
            nn.Sigmoid(),
            nn.Unflatten(dim=1, unflattened_size=(1, 28, 28)),
        )

    def forward(self, x):
        """
        Args:
            x: batch of input images of shape (N, 1, H, W)

        Returns:
            x_hat: reconstructed input data of shape (N, 1, H, W)
            mu: matrix representing posterior mu (N, Z)
            logvar: matrix representing estimated variance in log-space (N, Z)
        """
        features = self.encoder(x)
        mu = self.mu_layer(features)
        logvar = self.logvar_layer(features)

        z = reparametrize(mu, logvar)

        x_hat = self.decoder(z)

        return x_hat, mu, logvar


class CVAE(nn.Module):
    def __init__(self, input_size, num_classes=10, latent_size=15):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.num_classes = num_classes

        self.hidden_dim = 400

        self.encoder = nn.Sequential(
            nn.Linear(self.input_size + self.num_classes, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )

        self.mu_layer = nn.Linear(self.hidden_dim, self.latent_size)
        self.logvar_layer = nn.Linear(self.hidden_dim, self.latent_size)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size + self.num_classes, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_size),
            nn.Sigmoid(),
            nn.Unflatten(dim=1, unflattened_size=(1, 28, 28)),
        )

    def forward(self, x, c):
        """
        Args:
            x: input data for this timestep of shape (N, 1, H, W)
            c: one hot vector representing the input class (0-9) (N, C)
        """
        x_flat = torch.flatten(x, start_dim=1, end_dim=-1)
        x_concat = torch.cat((x_flat, c), dim=1)

        features = self.encoder(x_concat)
        mu = self.mu_layer(features)
        logvar = self.logvar_layer(features)

        z = reparametrize(mu=mu, logvar=logvar)

        x_hat = self.decoder(torch.cat((z, c), dim=1))

        return x_hat, mu, logvar


def reparametrize(mu, logvar):
    # reparametrization trick
    # z = mu + sigma * epsilon

    sigma = torch.sqrt(torch.exp(logvar))
    z = sigma * torch.randn_like(mu) + mu
    return z


def loss_function(x_hat, x, mu, logvar):
    # loss for VAEs contains two terms: a reconstruction loss and KL divergence term
    N = mu.shape[0]

    # The minus sign is handled by the BCE loss itself.
    rec_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")

    kl_div = 1 + logvar - mu**2 - torch.exp(logvar)
    kl_div = -0.5 * kl_div.sum()

    loss = rec_loss + kl_div
    loss /= N

    return loss
