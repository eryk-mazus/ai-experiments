# input_size: 1 * 28 * 28 = 784
# flatten: 784

# values: from 0 to 1

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


def reparametrize(mu, logvar):
    # z = mu + sigma * epsilon

    sigma = torch.sqrt(torch.exp(logvar))
    z = sigma * torch.randn_like(mu) + mu
    return z


def loss_function(x_hat, x, mu, logvar):
    """
    loss for VAEs contains two terms: a reconstruction loss and KL divergence term
    """
    N = mu.shape[0]

    # The minus sign is handled by the BCE loss itself.
    rec_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")

    kl_div = 1 + logvar - mu**2 - torch.exp(logvar)
    kl_div = -0.5 * kl_div.sum()

    loss = rec_loss + kl_div
    loss /= N

    return loss
