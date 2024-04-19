import torch
import torch.nn.functional as F
from torch import nn


class VectorQuantizer(nn.Module):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25
    ) -> None:
        """
        Args:
            num_embeddings: number of vectors in the quantized space (K)
            embedding_dim: dimensionality of the tensors in the quantized space (D)
            commitment_cost: scalar which controls the weighting of the loss terms (Beta)
        """
        super().__init__()

        self.commitment_cost = commitment_cost
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.init_weights()

    def init_weights(self) -> None:
        nn.init.uniform_(
            self.embedding.weight, -1 / self.num_embeddings, 1 / self.num_embeddings
        )

    def quantize(self, embedding_indices):
        z = self.embedding(embedding_indices)
        # move the channe dimension to it's original place:
        z = z.movedim(-1, 1)
        return z

    def forward(self, x: torch.Tensor):
        # input: N x C x H x W
        # embed: K x C

        # move the channel dimension at the end, and add dimension to broadcast during substraction
        dist = torch.linalg.vector_norm(
            x.movedim(1, -1).unsqueeze(-2) - self.embedding.weight, dim=-1
        )
        nearest_embedding_idx = torch.argmin(dist, dim=-1)
        quantized_latents = self.quantize(nearest_embedding_idx)

        # loss computation:
        codebook_loss = F.mse_loss(x.detach(), quantized_latents)
        commitment_loss = self.commitment_cost * F.mse_loss(
            x, quantized_latents.detach()
        )
        vq_loss = codebook_loss + commitment_loss

        # Passing the gradient from the decoder straight to encoder
        quantized_latents = x + (quantized_latents - x).detach()
        return quantized_latents, vq_loss


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()

        self._block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
        )

    def forward(self, x):
        return x + self._block(x)


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_dim: int = 256) -> None:
        super().__init__()
        self._encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_dim // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_dim // 2,
                out_channels=hidden_dim,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            # residual block starts with relu, so it is omitted here
            ResidualBlock(hidden_dim=hidden_dim),
            ResidualBlock(hidden_dim=hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self._encoder(x)


class Decoder(nn.Module):
    def __init__(
        self, in_channels: int, hidden_dim: int = 256, out_channels: int = 1
    ) -> None:
        super().__init__()
        self._decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            ResidualBlock(hidden_dim=hidden_dim),
            ResidualBlock(hidden_dim=hidden_dim),
            nn.ConvTranspose2d(
                in_channels=hidden_dim,
                out_channels=hidden_dim // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ConvTranspose2d(
                in_channels=hidden_dim // 2,
                out_channels=3,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
        )

    def forward(self, x):
        x = self._decoder(x)
        return torch.sigmoid(x)


class VQ_VAE(nn.Module):
    def __init__(
        self, in_channels: int = 3, hidden_dim: int = 256, num_embeddings: int = 512
    ) -> None:
        super().__init__()
        self._encoder = Encoder(in_channels, hidden_dim)
        self._vq = VectorQuantizer(
            num_embeddings=num_embeddings, embedding_dim=hidden_dim
        )
        self._decoder = Decoder(
            in_channels=hidden_dim, hidden_dim=hidden_dim, out_channels=in_channels
        )

    def forward(self, x):
        z_e = self._encoder(x)
        z, vq_loss = self._vq(z_e)
        out = self._decoder(z)
        return out, vq_loss

    def sample(self): ...


class VQ_VAE_Loss(nn.Module):
    def __init__(self, regularization: float = 1.0) -> None:
        super().__init__()
        self.reconstruction_loss_func = nn.MSELoss()
        self.regularization = regularization

    def forward(self, model_outputs, target):
        x_hat, vq_loss = model_outputs
        reconstruction_loss = self.reconstruction_loss_func(x_hat, target)
        total_loss = reconstruction_loss + self.regularization * vq_loss
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "vq_loss": vq_loss,
        }


if __name__ == "__main__":
    N, C, H, W = 2, 3, 32, 32
    x = torch.randn(N, C, H, W)

    model = VQ_VAE()
    y, loss = model(x)

    ...
