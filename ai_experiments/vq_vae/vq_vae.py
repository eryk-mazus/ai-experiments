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


class VectorQuantizerEMA(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        gamma: float = 0.99,
    ) -> None:
        """
        VQ with dictionary items being updated with Exponential Moving Averages

        Args:
            num_embeddings: number of vectors in the quantized space (K)
            embedding_dim: dimensionality of the tensors in the quantized space (D)
            commitment_cost: scalar which controls the weighting of the loss terms (Beta)
            gamma: decay parameter with a value between 0 and 1
        """
        super().__init__()

        self.commitment_cost = commitment_cost
        self.gamma = gamma

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)

        # EMA weights and cluster sizes
        self.ema_cluster_size = nn.Parameter(
            torch.zeros(num_embeddings), requires_grad=False
        )
        self.ema_embedding = nn.Parameter(
            torch.zeros_like(self.embedding.weight), requires_grad=False
        )

        self.init_weights()

    def init_weights(self) -> None:
        nn.init.uniform_(
            self.embedding.weight, -1 / self.num_embeddings, 1 / self.num_embeddings
        )
        # initialize EMA embedding to the initial weights
        self.ema_embedding.data.copy_(self.embedding.weight.data)

    def forward(self, x: torch.Tensor):
        # input: N x C x H x W
        # embed: K x C

        flat_x = x.view(-1, self.embedding_dim)  # -1 x C

        # -1 x 1 x C  -  K x C  ==> -1 x K x C  ==> sum(-1) ==> -1 x K
        distances = (flat_x.unsqueeze(1) - self.embedding.weight).pow(2).sum(-1)
        nearest_embedding_idx = distances.argmin(1)  # -1

        # mapping to embeddings, back to input dimensions:
        quantized_latents = self.embedding(nearest_embedding_idx).view_as(x)

        if self.training:
            # using one-hot encoding to find the usage of each embedding
            # -1 x K
            encodings = F.one_hot(nearest_embedding_idx, self.num_embeddings).type_as(
                self.embedding.weight
            )
            # usage of each embedding, n_i in paper
            encoding_sum = encodings.sum(0)
            encoding_sum = torch.where(
                encoding_sum > 0, encoding_sum, torch.ones_like(encoding_sum)
            )

            # EMA update of cluster sizes (N_i in paper)
            # mutiply each element by gamma and add count * (1 - gamma)
            self.ema_cluster_size.data.mul_(self.gamma).add_(
                encoding_sum, alpha=1 - self.gamma
            )

            # EMA update of embeddings (m_i in paper)
            dw = torch.matmul(encodings.t(), flat_x)
            self.ema_embedding.data.mul_(self.gamma).add_(dw, alpha=1 - self.gamma)

            # smoothing
            n = self.ema_cluster_size.sum()
            cluster_size_smooth = self.ema_cluster_size / n

            self.embedding.weight.data.copy_(
                self.ema_embedding / cluster_size_smooth.unsqueeze(1)
            )

        # Loss calculation (just commitment loss)
        vq_loss = self.commitment_cost * F.mse_loss(x, quantized_latents.detach())

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

    # model = VQ_VAE()
    # y, loss = model(x)

    vq = VectorQuantizerEMA(num_embeddings=10, embedding_dim=3)
    _ = vq(x)
    ...
