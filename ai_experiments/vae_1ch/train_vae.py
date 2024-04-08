import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch

from ai_experiments.dataloaders.mnist import get_mnist_dataloaders
from ai_experiments.vae_1ch.trainer import train_loop
from ai_experiments.vae_1ch.vae import VAE, loss_function

BATCH_SIZE = 128

# CVAE hyperparams
LATENT_SIZE = 15
INPUT_SIZE = 28 * 28
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3

device = "cuda"

train_dataloader, _ = get_mnist_dataloaders(batch_size=BATCH_SIZE)

# instantiate the CVAE model
vae_model = VAE(INPUT_SIZE, latent_size=LATENT_SIZE)
vae_model.cuda()

optimizer = torch.optim.Adam(vae_model.parameters(), lr=LEARNING_RATE)

for t in range(NUM_EPOCHS):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(
        train_dataloader,
        vae_model,
        loss_fn=loss_function,
        optimizer=optimizer,
        cond=False,
        batch_size=BATCH_SIZE,
    )

print("Training is done.")

# Visualizing VAE generations

# sampling from prior
z = torch.randn(10, LATENT_SIZE).to(device="cuda")

vae_model.eval()

samples = vae_model.decoder(z).data.cpu().numpy()

fig = plt.figure(figsize=(10, 1))
gspec = gridspec.GridSpec(1, 10)
gspec.update(wspace=0.05, hspace=0.05)
for i, sample in enumerate(samples):
    ax = plt.subplot(gspec[i])
    plt.axis("off")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect("equal")
    plt.imshow(sample.reshape(28, 28), cmap="Greys_r")
    plt.savefig("./ai_experiments/vae_1ch/vae_generation.jpg")

# Visualizing VAE interpolation:
# TODO
