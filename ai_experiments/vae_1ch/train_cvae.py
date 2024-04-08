import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch

from ai_experiments.dataloaders.mnist import get_mnist_dataloaders
from ai_experiments.vae_1ch.trainer import train_loop
from ai_experiments.vae_1ch.vae import CVAE, loss_function

BATCH_SIZE = 128

# CVAE hyperparams
LATENT_SIZE = 15
INPUT_SIZE = 28 * 28
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3

device = "cuda"

train_dataloader, _ = get_mnist_dataloaders(batch_size=BATCH_SIZE)

# instantiate the CVAE model
cvae = CVAE(INPUT_SIZE, latent_size=LATENT_SIZE)
cvae.cuda()

optimizer = torch.optim.Adam(cvae.parameters(), lr=LEARNING_RATE)

for t in range(NUM_EPOCHS):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(
        train_dataloader,
        cvae,
        loss_fn=loss_function,
        optimizer=optimizer,
        cond=True,
        batch_size=BATCH_SIZE,
    )

print("Training is done.")

# Visualizing CVAE generations

z = torch.randn(10, LATENT_SIZE)
c = torch.eye(10, 10)  # [one hot labels for 0-9]
import matplotlib.gridspec as gridspec

z = torch.cat((z, c), dim=-1).to(device="cuda")
cvae.eval()
samples = cvae.decoder(z).data.cpu().numpy()

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
    plt.savefig("./ai_experiments/vae_1ch/conditional_vae_generation.jpg")
