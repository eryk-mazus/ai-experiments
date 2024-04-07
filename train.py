import torch
from vae import VAE, loss_function
from mnist import BATCH_SIZE, train_dataloader


latent_size = 15
input_size = 28 * 28

device = "cuda"

vae_model = VAE(input_size=input_size, latent_size=latent_size)
vae_model.cuda()


def train_loop(dataloader, model, loss_fn, optimizer, cond: bool = False):
    size = len(dataloader.dataset)

    model.train()

    for batch, (X, y) in enumerate(dataloader):
        # compute prediction and loss:
        X = X.to(device="cuda:0")
        if cond:
            raise NotImplementedError()
        else:
            recon_batch, mu, logvar = model(X)

        loss = loss_fn(recon_batch, X, mu, logvar)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * BATCH_SIZE + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


num_epochs = 10
learning_rate = 1e-3

optimizer = torch.optim.Adam(vae_model.parameters(), lr=learning_rate)

for t in range(num_epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, vae_model, loss_fn=loss_function, optimizer=optimizer)

print("Training is done.")
