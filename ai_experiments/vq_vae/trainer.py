from typing import TYPE_CHECKING

import fire
import torch
from loguru import logger
from torch import nn, optim
from torch.utils.data import DataLoader

from ai_experiments.dataloaders.celeba import get_celeba_dataloaders
from ai_experiments.vq_vae.vq_vae import VQ_VAE, VQ_VAE_Loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VQ_VAE_Trainer:
    def __init__(
        self,
        model: nn.Module,
        loss_cls: VQ_VAE_Loss,
        optimizer,
        epochs: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str,
        batch_scheduler=None,
    ) -> None:
        self.model = model
        self.device = device

        self.optimizer = optimizer
        self.batch_scheduler = batch_scheduler

        self.loss_cls = loss_cls
        self.epochs = epochs

        self.train_loader = train_loader
        self.val_loader = val_loader

    def train_one_epoch(self):
        running_loss = 0.0
        last_loss = 0

        for i, batch in enumerate(self.train_loader):

            # Every data instance is an input + label pair
            inputs, _ = batch
            inputs = inputs.to(self.device)

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch: reconstructed image and vq_loss
            outputs = self.model(inputs)

            # Compute the loss and its gradients
            losses = self.loss_cls(outputs, inputs)
            losses["loss"].backward()

            # Adjust learning weights
            self.optimizer.step()

            if self.batch_scheduler is not None:
                self.batch_scheduler.step()

            # Gather data and report
            # TODO: report other losses as well
            running_loss += losses["loss"].item()
            if i % 10 == 9:
                last_loss = running_loss / 10
                logger.info("  batch {} loss: {}".format(i + 1, last_loss))
                running_loss = 0.0

        return last_loss

    def train(self):
        for epoch in range(self.epochs):
            logger.info("EPOCH {}:".format(epoch + 1))

            self.model.train(True)
            avg_loss = self.train_one_epoch()

            self.model.eval()
            running_vloss = 0.0

            with torch.no_grad():
                for i, v_batch in enumerate(self.val_loader):
                    vinputs, _ = v_batch
                    vinputs = vinputs.to(self.device)
                    voutputs = self.model(vinputs)
                    vlosses = self.loss_cls(voutputs, vinputs)
                    running_vloss += vlosses["loss"]

            avg_vloss = running_vloss / (i + 1)
            logger.info("LOSS train {} valid {}".format(avg_loss, avg_vloss))

            # todo: save checkpoint


def main(
    # dataloader hyperparams
    root: str = "./data/",  # path to the `celeba` folder, more details: https://pytorch.org/vision/main/_modules/torchvision/datasets/celeba.html
    image_size: int = 128,
    batch_size: int = 64,
    num_workers: int = 2,
    # VQ-VAE hyperparams
    hidden_dim: int = 256,
    num_embeddings: int = 512,
    # training hyperparams
    epochs: int = 20,
    learning_rate: float = 2e-4,
    weight_decay: float = 0.01,
    # other
    device: str = DEVICE,
):

    train_loader, val_loader = get_celeba_dataloaders(
        root=root, image_size=image_size, batch_size=batch_size, num_workers=num_workers
    )

    model = VQ_VAE(in_channels=3, hidden_dim=hidden_dim, num_embeddings=num_embeddings)
    model.to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    lr_scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
    )

    loss_cls = VQ_VAE_Loss(regularization=0.1)

    trainer = VQ_VAE_Trainer(
        model=model,
        loss_cls=loss_cls,
        optimizer=optimizer,
        epochs=epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        batch_scheduler=lr_scheduler,
    )

    trainer.train()

    print("done.")


if __name__ == "__main__":
    fire.Fire(main)
