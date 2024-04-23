from datetime import datetime
from pathlib import Path

import fire
import torch
from loguru import logger
from torch import nn, optim
from torch.utils.data import DataLoader

from ai_experiments.dataloaders.celeba import get_celeba_dataloaders
from ai_experiments.vq_vae.vq_vae import VQ_VAE, VQ_VAE_Loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


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
        output_dir: str,
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

        self.output_dir = output_dir

    def train_one_epoch(self, log_every: int = 10):
        loss_meter = AverageMeter()
        reconstruction_loss_meter = AverageMeter()
        vq_loss_meter = AverageMeter()

        for i, batch in enumerate(self.train_loader):
            inputs, _ = batch
            inputs = inputs.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            losses = self.loss_cls(outputs, inputs)

            losses["loss"].backward()
            self.optimizer.step()

            if self.batch_scheduler is not None:
                self.batch_scheduler.step()

            loss_meter.update(losses["loss"].item(), inputs.size(0))
            reconstruction_loss_meter.update(
                losses["reconstruction_loss"].item(), inputs.size(0)
            )
            vq_loss_meter.update(losses["vq_loss"].item(), inputs.size(0))

            if (i + 1) % log_every == 0:
                logger.info(
                    f"Step {i + 1} - Total Loss: {loss_meter.avg:.4f}, Reconstruction Loss: {reconstruction_loss_meter.avg:.4f}, VQ Loss: {vq_loss_meter.avg:.4f}"
                )

        return {
            "loss": loss_meter.avg,
            "reconstruction_loss": reconstruction_loss_meter.avg,
            "vq_loss": vq_loss_meter.avg,
        }

    def validate_one_epoch(self):
        loss_meter = AverageMeter()
        reconstruction_loss_meter = AverageMeter()
        vq_loss_meter = AverageMeter()

        self.model.eval()
        with torch.no_grad():
            for i, v_batch in enumerate(self.val_loader):
                vinputs, _ = v_batch
                vinputs = vinputs.to(self.device)
                voutputs = self.model(vinputs)
                vlosses = self.loss_cls(voutputs, vinputs)

                loss_meter.update(vlosses["loss"].item(), vinputs.size(0))
                reconstruction_loss_meter.update(
                    vlosses["reconstruction_loss"].item(), vinputs.size(0)
                )
                vq_loss_meter.update(vlosses["vq_loss"].item(), vinputs.size(0))

        return {
            "loss": loss_meter.avg,
            "reconstruction_loss": reconstruction_loss_meter.avg,
            "vq_loss": vq_loss_meter.avg,
        }

    def train(self, log_every: int = 10):
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.epochs):
            logger.info("EPOCH {}:".format(epoch + 1))

            self.model.train(True)
            train_summary = self.train_one_epoch(log_every=log_every)
            self.log_loss_summary(train_summary, stage="Training")

            validation_summary = self.validate_one_epoch()
            self.log_loss_summary(validation_summary, stage="Validation")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"model_{epoch}_{timestamp}.pth"
            model_path = output_path / model_name
            torch.save(self.model.state_dict(), model_path)

        # save the final checkpoint
        final_output_path = output_path / "final"
        final_output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = "model_{}.pth".format(timestamp)
        model_path = final_output_path / model_name

        torch.save(self.model.state_dict(), model_path)

    def log_loss_summary(self, loss_summary, stage="Training"):
        logger.info(f"{stage} Summary: ")
        for key, value in loss_summary.items():
            logger.info(f"{key}: {value:.4f}")


def main(
    # dataloader hyperparams
    root: str = "./data/",  # path to the `celeba` folder, more details: https://pytorch.org/vision/main/_modules/torchvision/datasets/celeba.html
    image_size: int = 128,
    batch_size: int = 32,
    num_workers: int = 2,
    # VQ-VAE hyperparams
    hidden_dim: int = 16,
    num_embeddings: int = 256,
    commitment_cost: float = 0.25,
    # training hyperparams
    epochs: int = 10,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.01,
    vq_loss_requlatization: float = 1.0,
    # other
    output_dir: str = "output",
    log_every: int = 10,
    device: str = DEVICE,
):
    train_loader, val_loader = get_celeba_dataloaders(
        root=root, image_size=image_size, batch_size=batch_size, num_workers=num_workers
    )

    model = VQ_VAE(
        in_channels=3,
        hidden_dim=hidden_dim,
        num_embeddings=num_embeddings,
        commitment_cost=commitment_cost,
    )
    model.to(device)
    logger.info(
        "Number of parameters: {:,}".format(sum(p.numel() for p in model.parameters()))
    )

    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    lr_scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
    )

    loss_cls = VQ_VAE_Loss(regularization=vq_loss_requlatization)

    trainer = VQ_VAE_Trainer(
        model=model,
        loss_cls=loss_cls,
        optimizer=optimizer,
        epochs=epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        batch_scheduler=lr_scheduler,
        output_dir=output_dir,
    )

    trainer.train(log_every=log_every)

    print("done.")


if __name__ == "__main__":
    fire.Fire(main)
