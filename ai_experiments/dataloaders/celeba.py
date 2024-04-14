from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vision_utils
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T


def plot_batch(ax, batch, title=None, **kwargs):
    imgs = vision_utils.make_grid(batch, padding=2, normalize=True)
    imgs = np.moveaxis(imgs.numpy(), 0, -1)
    ax.set_axis_off()
    if title is not None:
        ax.set_title(title)
    return ax.imshow(imgs, **kwargs)


def show_images(batch, title):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    plot_batch(ax, batch, title)
    plt.show()


def get_celeba_dataloaders(
    root: str,
    image_size: int = 128,
    batch_size: int = 64,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:

    train_transform = T.Compose(
        [
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ]
    )

    val_transform = T.Compose(
        [T.Resize(image_size), T.CenterCrop(image_size), T.ToTensor()]
    )

    training_data = datasets.CelebA(
        root, split="train", transform=train_transform, download=False
    )
    test_data = datasets.CelebA(
        root, split="test", transform=val_transform, download=False
    )

    train_dataloader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
    )

    return train_dataloader, test_dataloader


if __name__ == "__main__":
    # test, peek at the dataset:
    train_dataloader, _ = get_celeba_dataloaders(root="./data/")

    test_batch, _ = next(iter(train_dataloader))
    show_images(test_batch[:64], "Training Images")

    print("done.")
