from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T

NUM_CLASSES = 10


def get_mnist_dataloaders(batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
    training_data = datasets.MNIST(
        root="data", train=True, download=True, transform=T.ToTensor()
    )

    test_data = datasets.MNIST(
        root="data", train=False, download=True, transform=T.ToTensor()
    )

    train_dataloader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )

    return train_dataloader, test_dataloader
