import torch
from torchvision import datasets
from torchvision import transforms as T
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

BATCH_SIZE = 128
NUM_CLASSES = 10

training_data = datasets.MNIST(
    root="data", train=True, download=True, transform=T.ToTensor()
)

test_data = datasets.MNIST(
    root="data", train=False, download=True, transform=T.ToTensor()
)

# 1 x 28 x 28
# print(training_data[0][0].shape)

# # visualize some examples
# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(training_data), size=(1,)).item()
#     img, label = training_data[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.title(label=label)
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()

train_dataloader = DataLoader(
    training_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    drop_last=True,
)
test_dataloader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    drop_last=True,
)

# iterate through dataloader

# train_features, train_labels = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# img = train_features[0].squeeze()
# label = train_labels[0]
# plt.imshow(img, cmap="gray")
# plt.show()
# print(f"Label: {label}")
