import torch


def one_hot(labels, class_size):
    """
    Create one hot label matrix of size (N, C)

    Inputs:
    - labels: Labels Tensor of shape (N,) representing a ground-truth label
    for each MNIST image
    - class_size: Scalar representing of target classes our dataset
    Outputs:
    - targets: One-hot label matrix of (N, C), where targets[i, j] = 1 when
    the ground truth label for image i is j, and targets[i, :j] &
    targets[i, j + 1:] are equal to 0
    """
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return targets


def train_loop(
    dataloader, model, loss_fn, optimizer, cond: bool = False, batch_size: int = 64
):
    num_classes = 10
    size = len(dataloader.dataset)

    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device="cuda:0")

        if cond:
            one_hot_vec = one_hot(y, num_classes).to(device="cuda")
            recon_batch, mu, logvar = model(X, one_hot_vec)
        else:
            recon_batch, mu, logvar = model(X)

        loss = loss_fn(recon_batch, X, mu, logvar)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
