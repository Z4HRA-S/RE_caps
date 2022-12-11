import torch


def train_loop(dataloader, model, optimizer):
    size = len(dataloader.dataset)
    for batch, data in enumerate(dataloader):
        # Compute prediction and loss
        loss, masked, labels = model(data)
        f1 = model.f1_measure(masked, labels)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(data)
            print(f"loss: {loss:>7f} f1: {f1:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model):
    with torch.no_grad():
        for data in dataloader:
            loss, masked, labels = model(data)
            f1 = model.f1_measure(masked, labels)

    print(f"Test Error: \n F1: {(f1):>7f},  loss: {loss:>8f} \n")
    return loss
