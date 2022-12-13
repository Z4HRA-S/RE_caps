import torch


def train_loop(dataloader, model, optimizer):
    size = len(dataloader.dataset)
    for batch, data in enumerate(dataloader):
        # Compute prediction and loss
        loss, output, labels = model(data)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            print(f"loss: {loss.item():>7f} [{(batch + 1) * len(data):>5d}/{size:>5d}]")
    return loss


def test_loop(dataloader, model):
    with torch.no_grad():
        for data in dataloader:
            loss, output, labels = model(data)
            f1 = model.f1_measure(output, labels)

    print(f"Test Error: \n F1: {(f1):>7f},  loss: {loss:>8f} \n")
    return loss
