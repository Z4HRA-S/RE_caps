import torch


def train_loop(dataloader, model, loss_fn, optimizer, scheduler):
    size = len(dataloader.dataset)
    for batch, data in enumerate(dataloader):
        # Compute prediction and loss
        y = data["label"]
        pred = model(data)
        loss, f1 = loss_fn(pred, y)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(data)
            print(f"loss: {loss:>7f} f1: {f1:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for data in dataloader:
            pred = model(data)
            loss, f1 = loss_fn(pred, data["label"])

    print(f"Test Error: \n F1: {(f1):>7f},  loss: {loss:>8f} \n")
