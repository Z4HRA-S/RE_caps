import torch
import numpy as np
from sklearn.metrics import f1_score

"""def max_f1(logits, labels):
    threshold = np.arange(0, 1, 0.001)[1:]
    pred = [logits.gt(th).float() for th in threshold]
    f1 = np.array([f1_score(labels, p, average=None) for p in pred])
    max_f1 = f1.max(axis=0)
    max_th = np.array([threshold[i] for i in f1.argmax(axis=0)])
    return max_f1, max_th"""


def train_loop(dataloader, model, optimizer, logger):
    size = len(dataloader.dataset)
    epoch_pred = []
    epoch_label = []
    for batch, data in enumerate(dataloader):
        # Compute prediction and loss
        loss, masked, labels = model(data)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_label.append(labels)
        epoch_pred.append(masked)

        if batch % 100 == 0:
            logger.write(f"loss: {loss.item():>7f} [{(batch + 1) * len(data):>5d}/{size:>5d}]\n")
            print(f"loss: {loss.item():>7f} [{(batch + 1) * len(data):>5d}/{size:>5d}]")
    epoch_pred = torch.concat(epoch_pred).cpu()
    epoch_label = torch.concat(epoch_label).cpu()
    f1 = f1_score(epoch_label, epoch_pred, average=None)
    logger.write(f"F1: {f1}\n")
    print(f"F1: {f1}")
    return loss


def test_loop(dataloader, model, logger):
    with torch.no_grad():
        test_pred = []
        test_label = []
        test_loss = []
        for data in dataloader:
            loss, masked, labels = model(data)

            test_loss.append(loss)
            test_label.append(labels)
            test_pred.append(masked)

        test_pred = torch.concat(test_pred).cpu()
        test_label = torch.concat(test_label).cpu()
        test_loss = torch.stack(test_loss).mean().cpu()
        f1 = f1_score(test_label, test_pred, average=None)

    print(f"Test Error: \n F1: {f1},  loss: {test_loss:>8f} \n")
    logger.write(f"Test Error: \n F1: {f1},  loss: {test_loss:>8f} \n")
    return loss
