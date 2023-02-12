import torch
import numpy as np
from sklearn.metrics import f1_score
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler(enabled=True)


def train_loop(dataloader, model, optimizer, logger, batch_size):
    gradient_accumulations = 30
    size = len(dataloader.dataset)
    epoch_pred = []
    epoch_label = []
    for batch, data in enumerate(dataloader):
        # Compute prediction and loss
        with autocast():
            pred, labels, loss = model(data)

        # Backpropagation
        scaler.scale(loss / gradient_accumulations).backward()
        if (batch + 1) % gradient_accumulations == 0:
            logger.write(f"loss: {loss.item():>7f} [{(batch) * batch_size:>5d}/{size:>5d}]\n")
            print(f"loss: {loss.item():>7f} [{(batch) * batch_size}/{size:>5d}]")

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        epoch_label.append(labels)
        epoch_pred.append(pred)

    epoch_pred = torch.concat(epoch_pred).cpu()
    epoch_label = torch.concat(epoch_label).cpu()

    micro = f1_score(epoch_label, epoch_pred, average="micro")
    macro = f1_score(epoch_label, epoch_pred, average="macro")
    weighted = f1_score(epoch_label, epoch_pred, average="weighted")
    logger.write(f"F1: micro: {micro}, macro: {macro}, weighted: {weighted}\n")
    print(f"F1: micro: {micro}, macro: {macro}, weighted: {weighted}\n")
    return loss


def test_loop(dataloader, model, logger):
    with torch.no_grad():
        test_pred = []
        test_label = []
        test_loss = []
        for data in dataloader:
            pred, labels, loss = model(data, test=True)

            test_loss.append(loss)
            test_label.append(labels)
            test_pred.append(pred)

        test_pred = torch.concat(test_pred).cpu()
        test_label = torch.concat(test_label).cpu()
        test_loss = torch.stack(test_loss).mean().cpu()
        f1 = f1_score(test_label, test_pred, average="weighted")
        micro = f1_score(test_label, test_pred, average="micro")
        macro = f1_score(test_label, test_pred, average="macro")
        weighted = f1_score(test_label, test_pred, average="weighted")
        logger.write(f"F1: micro: {micro}, macro: {macro}, weighted: {weighted}\n")
        print(f"F1: micro: {micro}, macro: {macro}, weighted: {weighted}\n")

    print(f"loss: {test_loss:>8f} \n")
    logger.write(f"loss: {test_loss:>8f} \n")
    return loss
