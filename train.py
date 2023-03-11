import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import precision_recall_curve
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler(enabled=True)


def get_metrics(labels, logits, th=0.5):
    pred = logits.gt(th).float()
    f1 = f1_score(labels, pred, average="binary")
    pr = precision_score(labels, pred, average="binary")
    rcl = recall_score(labels, pred, average="binary")
    return pr, rcl, f1


def calculate_metrics(labels, logits):
    """fpr, tpr, thresholds = roc_curve(labels, logits)
    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = np.argmax(gmeans)
    pr, rcl, f1 = get_metrics(labels, logits, thresholds[ix])"""
    results = []
    logits = logits.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    for i in range(96):
        true = labels[:, i]
        pred = logits[:, i]
        pr, recall, thresholds = precision_recall_curve(true, pred)
        fscore = (2 * pr * recall) / (pr + recall + 1e-20)
        ix = np.argmax(fscore)
        weight = sum(true)
        results.append([weight, fscore[ix], thresholds[ix]])
    results = np.array(results)
    preds = np.array([logits[:, i].__gt__(results[i, -1]) for i in range(96)]).transpose()
    micro_f1 = f1_score(labels, preds, average="micro")
    average_f1 = f1_score(labels, preds, average="macro")
    weighted_f1 = f1_score(labels, preds, average="weighted")
    print(results[:,-1])
    return micro_f1, average_f1, weighted_f1


def train_loop(dataloader, model, optimizer, logger, batch_size):
    gradient_accumulations = 30
    size = len(dataloader.dataset)
    epoch_label = []
    epoch_output = []
    for batch, data in enumerate(dataloader):
        # Compute prediction and loss
        with autocast():
            output, labels, loss = model(data)

        # Backpropagation
        scaler.scale(loss / gradient_accumulations).backward()
        if (batch + 1) % gradient_accumulations == 0:
            logger.write(f"loss: {loss.item():>7f} [{(batch) * batch_size:>5d}/{size:>5d}]\n")
            print(f"loss: {loss.item():>7f} [{(batch) * batch_size}/{size:>5d}]")

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        epoch_label.append(labels)
        epoch_output.append(output)

    epoch_label = torch.concat(epoch_label).cpu()
    epoch_output = torch.concat(epoch_output).cpu()
    micro_f1, average_f1, weighted_f1 = calculate_metrics(epoch_label, epoch_output)
    logger.write(f"F1: Micro: {micro_f1}, Macro: {average_f1}, Weighted: {weighted_f1}\n")
    print(f"F1: Micro: {micro_f1}, Macro: {average_f1}, Weighted: {weighted_f1}\n")
    return loss


def test_loop(dataloader, model, logger):
    with torch.no_grad():
        test_label = []
        test_loss = []
        test_output = []
        for data in dataloader:
            output, labels, loss = model(data, test=True)

            test_loss.append(loss)
            test_label.append(labels)
            test_output.append(output)

        test_loss = torch.stack(test_loss).mean().cpu()
        test_output = torch.concat(test_output).cpu()
        test_label = torch.concat(test_label).cpu()

        micro_f1, average_f1, weighted_f1 = calculate_metrics(test_label, test_output)
        logger.write(f"F1: Micro: {micro_f1}, Macro: {average_f1}, Weighted: {weighted_f1}\n")
        print(f"F1: Micro: {micro_f1}, Macro: {average_f1}, Weighted: {weighted_f1}\n")

    print(f"loss: {test_loss:>8f} \n")
    logger.write(f"loss: {test_loss:>8f} \n")
    return loss
