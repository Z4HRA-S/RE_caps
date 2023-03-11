from torch import nn
import torch
import torch.nn.functional as F


class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=12, stride=3)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=12, stride=3)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.5)
        self.linear1 = nn.Linear(648, 128)
        self.linear2 = nn.Linear(128, 1)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.linear1.weight)
        # self.loss = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, x):
        # x = x.unsqueeze(1)
        output = nn.LeakyReLU()(self.conv1(x))  # 8 *  253
        output = nn.LeakyReLU()(self.conv2(output))  # 8 * 81
        output = self.flatten(output)  # 648
        output = nn.LeakyReLU()(self.linear1(output))
        output = self.dropout(output)
        output = self.linear2(output)
        return output

    def loss_func(self, logits, labels, alpha=0.75, gamma=2):
        # return self.loss(logits.squeeze(), labels.any(dim=1).float())
        return self.sigmoid_focal_loss(logits.squeeze(), labels.any(dim=1).float(), reduction="mean")

    def sigmoid_focal_loss(
            self,
            inputs: torch.Tensor,
            targets: torch.Tensor,
            alpha: float = 0.75,
            gamma: float = 2,
            reduction: str = "none",
    ):
        """
        Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        """
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

        return loss
