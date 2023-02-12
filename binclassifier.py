from torch import nn


class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(768, 32, 2, dropout=0.2)
        self.classifier = nn.Sequential(nn.Linear(32 * 5, 1),
                                        nn.Sigmoid()
                                        )
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        output, (hn, cn) = self.lstm(x)
        output = output.view(output.size(0), output.size(1) * output.size(2))
        output = self.classifier(output)
        return output

    def loss_fuc(self, logits, labels):
        return self.loss(logits.squeeze(), labels.any(dim=1).float())
