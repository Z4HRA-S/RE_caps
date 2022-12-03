from torch.utils.data import DataLoader
import torch
from model import Model
from train import train_loop, test_loop
from data_loader import DocRED
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

device = torch.device("cuda:0")
epochs = 30
batch_size = 5

loss_fn = balanced_loss()
loss_fn.to(device)

training_data = DocRED("dataset/train_annotated.json")
train_size = int(0.8 * len(training_data))
test_size = len(training_data) - train_size
train_data, test_data = torch.utils.data.random_split(training_data, [train_size, test_size])

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

model = Model()
model.to(device)

#####
warmup_ratio = 0.06
learning_rate = 4e-4
adam_epsilon = 1e-6
bert_lr = 3e-5
total_steps = int(len(train_dataloader) * epochs // 1)
warmup_steps = int(total_steps * warmup_ratio)

cur_model = model.module if hasattr(model, 'module') else model
extract_layer = ["extractor", "bilinear"]
bert_layer = ['bert_model']
optimizer_grouped_parameters = [
    {"params": [p for n, p in cur_model.named_parameters() if any(nd in n for nd in bert_layer)], "lr": bert_lr},
    {"params": [p for n, p in cur_model.named_parameters() if any(nd in n for nd in extract_layer)], "lr": 1e-4},
    {"params": [p for n, p in cur_model.named_parameters() if not any(nd in n for nd in extract_layer + bert_layer)]},
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
#####

for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer, scheduler)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
