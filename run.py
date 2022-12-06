from torch.utils.data import DataLoader
import torch
from model import Model
from train import train_loop, test_loop
from data_loader import DocRED

device = torch.device("cpu")
epochs = 30
batch_size = 2

train_data = DocRED("dataset/train_annotated.json")
test_data = DocRED("dataset/dev.json")

train_dataloader = DataLoader(train_data, batch_size=batch_size,
                              shuffle=True, collate_fn=train_data.custom_collate_fn)
test_dataloader = DataLoader(test_data, batch_size=batch_size,
                             shuffle=True, collate_fn=test_data.custom_collate_fn)

model = Model(train_data.get_token_embedding(),device=device)
model.to(device)

#####
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=15, min_lr=1e-6)
#####

for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_dataloader, model, optimizer)
    test_loss=test_loop(test_dataloader, model)
    scheduler.step(test_loss)
print("Done!")
