from torch.utils.data import DataLoader
import torch
import argparse
from model import Model
from train import train_loop, test_loop
from data_loader import DocRED

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", default="30", type=int)
    parser.add_argument("--lr", default="0.01", type=float)
    parser.add_argument("--batch-size", default="3", type=int)
    parser.add_argument("--num_class", default="96", type=int)
    parser.add_argument("--device", default="cpu", type=str)
    args = parser.parse_args([] if "__file__" not in globals() else None)

    device = torch.device(args.device)

    train_data = DocRED("dataset/train_annotated.json", args.num_class)
    test_data = DocRED("dataset/dev.json", args.num_class)

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size,
                                  shuffle=True, collate_fn=train_data.custom_collate_fn)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size,
                                 shuffle=True, collate_fn=test_data.custom_collate_fn)

    model = Model(train_data.get_token_embedding(), num_class=args.num_class, device=device)
    model.to(device)

    #####
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=15, min_lr=1e-6)
    #####

    for t in range(args.epoch):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, optimizer)
        test_loss = test_loop(test_dataloader, model)
        scheduler.step(test_loss)
    print("Done!")
