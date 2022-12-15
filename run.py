from torch.utils.data import DataLoader
import torch
import argparse
from model import Model
from train import train_loop, test_loop
from data_loader import DocRED
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", default="30", type=int)
    parser.add_argument("--lr", default="0.0001", type=float)
    parser.add_argument("--batch-size", default="3", type=int)
    parser.add_argument("--use_negative", default=False, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    args = parser.parse_args([] if "__file__" not in globals() else None)

    logger = open("log.log", "a")
    device = torch.device(args.device)
    model_path = "checkpoints/"
    current_epoch = 0

    train_data = DocRED("dataset/train_annotated.json", args.use_negative)
    test_data = DocRED("dataset/dev.json", args.use_negative)

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size,
                                  shuffle=True, collate_fn=train_data.custom_collate_fn)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size,
                                 shuffle=True, collate_fn=test_data.custom_collate_fn)

    model = Model(train_data.get_token_embedding(), use_negative=args.use_negative, device=device)
    model.to(device)

    #####
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=1, min_lr=1e-6)
    #####
    if os.path.exists(model_path):
        if len(os.listdir(model_path)) > 0:
            file_name = os.listdir(model_path)[-1]
            checkpoint = torch.load(f"{model_path}{file_name}")
            model.load_state_dict(checkpoint['model_state_dict'])
            model.train()
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            current_epoch = checkpoint['epoch']
            loss = checkpoint['loss']
    else:
        os.mkdir(model_path)

    for epoch in range(current_epoch, args.epoch):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        logger.write(f"Epoch {epoch + 1}\n-------------------------------\n")
        train_loss = train_loop(train_dataloader, model, optimizer,logger)
        test_loss = test_loop(test_dataloader, model, logger)
        scheduler.step(test_loss)
        if epoch > 15:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'loss': train_loss,
            }, f"{model_path}{epoch}.pt")
    print("Done!")
