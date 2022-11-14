import torch
import os
import time
import argparse
from model import DLO_net
from kitti import kitti
from torch.utils.data import DataLoader
from lietorch.lietorch import SE3


def parse_arguments():
    pass

def train_epoch(model, dataloader, optimizer):
    model.train()

    for data in enumerate(dataloader):
        optimizer.zero_grad()

        point_clouds, poses = [x.cuda().float() for x in data]

        poses = SE3(poses).inv()
        out = model()
        loss =
        loss.backward()

        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()

    return None

def evaluate(model, dataloader):
    model.eval()

    with torch.no_grad():
        for data in enumerate(dataloader):
            acc = 0.0

    print(f'Validation Accuracy: {acc}')
    return None

def train(args, device):
    dataset = kitti(None, "training")
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    model = DLO_net()
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.steps, pct_start=0.01,
                                                    cycle_momentum=False, anneal_strategy='Linear')

    for epoch in range(1, args.epoch + 1):
        tic = time.time()
        train_epoch(model, dataloader, optimizer)
        print(f"Time taken: {time.time()-tic}")

        evaluate(model, dataloader)

        scheduler.step(epoch)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--epochs', type=int, default=1000)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(args, device)