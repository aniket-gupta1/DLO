import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from randlanet import RandLANet
from cross_attention import Cross_Attention_Model, config
from kitti import kitti
from transformer import Transformer_Model
from torch.utils.tensorboard import SummaryWriter
import time
from model import DLO_net
from lietorch import SE3


def parse_arguments():
    pass

def train_epoch(model, optimizer, dataset,  loss_fn, writer):
    global prev_data
    model.train()

    losses = 0
    dataloader = DataLoader(dataset, batch_size=1)

    for index, data in enumerate(dataloader):
        # print(data['pointcloud'].size())
        # print(data['pose'])
        # print(data['frame_num'])
        # print(data['seq'])
        #
        # time.sleep(200000)
        print(f"Frame {data['frame_num']} has pointcloud of shape {data['pointcloud'].shape}")

        if data['frame_num']==0:
            prev_data = data
        else:
            T = model(prev_data, data)
            # print(f"T: {T}")
            optimizer.zero_grad()

            pose = data['pose'].type(torch.float32).to(device)

            loss = loss_fn(T, pose)

            loss.backward(retain_graph=False)
            optimizer.step()

            losses += loss.item()

            writer.add_scalar("Batch Loss/train", loss, index)
            print(f"Batch_index: {index} || Loss: {loss}")

    return losses/(len(dataloader)-1)

def train(cfg, device, writer):
    dataset = kitti(cfg, mode="training")
    net = DLO_net(cfg, device).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                                  betas=(0.9, 0.98), eps=1e-9)

    loss_fn = torch.nn.MSELoss()

    for epoch in range(cfg.num_epochs):
        tic = time.time()
        loss = train_epoch(net, optimizer, dataset, loss_fn, writer)
        writer.add_scalar("Loss", loss, epoch)
        print("===========================================================")
        print(f"Epoch: {epoch} || Loss: {loss} || Time: {time.time()-tic}")
        print("===========================================================")

if __name__=="__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter()

    cfg = config(512)
    train(cfg, device, writer)
    writer.close()