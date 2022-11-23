
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from randlanet import RandLANet
from transformer_cross_attention import Cross_Attention_Model, config
from kitti import kitti
from transformer import Transformer_Model
import time

class DLO_net(nn.Module):
    def __init__(self, cfg, device):
        super(DLO_net, self).__init__()
        self.backbone = RandLANet(d_in=3, num_neighbors=16, decimation=4, device=device)
        self.cross_attention = Cross_Attention_Model(cfg, device)

        self.prev_frame_encoding = None
        self.device = device

    def forward(self, input):
        if input['frame_num']==0:
            self.prev_frame_encoding = self.backbone(input['pointcloud'].to(self.device)).squeeze(3)
            return torch.Tensor([[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]])
        else:
            self.curr_frame_encoding = self.backbone(input['pointcloud'].to(self.device)).squeeze(3)
            # print(f"Prev size: {self.prev_frame_encoding.size()}")
            # print(f"Curr size: {self.curr_frame_encoding.size()}")
            # print(f"permuted: {self.prev_frame_encoding.transpose(1,2).size()}")

            T = self.cross_attention(self.prev_frame_encoding.transpose(1,2),
                                     self.curr_frame_encoding.transpose(1,2))
            self.prev_frame_encoding = self.curr_frame_encoding

        return T

def train_epoch(model, optimizer, dataset,  loss_fn):
    model.train()

    losses = 0
    dataloader = DataLoader(dataset, batch_size=1)

    for data in dataloader:
        # print(data['pointcloud'].size())
        # print(data['pose'])
        # print(data['frame_num'])
        # print(data['seq'])
        #
        # time.sleep(200000)
        print(f"Frame {data['frame_num']} has pointcloud of shape {data['pointcloud'].shape}")
        T = model(data)

        if data['frame_num']==0:
            continue
        else:
            optimizer.zero_grad()

            pose = data['pose'].type(torch.float32).to(device)

            loss = loss_fn(T, pose)

            loss.backward(retain_graph=False)
            optimizer.step()

            losses += loss.item()

    return losses/(len(dataloader)-1)

def train(cfg, device):
    dataset = kitti(cfg, mode="training")
    net = DLO_net(cfg, device).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                                  betas=(0.9, 0.98), eps=1e-9)

    loss_fn = torch.nn.MSELoss()

    for epoch in range(cfg.num_epochs):
        tic = time.time()
        loss = train_epoch(net, optimizer, dataset, loss_fn)
        print(f"Epoch: {epoch} || Loss: {loss} || Time: {time.time()-tic}")

if __name__=="__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    cfg = config(512)
    train(cfg, device)




