
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from randlanet import RandLANet
from cross_attention import Cross_Attention_Model, config
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
            curr_frame_encoding = self.backbone(input['pointcloud'].to(self.device)).squeeze(3)
            # print(f"Prev size: {self.prev_frame_encoding.size()}")
            # print(f"Curr size: {self.curr_frame_encoding.size()}")
            # print(f"permuted: {self.prev_frame_encoding.transpose(1,2).size()}")
            print(self.prev_frame_encoding)

            T = self.cross_attention(self.prev_frame_encoding.transpose(1,2),
                                     curr_frame_encoding.transpose(1,2))
            with torch.no_grad():
                self.prev_frame_encoding = curr_frame_encoding

        return T

if __name__=="__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    cfg = config(512)
    train(cfg, device)




