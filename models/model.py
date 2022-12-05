
import torch
import torch.nn as nn
from models.backbone.randlanet import RandLANet
from models.cross_attention import Cross_Attention_Model
from models.transformer import Transformer_Model
import time
from config.config import Config

class FCN_regression(nn.Module):
    def __init__(self, cfg):
        super(FCN_regression, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.input_dim),
            nn.ReLU(),
            nn.Linear(cfg.input_dim, cfg.input_dim),
            nn.ReLU(),
            nn.Linear(cfg.input_dim, cfg.output_dim)
        )

    def forward(self, x):
        return self.model(x)



class DLO_net_single(nn.Module):
    def __init__(self, cfg, device):
        super(DLO_net_single, self).__init__()
        self.backbone = RandLANet(d_in=3, num_neighbors=16, decimation=4, device=device)
        self.cross_attention = Cross_Attention_Model(cfg, device)
        self.regressor = FCN_regression(cfg)

        self.prev_frame_encoding = None
        self.device = device

    def forward(self, input):
        if input['frame_num']==0:
            self.prev_frame_encoding = self.backbone(input['pointcloud'].to(self.device)).squeeze(3)
            return torch.Tensor([[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]])
        else:
            curr_frame_encoding = self.backbone(input['pointcloud'].to(self.device)).squeeze(3)
            print(self.prev_frame_encoding)

            attention_features = self.cross_attention(self.prev_frame_encoding.transpose(1,2),
                                     curr_frame_encoding.transpose(1,2))

            T = self.regressor(attention_features)

            with torch.no_grad():
                self.prev_frame_encoding = curr_frame_encoding

        return T

class DLO_net(nn.Module):
    def __init__(self, cfg, device):
        super(DLO_net, self).__init__()
        self.backbone = RandLANet(d_in=3, num_neighbors=16, decimation=4, device=device)
        self.cross_attention = Cross_Attention_Model(cfg, device)
        self.regressor = FCN_regression(cfg)

        self.device = device

    def forward(self, prev_input, curr_input):
        prev_frame_encoding = self.backbone(prev_input['pointcloud'].to(self.device)).squeeze(3)
        curr_frame_encoding = self.backbone(curr_input['pointcloud'].to(self.device)).squeeze(3)
        attention_features = self.cross_attention(prev_frame_encoding.transpose(1,2), curr_frame_encoding.transpose(1,2))
        T = self.regressor(attention_features)

        return T

if __name__=="__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cfg = Config(512)




