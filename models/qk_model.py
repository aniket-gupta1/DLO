
import torch
import torch.nn as nn
from models.backbone.randlanet import RandLANet
from models.cross_attention import Cross_Attention_Model
from models.transformer import Transformer_Model
import time
from utils.utils import compute_rigid_transform
import numpy as np
from config.config import Config

def softmax_correlation(feat0:torch.Tensor, feat1:torch.Tensor):
    b, n, d = feat0.shape

    correlation = torch.matmul(feat0, feat1.permute(0,2,1))/(d**0.5) #[B, N, N]
    prob = torch.nn.functional.softmax(correlation, dim=-1) #[B, N, N]
    init_grid = torch.arange(n).float().cuda().requires_grad_() #[B, N]

    correspondence = torch.matmul(prob, init_grid) #[B, N]

    return correspondence

class DLO_net_single(nn.Module):
    def __init__(self, cfg, device):
        super(DLO_net_single, self).__init__()
        self.cfg = cfg

        self.backbone = RandLANet(d_in=3, num_neighbors=16, decimation=4,
                                  num_features=cfg.downsampled_features, device=device)

        self.cross_attention = Cross_Attention_Model(cfg, device)

        # self.matcher = Feature_matcher(cfg)

        self.device = device

        self.prev_frame_encoding = None
        self.prev_frame_coords = None

    def forward(self, curr_input):
        if curr_input['frame_num']==0:
            self.prev_frame_encoding, self.prev_frame_coords = self.backbone(curr_input['pointcloud'].to(self.device))
            return torch.eye(4)

        curr_frame_encoding, curr_frame_coords = self.backbone(curr_input['pointcloud'].to(self.device))

        attention_features = self.cross_attention(self.prev_frame_encoding, curr_frame_encoding)
        # print(attention_features)
        cp_ind_1t2 = softmax_correlation(attention_features[0], attention_features[1]).long()
        cp_1t2 = torch.gather(curr_frame_coords, 1, cp_ind_1t2.unsqueeze(-1).expand(-1, -1, curr_frame_coords.size(-1)))

        cp_ind_2t1 = softmax_correlation(attention_features[1], attention_features[0]).long()
        cp_2t1 = torch.gather(self.prev_frame_coords, 1, cp_ind_2t1.unsqueeze(-1).expand(-1, -1, self.prev_frame_coords.size(-1)))

        # print(cp_2t1)
        T = compute_rigid_transform(cp_1t2, cp_2t1)


        # src_cp, tgt_cp, src_overlap, tgt_overlap = self.regressor(attention_features[0], attention_features[1])
        #
        # src_features = torch.cat((self.prev_frame_coords, tgt_cp), dim=1)
        # tgt_features = torch.cat((curr_frame_coords, src_cp), dim=1)
        # overlap = torch.cat((src_overlap, tgt_overlap), dim=1)
        #
        # T = compute_rigid_transform(src_features.squeeze(), tgt_features.squeeze(), overlap.squeeze()).unsqueeze(0)

        self.prev_frame_encoding = curr_frame_encoding.detach()
        self.prev_frame_coords = curr_frame_coords.detach()
        return T

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


if __name__=="__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # cfg = Config(512)
    a = torch.randn(2, 10, 3)
    b = torch.randn(2, 10, 3)
    c = softmax_correlation(a,b)



