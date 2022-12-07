
import torch
import torch.nn as nn
from models.backbone.randlanet import RandLANet
from models.cross_attention import Cross_Attention_Model
from models.transformer import Transformer_Model
import time
from utils.utils import compute_rigid_transform
import numpy as np
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

class Correspondance_Regressor(nn.Module):
    def __init__(self, cfg):
        super(Correspondance_Regressor, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(cfg.embed_size, cfg.embed_size),
            nn.ReLU(),
            nn.Linear(cfg.embed_size, cfg.embed_size),
            nn.ReLU(),
            nn.Linear(cfg.embed_size, 3)
        )

        self.overlap_score_decoder = nn.Sequential(
            nn.Linear(cfg.embed_size, 1),
            nn.Sigmoid())

    def forward(self, src, tgt):
        src_correspondence = self.model(src)
        tgt_correspondence = self.model(tgt)

        src_overlap = self.overlap_score_decoder(src)
        tgt_overlap = self.overlap_score_decoder(src)

        return src_correspondence, tgt_correspondence, src_overlap, tgt_overlap


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
        if cfg.use_random_sampling:
            self.backbone = None
            self.fc = nn.Linear(3, 512)
        else:
            self.backbone = RandLANet(d_in=3, num_neighbors=16, decimation=4,
                                      num_features=cfg.downsampled_features, device=device)

        self.cross_attention = Cross_Attention_Model(cfg, device)

        self.regress_T = cfg.regress_transformation
        if self.regress_T:
            self.regressor = FCN_regression(cfg)
        else:
            self.regressor = Correspondance_Regressor(cfg)

        self.device = device

    def forward(self, prev_input, curr_input):
        if self.backbone is not None:
            prev_frame_encoding, prev_frame_coords = self.backbone(prev_input['pointcloud'].to(self.device))
            curr_frame_encoding, curr_frame_coords = self.backbone(curr_input['pointcloud'].to(self.device))
        else:
            prev_frame_encoding, prev_frame_coords = self.fc(prev_input['pointcloud'][:,np.random.randint(0,
                                        len(prev_input['pointcloud']), 512),:].to(self.device))
            curr_frame_encoding, curr_frame_coords = self.fc(curr_input['pointcloud'][:,np.random.randint(0,
                                        len(curr_input['pointcloud']), 512),:].to(self.device))

        # print("prev_encoding: ",prev_frame_encoding.size())
        # print("curr_encoding: ",curr_frame_encoding.size())
        # print("prev_frame_coords: ", prev_frame_coords.size())
        # print("curr_frame_coords: ", curr_frame_coords.size())
        # raise ValueError
        attention_features = self.cross_attention(prev_frame_encoding, curr_frame_encoding)

        if self.regress_T:
            T = self.regressor(attention_features)
        else:
            src_cp, tgt_cp, src_overlap, tgt_overlap = self.regressor(attention_features[0], attention_features[1])

            src_features = torch.cat((prev_frame_coords, tgt_cp), dim=1)
            tgt_features = torch.cat((curr_frame_coords, src_cp), dim=1)
            overlap = torch.cat((src_overlap, tgt_overlap), dim=1)
            # print("src: ", src_features.size())
            # print("tgt: ", tgt_features.size())
            # print("overlap: ", overlap.size())
            T = compute_rigid_transform(src_features.squeeze(), tgt_features.squeeze(), overlap.squeeze()).unsqueeze(0)
            # print(T)
            # print(torch.linalg.det(T[:3,:3]))
            # raise ValueError


        return T

if __name__=="__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cfg = Config(512)




