
import torch
import torch.nn as nn
from models.cross_attention import Cross_Attention_Model
import time
from utils.utils import compute_rigid_transform, pad_sequence
import numpy as np
from models.backbone.backbone_kpconv.kpconv import KPFEncoder, PreprocessorGPU, compute_overlaps

def softmax_correlation(feat0:torch.Tensor, feat1:torch.Tensor, pts0, pts1):
    b, n, d = feat0.shape
    _, m, _ = feat1.shape

    print(b, n, d, m)

    if n>m:
        correlation = torch.matmul(feat0, feat1.permute(0,2,1))/(d**0.5) #[B, N, M]
        print(correlation.shape)
        prob = torch.nn.functional.softmax(correlation, dim=-1) #[B, N, M]
        print(prob.shape)
        val_max, ind_max = torch.max(prob, dim=-1)
        val_m, ind_m = torch.topk(val_max, m)
        print(f"val shape: {val_m.shape}")
        print(f"ind shape: {ind_m.shape}")

        cp_pts0 = []
        for i, pts in enumerate(pts0):
            cp_pts0.append(torch.gather(pts, 1, ind_m))



        init_grid = torch.arange(m).float().cuda().requires_grad_() #[B, N]

    else:
        correlation = torch.matmul(feat1, feat0.permute(0, 2, 1)) / (d ** 0.5) # [B, M, N]
        print(correlation.shape)
        prob = torch.nn.functional.softmax(correlation, dim=-1)  # [B, M, N]
        print(prob.shape)
        init_grid = torch.arange(n).float().cuda().requires_grad_()  # [B, N]

    correspondence = torch.matmul(prob, init_grid) #[B, N]

    return correspondence

def split_src_tgt(feats, stack_lengths, dim=0):
    if isinstance(stack_lengths, torch.Tensor):
        stack_lengths = stack_lengths.tolist()

    B = len(stack_lengths) // 2
    separate = torch.split(feats, stack_lengths, dim=dim)
    return separate[:B], separate[B:]

class DLO_net_single(nn.Module):
    def __init__(self, cfg, device):
        super(DLO_net_single, self).__init__()
        self.cfg = cfg

        self.preprocessor = PreprocessorGPU(cfg)
        self.backbone = KPFEncoder(cfg, cfg.input_dim)
        self.feat_proj = nn.Linear(self.backbone.encoder_skip_dims[-1], cfg.input_dim, bias=True)

        self.cross_attention = Cross_Attention_Model(cfg, device)

        self.device = device

        self.prev_frame_encoding = None
        self.prev_frame_coords = None

    def forward(self, batch):
        B = len(batch['src_xyz'])

        kpconv_meta = self.preprocessor(batch['src_xyz'] + batch['tgt_xyz'])
        batch['kpconv_meta'] = kpconv_meta
        slens = [s.tolist() for s in kpconv_meta['stack_lengths']]
        slens_c = slens[-1]
        feats0 = torch.ones_like(kpconv_meta['points'][0][:, 0:1]).cuda()
        print(f"feats0: {feats0.shape}")
        # Pass through KP_Conv encoder
        feats_un, skip_x = self.backbone(feats0, kpconv_meta)
        print(f"feats_un: {feats_un.shape}")
        print(len(skip_x))
        print(skip_x[0].shape)
        both_feats_un = self.feat_proj(feats_un)
        print(f"both_feats_un: {both_feats_un.shape}")
        # print("both_feats_un: ", both_feats_un.size())
        prev_frame_encoding, curr_frame_encoding = split_src_tgt(both_feats_un, slens_c)
        print(f"prev_enc 1: {prev_frame_encoding[0].shape}")
        print(f"prev_enc 2: {prev_frame_encoding[1].shape}")
        print(f"curr_enc 1: {curr_frame_encoding[0].shape}")
        print(f"curr_enc 2: {curr_frame_encoding[1].shape}")
        prev_frame_coords, curr_frame_coords = split_src_tgt(kpconv_meta['points'][-1], slens_c)
        print(f"prev coords 1: {prev_frame_coords[0].shape}")
        print(f"prev coords 2: {prev_frame_coords[1].shape}")
        print(f"curr coords 1: {curr_frame_coords[0].shape}")
        print(f"curr coords 2: {curr_frame_coords[1].shape}")

        prev_frame_encoding, _, _ = pad_sequence(prev_frame_encoding, require_padding_mask=False, batch_first=True)
        curr_frame_encoding, _, _ = pad_sequence(curr_frame_encoding, require_padding_mask=False, batch_first=True)

        print(f"prev_enc: {prev_frame_encoding.shape}")
        print(f"curr_enc: {curr_frame_encoding.shape}")

        attention_features = self.cross_attention(prev_frame_encoding, curr_frame_encoding)

        print(f"attention features 1: {attention_features[0].shape}")
        print(f"attention features 2: {attention_features[1].shape}")

        cp_ind_1t2 = softmax_correlation(attention_features[0], attention_features[1], prev_frame_coords, curr_frame_coords).long()
        cp_1t2 = torch.gather(curr_frame_coords, 1, cp_ind_1t2.unsqueeze(-1).expand(-1, -1, curr_frame_coords.size(-1)))

        # cp_ind_2t1 = softmax_correlation(attention_features[1], attention_features[0]).long()
        # cp_2t1 = torch.gather(prev_frame_coords, 1, cp_ind_2t1.unsqueeze(-1).expand(-1, -1, self.prev_frame_coords.size(-1)))

        T = compute_rigid_transform(cp_1t2, cp_2t1)

        return T

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


if __name__=="__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # cfg = Config(512)
    a = torch.randn(2, 10, 3).cuda()
    b = torch.randn(2, 10, 3).cuda()
    c = softmax_correlation(a,b)
    print(c.shape)



