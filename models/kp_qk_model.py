
import torch
import torch.nn as nn
from models.cross_attention import Cross_Attention_Model
import time
from utils.utils import compute_rigid_transform, pad_sequence
import numpy as np
from models.backbone.backbone_kpconv.kpconv import KPFEncoder, PreprocessorGPU, compute_overlaps

def softmax_correlation(feat0:torch.Tensor, feat1:torch.Tensor, pts0: torch.Tensor, pts1: torch.Tensor):
    b, n, d = feat0.shape
    _, m, _ = feat1.shape

    print(b, n, d, m)

    correlation = torch.matmul(feat0, feat1.permute(0, 2, 1)) / (d ** 0.5)  # [B, N, M]
    prob = torch.nn.functional.softmax(correlation, dim=-1)  # [B, N, M]

    if n>m:
        val, ind = torch.max(prob, dim=1)  # [B,M]
        # pts0 -> [B, N, 3] ; pts1 -> [B, M, 3]
        # val -> [B,M] ; ind -> [B,M]
        src_pts = torch.gather(pts0, 1, ind.unsqueeze(-1).expand(-1, -1, 3))  # [B, N, 3] -> [B, M, 3]
        print("src_pts shape: ", src_pts.shape)
        tgt_pts = pts1
        print("tgt_pts.shape", tgt_pts.shape)
        T = compute_rigid_transform(src_pts, tgt_pts, weights=val)

    else:
        val, ind = torch.max(prob, dim=2)  # [B,N]
        # pts0 -> [B, N, 3] ; pts1 -> [B, M, 3]
        # val -> [B,N] ; ind -> [B,N]
        tgt_pts = torch.gather(pts1, 1, ind.unsqueeze(-1).expand(-1, -1, 3))  # [B, M, 3] -> [B, N, 3]
        print("src_pts shape: ", tgt_pts.shape)
        src_pts = pts0
        print("tgt_pts.shape", tgt_pts.shape)
        T = compute_rigid_transform(src_pts, tgt_pts, weights=val)

    return T

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

        prev_frame_encoding, prev_frame_encoding_mask, _ = pad_sequence(prev_frame_encoding, require_padding_mask=True, batch_first=True)
        curr_frame_encoding, curr_frame_encoding_mask, _ = pad_sequence(curr_frame_encoding, require_padding_mask=True, batch_first=True)

        padded_prev_coords = torch.nn.utils.rnn.pad_sequence(prev_frame_coords, batch_first=True)
        padded_curr_coords = torch.nn.utils.rnn.pad_sequence(curr_frame_coords, batch_first=True)

        print(f"prev_enc: {prev_frame_encoding.shape}")
        print(f"curr_enc: {curr_frame_encoding.shape}")

        print(f"prev_enc_mask: {prev_frame_encoding_mask.shape}")
        print(f"curr_enc_mask: {curr_frame_encoding_mask.shape}")

        print(f"prev_pad_coords: {padded_prev_coords.shape}")
        print(f"curr_pad_coords: {padded_curr_coords.shape}")

        attention_features = self.cross_attention(prev_frame_encoding, curr_frame_encoding,
                                                  prev_frame_encoding_mask, curr_frame_encoding_mask)

        print(f"attention features 1: {attention_features[0].shape}")
        print(f"attention features 2: {attention_features[1].shape}")

        T = softmax_correlation(attention_features[0], attention_features[1], padded_prev_coords, padded_curr_coords)
        print("calculated T: ", T)
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



