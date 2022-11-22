
import torch
import torch.nn as nn
import torch.nn.functional as F
from randlanet import RandLANet
import math

class DLO_net(nn.Module):
    def __init__(self, config, device):
        super(DLO_net, self).__init__()
        self.encoder = RandLANet(d_in=3, num_neighbors=16, decimation=4, device=device)
        self.cross_attention = Transformer_Model()

    def forward(self, input):
        input = self.encoder(input)
        output = self.cross_attention(input)

        return output

if __name__=="__main__":
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #
    # d_in = 3
    # pc1 = 1000 * torch.randn(1, 2 ** 16, d_in).to(device)
    # pc2 = 1000 * torch.randn(1, 2 ** 16, d_in).to(device)
    #
    # backbone = RandLANet(d_in, 16, 4, device)
    # backbone.to(device)
    #
    # pc1_encoding = backbone(pc1)
    # pc2_encoding = backbone(pc2)
    #
    # cross_attention = Transformer_Model()
    # transformation_matrix = cross_attention(pc1_encoding, pc2_encoding)

    x = torch.randn((2, 10, 8))
    mask = torch.randn((2, 10)) > 0.5
    mask = mask.unsqueeze(1).unsqueeze(-1)
    num_heads = 4
    model = MHA(8,8, 8, num_heads)
    y = model(x, x, x, mask)
    print(y.shape)
    assert len(y.shape) == len(x.shape)
    for dim_x, dim_y in zip(x.shape, y.shape):
        assert dim_x == dim_y
    print(y.shape)




