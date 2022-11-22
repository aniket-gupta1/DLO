
import torch
import torch.nn as nn
import torch.nn.functional as F
from randlanet import RandLANet
import math

class MHA(nn.Module):
    def __init__(self, input_dim_Q:int, input_dim_K:int, input_dim_V:int, num_heads:int):
        """
        :param input_dim_Q: dimensions of input query
        :param input_dim_K: dimensions of input key
        :param input_dim_V: dimensions of input value
        :param num_heads: number of attention heads
        """
        super(MHA, self).__init__()

        assert input_dim_Q%num_heads==0 and input_dim_K%num_heads==0 and input_dim_V%num_heads==0

        self.input_dim_Q = input_dim_Q
        self.input_dim_K = input_dim_K
        self.input_dim_V = input_dim_V
        self.num_heads = num_heads

        self.dim_per_head = input_dim_Q//num_heads #todo Check this from the maths

        # Define the linear transformation layers for key, value and query
        self.wq = nn.Linear(self.input_dim_Q, self.input_dim_Q)
        self.wk = nn.Linear(self.input_dim_K, self.input_dim_K)
        self.wv = nn.Linear(self.input_dim_V, self.input_dim_V)

        # Define the output layer
        self.output = nn.Linear(self.input_dim_V, self.input_dim_V) #todo: Check this mathematically.

    def forward(self, query:torch.Tensor, key:torch.Tensor, value:torch.Tensor, mask:torch.Tensor = None):
        """
        :param query: tensor of shape BxLxC where B is batch size, L is sequence length, C is channel dimension
        :param key: tensor of the shape BxLxC
        :param value: tensor of the shape BxLxC
        :param mask: tensor indicating where the attention should not be performed
        :return: output of the MHA module
        """
        B = query.size(0)
        C = query.size(2)

        query_reshaped = self.wq(query).view(B, -1, self.num_heads, self.dim_per_head)
        key_reshaped = self.wk(key).view(B, -1, self.num_heads, self.dim_per_head)
        value_reshaped = self.wv(value).view(B, -1, self.num_heads, self.dim_per_head)

        dot_prod_scores = torch.matmul(query_reshaped.transpose(1, 2),
                                       key_reshaped.transpose(1, 2).transpose(2, 3)) / math.sqrt(C)

        if mask is not None:
            # We simply set the similarity scores to be near zero for the positions
            # where the attention should not be done. Think of why we do this.
            dot_prod_scores = dot_prod_scores.masked_fill(mask == 0, -1e9)

        attention_scores = F.softmax(dot_prod_scores, dim=-1)
        modulated_scores = torch.matmul(attention_scores, value_reshaped.transpose(1, 2))
        modulated_scores = modulated_scores.transpose(1, 2)
        modulated_scores = modulated_scores.reshape(B, -1, self.num_heads * self.dim_per_head)

        out = self.output(modulated_scores)

        return out

class FFN(nn.Module):
    def __init__(self, input_dim:int, ff_dim:int, dropout):
        super(FFN, self).__init__()

        # Define the FFN to be put in front of the multi-head attention module
        self.fc1 = nn.Linear(input_dim, ff_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(ff_dim, input_dim)

        self.dropout0 = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.norm3 = nn.LayerNorm(input_dim)


    def forward(self, x:torch.Tensor):
        y = self.fc2(self.relu(self.fc1(x)))
        return y

class Cross_EncoderCell(nn.Module):
    def __init__(self, input_dim_Q:int, input_dim_K:int, input_dim_V:int, num_heads:int, ff_dim:int, dropout:float):
        super(Cross_EncoderCell, self).__init__()
        self.attn = MHA(input_dim_Q, input_dim_K, input_dim_V, num_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(input_dim_Q)

        self.fc_model = FFN(input_dim_Q, ff_dim, dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm2 = nn.LayerNorm(input_dim_Q)

    def forward(self, x: torch.Tensor, mask:torch.Tensor=None):
        y_attn = self.attn(x, x, x, mask)
        y_attn = self.dropout1(y_attn)
        y_attn = self.layer_norm1(y_attn + x)

        y = self.fc_model(y_attn)
        y = self.dropout2(y)
        y = self.layer_norm2(y_attn + y)

        return y

class Cross_Encoder(nn.Module):
    def __init__(self, input_dim_Q:int, input_dim_K:int, input_dim_V:int, num_heads:int, ff_dim:int, num_cells:int, dropout:float = 0.1):
        super(Cross_Encoder, self).__init__()
        self.model = nn.ModuleList(
            [Cross_EncoderCell(input_dim_Q, input_dim_K, input_dim_V, num_heads, ff_dim, dropout) for _ in range(num_cells)])
        self.layer_norm = nn.LayerNorm(input_dim_Q)

    def forward(self, x:torch.Tensor, mask:torch.Tensor = None):
        for layer in self.model:
            x = layer(x, mask)

        y = self.layer_norm(x)
        return y


class PositionalEncoding(nn.Module):
    def __init__(self, input_dim:int, device, max_len:int=10000):
        super(PositionalEncoding, self).__init__()

        self.input_dim = input_dim
        self.max_len = max_len
        self.device = device

    def forward(self, x):
        L = x.size(1)
        N = x.size(2)

        norm = self.max_len ** (torch.arange(0, self.input_dim, 2)/N)
        pos = torch.arange(L).unsqueeze(1)
        pe = torch.zeros(L,N)
        pe [:,::2] = torch.sin(pos/norm)
        pe [:,1::2] = torch.cos(pos/norm)

        x = x + pe.to(self.device)

        return x

class Transformer_Model(nn.Module):
    def __init__(self, input_dim, device):
        super(Transformer_Model, self).__init__()

        self.pe = PositionalEncoding(input_dim, device)
        self.encoder = TransformerEncoder(dim_Q, dim_k, dim_V, num_heads, ff_dim, num_cells, dropout)
        self.decoder = TransformerDecoder(dim_Q, dim_k, dim_V, num_heads, ff_dim, num_cells, dropout)
        self.output = nn.Linear(dim_Q, target)

    def forward(self, src, tgt, src_mask, tgt_mask):
        x_src = self.pe(src)
        x_tgt = self.pe(tgt)

        encoder_output = self.encoder(x_src, src_mask)
        decoder_output = self.decoder(x_tgt, encoder_output, src_mask, tgt_mask)

        logits = self.output(decoder_output)

        return logits



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




