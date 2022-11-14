#Todo; Create a model
#Todo: Write a dataloader for the model

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.query_ltf = nn.Linear(self.input_dim_Q, self.num_heads*self.input_dim_Q)
        self.key_ltf = nn.Linear(self.input_dim_K, self.num_heads*self.input_dim_K)
        self.value_ltf = nn.Linear(self.input_dim_V, self.num_heads*self.input_dim_V)

        # Define the output layer
        self.output = nn.Linear(self.num_heads*input_dim_Q, self.input_dim_Q) #todo: Check this mathematically.

    def forward(self, query:torch.Tensor, key:torch.Tensor, value:torch.Tensor, mask:torch.Tensor = None):
        """
        :param query: tensor of shape BxLxC where B is batch size, L is sequence length, C is channel dimension
        :param key: tensor of the shape BxLxC
        :param value: tensor of the shape BxLxC
        :param mask: tensor indicating where the attention should not be performed
        :return: output of the MHA module
        """

class FFN(nn.Module):
    def __init__(self, input_dim:int, ff_dim:int, dropout):
        super(FFN, self).__init__()

        # Define the FFN to be put in front of the multi-head attention module

    def forward(self, x:torch.Tensor):
        return x

class TransformerEncoderCell(nn.Module):
    def __init__(self, input_dim_Q:int, input_dim_K:int, input_dim_V:int, num_heads:int, ff_dim:int, dropout:float):
        super(TransformerEncoderCell, self).__init__()

    def forward(self, x: torch.Tensor, mask:torch.Tensor=None):
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim_Q:int, input_dim_K:int, input_dim_V:int, num_heads:int, ff_dim:int, dropout:float = 0.1):
        super(TransformerEncoder, self).__init__()

    def forward(self, x:torch.Tensor, mask:torch.Tensor = None):
        return x

class TransformerDecoderCell(nn.Module):
    def __init__(self, input_dim_Q: int, input_dim_K: int, input_dim_V: int, num_heads: int, ff_dim: int,
                 dropout: float):
        super(TransformerDecoderCell, self).__init__()

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,  src_mask: torch.Tensor = None, tgt_mask:torch.Tensor = None):
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, input_dim_Q:int, input_dim_K:int, input_dim_V:int, num_heads:int, ff_dim:int, dropout:float = 0.1):
        super(TransformerDecoder, self).__init__()

    def forward(self, x:torch.Tensor, encoder_output: torch.Tensor,  src_mask: torch.Tensor = None, tgt_mask:torch.Tensor = None):
        return x

class Transformer_Model(nn.Module):
    def __init__(self):
        super(Transformer_Model, self).__init__()
        pass

class KPConv(nn.Module):
    def __init__(self):
        super(KPConv, self).__init__()
        pass


class DLO_net(nn.Module):
    def __init__(self):
        super(DLO_net, self).__init__()
        pass

