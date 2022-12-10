import torch
import torch.nn as nn
import torch.nn.functional as F
import pt_utils
from typing import List, Tuple
import numpy as np
import pickle

class Network(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc0 = Conv1d(3, 8, kernel_size=1, bn=True)

        self.dilated_res_blocks = nn.ModuleList()
        d_in = 8

        d_out_list = [16, 64, 128, 256]
        for i in range(4):
            d_out = d_out_list[i]
            self.dilated_res_blocks.append(Dilated_res_block(d_in, d_out))
            d_in = 2 * d_out

        d_out = d_in
        self.decoder_0 = Conv2d(d_in, d_out, kernel_size=(1,1), bn=True)


    def forward(self, end_points):

        features = end_points['features']  # Batch*channel*npoints
        features = self.fc0(features)

        features = features.unsqueeze(dim=3)  # Batch*channel*npoints*1

        # ###########################Encoder############################
        f_encoder_list = []
        for i in range(4):
            f_encoder_i = self.dilated_res_blocks[i](features, end_points['xyz'][i], end_points['neigh_idx'][i])

            f_sampled_i = self.random_sample(f_encoder_i, end_points['sub_idx'][i])
            features = f_sampled_i
            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)
        # ###########################Encoder############################

        features = self.decoder_0(f_encoder_list[-1])

        return end_points

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        num_neigh = pool_idx.shape[-1]
        d = feature.shape[1]
        batch_size = pool_idx.shape[0]
        pool_idx = pool_idx.reshape(batch_size, -1)  # batch*(npoints,nsamples)
        pool_features = torch.gather(feature, 2, pool_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))
        pool_features = pool_features.reshape(batch_size, d, -1, num_neigh)
        pool_features = pool_features.max(dim=3, keepdim=True)[0]  # batch*channel*npoints*1
        return pool_features


class Dilated_res_block(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

        self.mlp1 = Conv2d(d_in, d_out//2, kernel_size=(1,1), bn=True)
        self.lfa = Building_block(d_out)
        self.mlp2 = Conv2d(d_out, d_out*2, kernel_size=(1, 1), bn=True, activation=None)
        self.shortcut = Conv2d(d_in, d_out*2, kernel_size=(1,1), bn=True, activation=None)

    def forward(self, feature, xyz, neigh_idx):
        f_pc = self.mlp1(feature)  # Batch*channel*npoints*1
        f_pc = self.lfa(xyz, f_pc, neigh_idx)  # Batch*d_out*npoints*1
        f_pc = self.mlp2(f_pc)
        shortcut = self.shortcut(feature)
        return F.leaky_relu(f_pc+shortcut, negative_slope=0.2)

class Building_block(nn.Module):
    def __init__(self, d_out):  #  d_in = d_out//2
        super().__init__()
        self.mlp1 = Conv2d(10, d_out//2, kernel_size=(1,1), bn=True)
        self.att_pooling_1 = Att_pooling(d_out, d_out//2)

        self.mlp2 = Conv2d(d_out//2, d_out//2, kernel_size=(1, 1), bn=True)
        self.att_pooling_2 = Att_pooling(d_out, d_out)

    def forward(self, xyz, feature, neigh_idx):  # feature: Batch*channel*npoints*1
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx)  # batch*npoint*nsamples*10
        f_xyz = f_xyz.permute((0, 3, 1, 2))  # batch*10*npoint*nsamples
        f_xyz = self.mlp1(f_xyz)
        f_neighbours = self.gather_neighbour(feature.squeeze(-1).permute((0, 2, 1)), neigh_idx)  # batch*npoint*nsamples*channel
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))  # batch*channel*npoint*nsamples
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.att_pooling_1(f_concat)  # Batch*channel*npoints*1

        f_xyz = self.mlp2(f_xyz)
        f_neighbours = self.gather_neighbour(f_pc_agg.squeeze(-1).permute((0, 2, 1)), neigh_idx)  # batch*npoint*nsamples*channel
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))  # batch*channel*npoint*nsamples
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.att_pooling_2(f_concat)
        return f_pc_agg

    def relative_pos_encoding(self, xyz, neigh_idx):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)  # batch*npoint*nsamples*3

        xyz_tile = xyz.unsqueeze(2).repeat(1, 1, neigh_idx.shape[-1], 1)  # batch*npoint*nsamples*3
        relative_xyz = xyz_tile - neighbor_xyz  # batch*npoint*nsamples*3
        relative_dis = torch.sqrt(torch.sum(torch.pow(relative_xyz, 2), dim=-1, keepdim=True))  # batch*npoint*nsamples*1
        relative_feature = torch.cat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], dim=-1)  # batch*npoint*nsamples*10
        return relative_feature

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):  # pc: batch*npoint*channel
        # gather the coordinates or features of neighboring points
        batch_size = pc.shape[0]
        num_points = pc.shape[1]
        d = pc.shape[2]
        index_input = neighbor_idx.reshape(batch_size, -1)
        features = torch.gather(pc, 1, index_input.unsqueeze(-1).repeat(1, 1, pc.shape[2]))
        features = features.reshape(batch_size, num_points, neighbor_idx.shape[-1], d)  # batch*npoint*nsamples*channel
        return features

class Att_pooling(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Conv2d(d_in, d_in, (1, 1), bias=False)
        self.mlp = Conv2d(d_in, d_out, kernel_size=(1,1), bn=True)

    def forward(self, feature_set):

        att_activation = self.fc(feature_set)
        att_scores = F.softmax(att_activation, dim=3)
        f_agg = feature_set * att_scores
        f_agg = torch.sum(f_agg, dim=3, keepdim=True)
        f_agg = self.mlp(f_agg)
        return f_agg

class _ConvBase(nn.Sequential):

    def __init__(self,in_size,out_size,kernel_size,stride,padding,activation,bn,init,conv=None,batch_norm=None,
                 bias=True,preact=False,name="",instance_norm=False,instance_norm_func=None):
        super().__init__()

        bias = bias and (not bn)
        conv_unit = conv(
            in_size,
            out_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        init(conv_unit.weight)
        if bias:
            nn.init.constant_(conv_unit.bias, 0)

        if bn:
            if not preact:
                bn_unit = batch_norm(out_size)
            else:
                bn_unit = batch_norm(in_size)
        if instance_norm:
            if not preact:
                in_unit = instance_norm_func(out_size, affine=False, track_running_stats=False)
            else:
                in_unit = instance_norm_func(in_size, affine=False, track_running_stats=False)

        if preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)

            if activation is not None:
                self.add_module(name + 'activation', activation)

            if not bn and instance_norm:
                self.add_module(name + 'in', in_unit)

        self.add_module(name + 'conv', conv_unit)

        if not preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)

            if activation is not None:
                self.add_module(name + 'activation', activation)

            if not bn and instance_norm:
                self.add_module(name + 'in', in_unit)


class _BNBase(nn.Sequential):

    def __init__(self, in_size, batch_norm=None, name=""):
        super().__init__()
        self.add_module(name + "bn", batch_norm(in_size, eps=1e-6, momentum=0.99))

        nn.init.constant_(self[0].weight, 1.0)
        nn.init.constant_(self[0].bias, 0)


class BatchNorm1d(_BNBase):

    def __init__(self, in_size: int, *, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm1d, name=name)


class BatchNorm2d(_BNBase):

    def __init__(self, in_size: int, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm2d, name=name)


class Conv1d(_ConvBase):

    def __init__(self,in_size: int,out_size: int,*,kernel_size: int = 1,stride: int = 1,padding: int = 0,
                 activation=nn.LeakyReLU(negative_slope=0.2, inplace=True),bn: bool = False,init=nn.init.kaiming_normal_,
                 bias: bool = True,preact: bool = False,name: str = "",instance_norm=False):
        super().__init__(in_size,out_size,kernel_size,stride,padding,activation,bn,init,conv=nn.Conv1d,
                         batch_norm=BatchNorm1d,bias=bias,preact=preact,name=name,instance_norm=instance_norm,
                         instance_norm_func=nn.InstanceNorm1d)


class Conv2d(_ConvBase):

    def __init__(self,in_size: int,out_size: int,*,kernel_size: Tuple[int, int] = (1, 1),stride: Tuple[int, int] = (1, 1),
                 padding: Tuple[int, int] = (0, 0),activation=nn.LeakyReLU(negative_slope=0.2, inplace=True),
                 bn: bool = False,init=nn.init.kaiming_normal_,bias: bool = True,preact: bool = False,
                 name: str = "",instance_norm=False):
        super().__init__(in_size,out_size,kernel_size,stride,padding,activation,bn,init,conv=nn.Conv2d,
                         batch_norm=BatchNorm2d,bias=bias,preact=preact,name=name,instance_norm=instance_norm,
                         instance_norm_func=nn.InstanceNorm2d)


def crop_pc(points,search_tree, pick_idx):
    # crop a fixed size point cloud for training
    center_point = points[pick_idx, :].reshape(1, -1)
    select_idx = search_tree.query(center_point, k=4096*11)[1][0]
    select_points = points[select_idx]

    # Get other data
    for i in range(4):
        neighbour_idx = DP.knn_search(batch_pc, batch_pc, cfg.k_n)
        sub_points = batch_pc[:, :batch_pc.shape[1] // cfg.sub_sampling_ratio[i], :]
        pool_i = neighbour_idx[:, :batch_pc.shape[1] // cfg.sub_sampling_ratio[i], :]
        up_i = DP.knn_search(sub_points, batch_pc, 1)
        input_points.append(batch_pc)
        input_neighbors.append(neighbour_idx)
        input_pools.append(pool_i)
        input_up_samples.append(up_i)
        batch_pc = sub_points

    return select_points, select_idx


if __name__=="__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    pc_path = "/home/ngc/SemSeg/Datasets/KITTI_short/dataset/sequences_0.06/00/velodyne/000000.npy"
    tree_path = "/home/ngc/SemSeg/Datasets/KITTI_short/dataset/sequences_0.06/00/KDTree/000000.pkl"
    with open(tree_path, 'rb') as f:
        search_tree = pickle.load(f)
    points = np.array(search_tree.data, copy=False)

    pick_idx = np.random.choice(len(points), 1)
    selected_pc, selected_idx = crop_pc(points, search_tree, pick_idx)

    for i in range(4):
        neigh


    d_in = 3
    cloud = 1000 * torch.randn(1, d_in, 2 ** 17).to(device)
    print(cloud.size())
    model = Network()
    model.to(device)

    pred, coords = model(cloud)
    print(pred.size())