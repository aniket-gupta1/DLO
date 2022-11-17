import torch
import open3d.core as o3c
import numpy as np

def knn_search(support_pts, query_pts, k):
    """KNN search.
    Args:
        support_pts: points of shape (B, N, d)
        query_pts: points of shape (B, M, d)
        k: Number of neighbours in knn search
    Returns:
        neighbor_idx: neighboring points data (index, distance)
    """

    # Torch.cdist outputs a distance vector of shape (B, N, M)
    dist, idx = torch.cdist(support_pts, query_pts).topk(k)

    return dist, idx

class RandLANet(torch.nn.Module):
    def __init__(self, config, device):
        super(RandLANet, self).__init__()
        self.device = device
        self.config = config
        self.decimation = 4
        self.num_neighbours = 4

        # self.fc0 = torch.nn.Linear(self.config.in_channels, self.config.dim_features)
        self.fc0 = torch.nn.Linear(3, 8)
        self.bn0 = torch.nn.BatchNorm2d(8, eps=1e-6, momentum=0.01)
        self.lrelu = torch.nn.LeakyReLU(0.2)

        # Define the Encoder
        # self.encoder = []
        # dim_feature = self.config.dim_features
        # for i in range(self.config.num_layers):
        #     self.encoder.append(LocalFeatureAggregation(dim_feature, self.config.dim_output[i],
        #                                                 self.config.num_neighbours))
        #     dim_feature = 2*self.config.dim_output[i]
        #
        # self.encoder = torch.nn.ModuleList(self.encoder)

        self.encoder = torch.nn.ModuleList([
            LocalFeatureAggregation(8, 16, self.num_neighbours),
            LocalFeatureAggregation(32, 64, self.num_neighbours),
            LocalFeatureAggregation(128, 128, self.num_neighbours),
            LocalFeatureAggregation(256, 256, self.num_neighbours)
        ])

        # self.mlp = SharedMLP(dim_feature, dim_feature, activation_fn=torch.nn.LeakyReLU(0.2))
        self.mlp = SharedMLP(512, 512, activation_fn=torch.nn.LeakyReLU(0.2))

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        Randomly sample points from the pointcloud
        :param feature: input features; torch.Tensor (B, d, N, 1)
        :param pool_idx: selected positions; torch.Tensor (B, N', max_num] N'<N are the selected positions after pooling.
        :return: Pooled features matrix; torch.Tensor (B, N', d)
        """
        feature = feature.squeeze(3)
        num_neighbours = pool_idx.size()[2]
        batch_size = feature.size()[0]
        d = feature.size()[1]

        pool_idx = torch.reshape(pool_idx, (batch_size, -1))
        pool_idx = pool_idx.unsqueeze(2).expand(batch_size, -1, d)

        feature = feature.transpose(1,2)
        pool_features = torch.gather(feature, 1, pool_idx)
        pool_features = pool_features.permute(0,3,1,2)

        return pool_features

    def forward(self, input):
        """
        Forward pass of the complete model
        :param input: torch.Tensor of shape (B,N,d_in)
        :return: torch.Tensor of shape
        """
        print(f"Input size: {input.size()}")
        N = input.size(1)
        coords = input[..., :3].clone().cpu()

        # Pass input pointcloud through the first layer
        x = self.fc0(input).transpose(-2,-1).unsqueeze(-1)
        x = self.lrelu(self.bn0(x))

        print(f" Feature vector size: {x.size()}")
        d = self.decimation
        decimation_ratio = 1

        permutation = torch.randperm(N)
        coords = coords[:, permutation]
        x = x[:, :, permutation]

        for lfa in self.encoder:
            # at iteration i, x.shape = (B, N//(d**i), d_in)
            x = lfa(coords[:, :N // decimation_ratio], x)
            decimation_ratio *= d
            x = x[:, :, :N // decimation_ratio]

        feat = self.mlp(x)

        # feat = input['features'].to(self.device)
        # coords_list = [arr.to(self.device) for arr in input['coords']]
        # neighbour_indices_list = [arr.to(self.device) for arr in input['neighbour_indices']]
        # subsample_indices_list = [arr.to(self.device) for arr in input['sub_idx']]
        #
        # feat = self.fc0(feat).transpose(-2, -1).unsqueeze(-1) # (B, dim_feature, N, 1)
        # feat = self.bn0(feat)
        # feat = self.lrelu(feat)
        #
        # # Pass through the encoder and get the pointcloud encoding
        # for i in range(self.config.num_layers):
        #     feat_encoder_i = self.encoder[i](coords_list[i], feat, neighbour_indices_list[i])
        #
        #     feat_sampled_i = self.random_sample(feat_encoder_i, subsample_indices_list[i])
        #
        #     feat = feat_sampled_i
        #
        # feat = self.mlp(feat)

        return feat

class SharedMLP(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, bn=True, activation_fn=None):
        super(SharedMLP, self).__init__()

        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                    padding=(kernel_size-1)//2)

        self.batch_norm = torch.nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.01) if bn else None

        self.activation = activation_fn

    def forward(self, input):
        """
        Forward pass
        :param input: torch.Tensor of shape (B, dim_in, N, K)
        :return: torch.Tensor of shape (B, dim_out, N, K)
        """

        input = self.conv(input)
        if self.batch_norm:
            input = self.batch_norm(input)
        if self.activation:
            input = self.activation(input)

        return input

class LocalSpatialEncoding(torch.nn.Module):
    """
    This module computes K neighbour feature encoding for each point. This encoding consists of absolute distance,
    relative distance and positions
    """
    def __init__(self, dim_in, dim_out, num_neighbours, encode_pos=False):
        super(LocalSpatialEncoding, self).__init__()
        self.num_neighbours = num_neighbours
        self.mlp = SharedMLP(dim_in, dim_out, activation_fn=torch.nn.LeakyReLU(0.2))
        self.encode_pos = encode_pos

    def gather_neighbour(self, coords, neighbour_indices):
        """
        Gather features based on the neighbour indices
        :param coords: torch.tensor of shape (B,N,d)
        :param neighbour_indices: torch.Tensor of shape (B,N,K)
        :return: torch.Tensor; gathered neighbours of shape (B, dim, N, K)
        """

        B,N,K = neighbour_indices.size()
        dim = coords.shape[2]

        extended_indices = neighbour_indices.unsqueeze(1).expand(B, dim, N, K)
        extended_coords = coords.transpose(-2, -1).unsqueeze(-1).expand(B,dim, N, K)
        neighbour_coords = torch.gather(extended_coords, 2, extended_indices)

        return neighbour_coords

    def forward(self, coords, features, neighbour_indices, relative_features=None):
        """
        Forward pass
        :param coords: coordinates of the point cloud; torch.Tensor (B, N, 3)
        :param features: features of the point cloud; torch.Tensor (B, d, N, 1)
        :param neighbour_indices: indices of k neighbours; torch.Tensor (B, N, K)
        :param relative_features: relative neighbour features calculated on first pass.
        :return: torch.Tensor of shape (B, 2*d, N, K)
        """

        # Find neighbouring points
        B, N, K = neighbour_indices.size()

        if self.encode_pos:
            neighbour_coords = self.gather_neighbour(coords, neighbour_indices)
            extended_coords = coords.transpose(-2, -1).unsqueeze(-1).expand(B,3, N, K)
            relative_pos = extended_coords - neighbour_coords
            relative_dist = torch.sqrt(torch.sum(torch.square(relative_pos), dim=1, keepdim=True))
            relative_features = torch.cat([relative_dist, relative_pos, extended_coords, neighbour_coords], dim=1)

        else:
            if relative_features is None:
                raise ValueError("LocSE module requires relative featues for second pass")

        print(relative_features.size())
        relative_features = self.mlp(relative_features)
        print(relative_features.size())

        neighbour_features = self.gather_neighbour(features.transpose(1,2).squeeze(3), neighbour_indices)
        print(neighbour_features.size())

        return torch.cat([neighbour_features, relative_features], dim=1), relative_features

class AttentivePooling(torch.nn.Module):
    """
    This module generates single encoding from k neighbour features using weighted average with attention scores.
    """
    def __init__(self, in_channels, out_channels):
        super(AttentivePooling, self).__init__()
        self.score_fn = torch.nn.Sequential(torch.nn.Linear(in_channels, out_channels),
                                            torch.nn.Softmax(dim=-2)) # -2 means along the 2nd last dimension

        self.mlp = SharedMLP(in_channels,out_channels, activation_fn=torch.nn.LeakyReLU(0.2))

    def forward(self, input):
        """
        Forward pass
        :param input: torch.Tensor of shape (B, dim_in, N, K)
        :return: torch.Tensor of shape (B, dim_out, N, 1)
        """
        scores = self.score_fn(input.permute(0,2,3,1)).permute(0,3,1,2)
        features = torch.sum(scores*input, dim=-1, keepdim=True)

        return self.mlp(features)

class LocalFeatureAggregation(torch.nn.Module):
    """
    Neighbour features returned from LocSE and pooled from Attentive Pooling are aggregated and processed in multiple
    layers in this module.
    """
    def __init__(self, d_in, d_out, num_neighbours):
        super(LocalFeatureAggregation, self).__init__()
        self.num_neighbours = num_neighbours

        self.mlp1 = SharedMLP(d_in, d_out, activation_fn=torch.nn.LeakyReLU(0.2))
        self.lse1 = LocalSpatialEncoding(10, d_out//2, num_neighbours, encode_pos=True)
        self.pool1 = AttentivePooling(d_out, d_out//2)

        self.lse2 = LocalSpatialEncoding(d_out//2, d_out//2, num_neighbours)
        self.pool2 = AttentivePooling(d_out, d_out)
        self.mlp2 = SharedMLP(d_out, 2*d_out)

        self.shortcut = SharedMLP(d_in, 2*d_out)
        self.lrelu = torch.nn.LeakyReLU()

    def forward(self, coords, features):
        """
        Forward pass
        :param coords: coordinates of the point cloud; torch.Tensor (B, N, 3)
        :param features: features of the point cloud; torch.Tensor (B, d, N, 1)
        :param neighbour_indices: Indices of neighbour
        :return: torch.Tensor of shape (B, 2*d_out, N, 1)
        """

        dist, neighbour_indices = knn_search(coords, coords, self.num_neighbours)
        print(f"indices: {neighbour_indices.size()}")

        print(f"Before linear layer: {features.size()}")

        x = self.mlp1(features)

        print(f"After linear layer: {x.size()}")
        x, neighbour_features = self.lse1(coords, x, neighbour_indices)
        print(f"After lse layer: {x.size()} {neighbour_features.size()}")

        x = self.pool1(x)

        x, _ = self.lse2(coords, x, neighbour_indices, relative_features = neighbour_features)
        x = self.pool2(x)

        return self.lrelu(self.mlp2(x) + self.shortcut(features))
