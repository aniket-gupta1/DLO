import time
import torch
import torch.nn as nn
import numpy as np
from torch_points_kernels import knn
import cpp_wrappers.nearest_neighbors.lib.python.nearest_neighbors as nene
# def knn_search(support_pts, query_pts, k):
#     """KNN search.
#     Args:
#         support_pts: points of shape (B, N, d)
#         query_pts: points of shape (B, M, d)
#         k: Number of neighbours in knn search
#     Returns:
#         neighbor_idx: neighboring points data (index, distance)
#     """
#     print(support_pts.size())
#     print(query_pts.size())
#     print(k)
#     # Torch.cdist outputs a distance vector of shape (B, N, M)
#     dist, idx = torch.cdist(support_pts, query_pts).topk(k)
#
#     return idx, dist

def knn_search(support_pts, query_pts, k):
    """
    :param support_pts: points you have, B*N1*3
    :param query_pts: points you want to know the neighbour index, B*N2*3
    :param k: Number of neighbours in knn search
    :return: neighbor_idx: neighboring points indexes, B*N2*k
    """

    neighbor_idx = nene.knn_batch(support_pts, query_pts, k, omp=True)
    return neighbor_idx.astype(np.int32)

class SharedMLP(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding_mode='zeros',
                     bn=False, activation_fn=None):
        super(SharedMLP, self).__init__()

        self.conv = nn.Conv2d( in_channels, out_channels, kernel_size, stride=stride,
                             padding_mode=padding_mode)
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99) if bn else None
        self.activation_fn = activation_fn

    def forward(self, input):
        """
        Forward pass
        :param input: torch.Tensor of shape (B, dim_in, N, K)
        :return: torch.Tensor of shape (B, dim_out, N, K)
        """

        x = self.conv(input)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x

class LocalSpatialEncoding(nn.Module):
    def __init__(self, d, num_neighbors, device):
        super(LocalSpatialEncoding, self).__init__()

        self.num_neighbors = num_neighbors
        self.mlp = SharedMLP(10, d, bn=True, activation_fn=nn.ReLU())

        self.device = device

    def forward(self, coords, features, knn_output):
        """
        Forward pass
        :param coords: coordinates of the point cloud; torch.Tensor (B, N, 3)
        :param features: features of the point cloud; torch.Tensor (B, d, N, 1)
        :param knn_output: k nearest neighbours and their distances
        :return: torch.Tensor of shape (B, 2*d, N, K)
        """
        # finding neighboring points
        idx, dist = knn_output
        B, N, K = idx.size()
        # idx(B, N, K), coords(B, N, 3)
        # neighbors[b, i, n, k] = coords[b, idx[b, n, k], i] = extended_coords[b, i, extended_idx[b, i, n, k], k]
        extended_idx = idx.unsqueeze(1).expand(B, 3, N, K)
        extended_coords = coords.transpose(-2,-1).unsqueeze(-1).expand(B, 3, N, K)
        neighbors = torch.gather(extended_coords, 2, extended_idx) # shape (B, 3, N, K)
        # if USE_CUDA:
        #     neighbors = neighbors.cuda()
        # print("extended_idx: ", extended_idx.size())
        # print("extended_coords: ", extended_coords.size())
        # print("neighbors: ", neighbors.size())

        # relative point position encoding
        concat = torch.cat((
            extended_coords,
            neighbors,
            extended_coords - neighbors,
            dist.unsqueeze(-3)
        ), dim=-3).to(self.device)

        #print(concat.size())
        return torch.cat((
            self.mlp(concat),
            features.expand(B, -1, N, K)
        ), dim=-3)

class AttentivePooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentivePooling, self).__init__()

        self.score_fn = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=False),
            nn.Softmax(dim=-2)
        )
        self.mlp = SharedMLP(in_channels, out_channels, bn=True, activation_fn=nn.ReLU())

    def forward(self, x):
        """
        Forward pass
        :param input: torch.Tensor of shape (B, dim_in, N, K)
        :return: torch.Tensor of shape (B, dim_out, N, 1)
        """

        # computing attention scores
        scores = self.score_fn(x.permute(0,2,3,1)).permute(0,3,1,2)

        # sum over the neighbors
        features = torch.sum(scores * x, dim=-1, keepdim=True) # shape (B, d_in, N, 1)

        return self.mlp(features)

class LocalFeatureAggregation(nn.Module):
    def __init__(self, d_in, d_out, num_neighbors, device):
        super(LocalFeatureAggregation, self).__init__()

        self.num_neighbors = num_neighbors

        self.mlp1 = SharedMLP(d_in, d_out//2, activation_fn=nn.LeakyReLU(0.2))
        self.mlp2 = SharedMLP(d_out, 2*d_out)
        self.shortcut = SharedMLP(d_in, 2*d_out, bn=True)

        self.lse1 = LocalSpatialEncoding(d_out//2, num_neighbors, device)
        self.lse2 = LocalSpatialEncoding(d_out//2, num_neighbors, device)

        self.pool1 = AttentivePooling(d_out, d_out//2)
        self.pool2 = AttentivePooling(d_out, d_out)

        self.lrelu = nn.LeakyReLU()

    def forward(self, coords, features):
        """
        Forward pass
        :param coords: coordinates of the point cloud; torch.Tensor (B, N, 3)
        :param features: features of the point cloud; torch.Tensor (B, d, N, 1)
        :return: torch.Tensor of shape (B, 2*d_out, N, 1)
        """
        # knn_output = knn(coords.cpu().contiguous(), coords.cpu().contiguous(), self.num_neighbors)
        # tic = time.time()
        knn_output = knn(coords.cpu().contiguous(), coords.cpu().contiguous(), self.num_neighbors)
        # print("Time knn: ", time.time()-tic)
        # tic = time.time()
        # temp = knn_search(coords, coords, self.num_neighbors)
        # print("Time: ", time.time()-tic)


        # print("coords (lfa): ", coords.size())
        # print("features (lfa): ", features.size())
        x = self.mlp1(features)

        x = self.lse1(coords, x, knn_output)
        x = self.pool1(x)

        x = self.lse2(coords, x, knn_output)
        x = self.pool2(x)

        return self.lrelu(self.mlp2(x) + self.shortcut(features))

class RandLANet(nn.Module):
    def __init__(self, d_in, num_neighbors=4, decimation=4, num_features=500, device=torch.device('cpu')):
        super(RandLANet, self).__init__()
        self.num_neighbors = num_neighbors
        self.decimation = decimation
        self.num_features = num_features

        self.fc_start = nn.Linear(d_in, 8)
        self.bn_start = nn.Sequential(
            nn.BatchNorm2d(8, eps=1e-6, momentum=0.99),
            nn.LeakyReLU(0.2)
        )

        # encoding layers
        self.encoder = nn.ModuleList([
            LocalFeatureAggregation(8, 16, num_neighbors, device),
            LocalFeatureAggregation(32, 64, num_neighbors, device),
            LocalFeatureAggregation(128, 128, num_neighbors, device),
            LocalFeatureAggregation(256, 256, num_neighbors, device)
        ])

        self.mlp = SharedMLP(512, 512, activation_fn=nn.ReLU())

        self.device = device

    def forward(self, input):
        """
        Forward pass of the complete model
        :param input: torch.Tensor of shape (B,N,d_in)
        :return: torch.Tensor of shape (B, 512, N//256, 1)
        """

        N = input.size(1)
        d = self.decimation


        coords = input[...,:3].clone().cpu()
        x = self.fc_start(input).transpose(-2,-1).unsqueeze(-1)
        x = self.bn_start(x) # shape (B, d, N, 1)

        decimation_ratio = 1

        permutation = torch.randperm(N)
        coords = coords[:,permutation]
        x = x[:,:,permutation]


        for i, lfa in enumerate(self.encoder):
            # at iteration i, x.shape = (B, N//(d**i), d_in)
            # print("x before lfa: ", x.size())
            x = lfa(coords[:,:N//decimation_ratio], x)
            # print("x after lfa: ", x.size())

            decimation_ratio *= d
            if i==len(self.encoder)-1:
                x = x[:,:,:self.num_features]
            else:
                x = x[:,:,:N//decimation_ratio]
            # print("x after decimation: ", x.size())

        sampled_coords = coords[:,:self.num_features].to(self.device)
        # print(sampled_coords.size())

        y = self.mlp(x).transpose(1,2).squeeze(3)

        return y, sampled_coords


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    d_in = 4
    cloud = 1000*torch.randn(1, 2**17, d_in).to(device)
    print(cloud.size())
    model = RandLANet(d_in, 16, 4, 500, device)
    model.to(device)

    pred, coords = model(cloud)
    print(pred.size())
