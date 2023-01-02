import open3d as o3d
import torch
import numpy as np
import time
# from models.backbone.backbone_kpconv.kpconv import *

class config():
    def __init__(self):
        self.num_layers= 2
        self.neighborhood_limits= [50, 50]
        self.aggregation_mode= sum
        self.first_subsampling_dl= 0.03  # Set smaller to have a higher resolution
        self.first_feats_dim= 512
        self.fixed_kernel_points = "center"
        self.in_feats_dim= 1
        self.in_points_dim= 3
        self.conv_radius= 2.75
        self.deform_radius= 5.0
        self.KP_extent= 2.0
        self.KP_influence= "linear"
        self.overlap_radius= 0.04
        self.use_batch_norm= True
        self.batch_norm_momentum= 0.02
        self.modulated= False
        self.num_kernel_points= 15
        self.architecture= ['simple',
                       'resnetb',
                       'resnetb',
                       'resnetb_strided',
                       'resnetb',
                       'resnetb', ]


def softmax_correlation(feat0: torch.Tensor, feat1: torch.Tensor, pts0: torch.Tensor, pts1: torch.Tensor):
    b, n, d = feat0.shape
    _, m, _ = feat1.shape

    print(b, n, d, m)

    if n >= m:
        correlation = torch.matmul(feat0, feat1.permute(0, 2, 1)) / (d ** 0.5)  # [B, N, M]
        prob = torch.nn.functional.softmax(correlation, dim=-1)  # [B, N, M]

        val, ind = torch.max(prob, dim=1)  # [B,M]

        # pts0 -> [B, N, 3] ; pts1 -> [B, M, 3]
        # val -> [B,M] ; ind -> [B,M]
        src_pts = torch.gather(pts0, 1, ind.unsqueeze(-1).expand(-1, -1, 3)) # [B, N, 3] -> [B, M, 3]
        print("src_pts shape: ", src_pts.shape)
        tgt_pts = pts1
        print("tgt_pts.shape", tgt_pts.shape)

        # init_grid = torch.arange(m).float().cuda().requires_grad_()  # [B, N]
        raise ValueError

    else:
        correlation = torch.matmul(feat1, feat0.permute(0, 2, 1)) / (d ** 0.5)  # [B, M, N]
        print(correlation.shape)
        prob = torch.nn.functional.softmax(correlation, dim=-1)  # [B, M, N]
        print(prob.shape)
        init_grid = torch.arange(n).float().cuda().requires_grad_()  # [B, N]

    correspondence = torch.matmul(prob, init_grid)  # [B, N]

    return correspondence

if __name__=="__main__":
    a = torch.randn(2, 5, 3)
    b = torch.randn(2, 4, 3)
    a_f = torch.randn(2,5,6)
    b_f = torch.randn(2,4,6)
    print("src: ", a)
    print("tgt: ", b)
    print("src_feat:", a_f)
    print("tgt_feat:", b_f)
    softmax_correlation(a_f, b_f, a, b)

    # cfg = config()
    #
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #
    # d_in = 4
    # cloud = 1000 * torch.randn(1, 2 ** 17, d_in).to(device)
    # print(cloud.size())
    # model = KPFEncoder(cfg, 256)
    # model.to(device)
    #
    # out = model(cloud)
    # print(out.size())


    # Make up some points in the valid range

    # pts = np.fromfile("/home/ngc/SemSeg/Datasets/SemanticKITTI/dataset/sequences/00/velodyne/000000.bin",
    #                   dtype=np.float32).reshape((-1, 4))[:, 0:3]
    # print(pts.shape)
    #
    #
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pts)
    # # o3d.visualization.draw_geometries([pcd])
    #
    # v = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.05)
    #
    # # o3d.visualization.draw_geometries([v])
    # tic = time.time()
    # # d = {}
    # for i in range(pts.shape[0]):
    #
    #     v.get_voxel(pts[i, :])
    #     # d[tuple(pts[i,:])] = v.get_voxel(pts[i, :])
    #     #print("Voxel index: ", v.get_voxel(pts[0,:]))
    #
    # # print(len(d.keys()))
    # # len(v.get_voxels())
    # print(time.time()-tic)
