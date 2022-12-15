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

if __name__=="__main__":
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

    pts = np.fromfile("/home/ngc/SemSeg/Datasets/SemanticKITTI/dataset/sequences/00/velodyne/000000.bin",
                      dtype=np.float32).reshape((-1, 4))[:, 0:3]
    print(pts.shape)


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    # o3d.visualization.draw_geometries([pcd])

    v = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.05)

    # o3d.visualization.draw_geometries([v])
    tic = time.time()
    # d = {}
    for i in range(pts.shape[0]):

        v.get_voxel(pts[i, :])
        # d[tuple(pts[i,:])] = v.get_voxel(pts[i, :])
        #print("Voxel index: ", v.get_voxel(pts[0,:]))

    # print(len(d.keys()))
    # len(v.get_voxels())
    print(time.time()-tic)
