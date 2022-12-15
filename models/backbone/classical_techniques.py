import open3d as o3d
import torch
import numpy as np
import time
class voxel_subsampling():
    def __init__(self, cfg):
        self.voxel_size1 = cfg.voxel_size1
        self.voxel_size2 = cfg.voxel_size2

    def __call__(self, pts):
        tic = time.time()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size1)
        print(time.time()-tic)

        a = np.array(pcd.voxel_down_sample(voxel_size = self.voxel_size2).points)
        print(time.time()-tic)
        return a

class config():
    def __init__(self):
        self.voxel_size1 = 0.1
        self.voxel_size2 = 0.1

if __name__=="__main__":
    a = torch.randn(2**17, 3)
    cfg = config()
    voxel_subsampling(cfg)(a.numpy())