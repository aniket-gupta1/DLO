import torch
from torch.utils.data import DataLoader
from datasets.kitti import kitti
from torch.utils.tensorboard import SummaryWriter
import time
from models.model import DLO_net_single
from lietorch import SE3, SO3
from config.config import Config
from utils.utils import *
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import os
import open3d as o3d


def check_correctness(prev, curr):
    print(curr['pose'])
    T = curr['pose'].cpu().numpy().reshape(4,4).astype(np.float32)
    # T = curr['pose'].cpu().numpy().squeeze().astype(np.float32)
    pc1 = prev['pointcloud'].cpu().numpy().squeeze().astype(np.float32)
    pc2 = curr['pointcloud'].cpu().numpy().squeeze().astype(np.float32)

    print(type(pc1))
    print(pc1.shape)
    print(pc2.shape)
    print(T[:3,3])
    # pts =  T[:3,:3] @ pc2.T + np.expand_dims(T[:3, 3], -1)
    pts =  T[:3,:3] @ pc2.T + np.array([[T[2,3]], [-T[0,3]], [T[1,3]]])
    print(type(pts))
    print(pts.shape)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.T)

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pc1)

    o3d.visualization.draw_geometries([pcd])

    return False

def train(cfg, device, writer):
    dataset = kitti(cfg, mode = "training", inbetween_poses = cfg.inbetween_poses,
                    form_transformation = cfg.form_transformation)
    dataloader = DataLoader(dataset, batch_size=1)

    model = DLO_net_single(cfg, device).to(device)

    T_ptw = torch.eye(4).double()
    T_ctw = torch.eye(4).double()
    error = []
    error2 = []
    for index, data in enumerate(dataloader):
        # T_ctw = data['true_pose']
        #
        # T_ctp = torch.matmul(T_ctw, torch.linalg.inv(T_ptw))
        # e = torch.sum(T_ctp - data['pose'])
        # error.append(e)
        #
        # T_gt_ctw = torch.matmul(T_ctp, T_ptw)
        # e2 = torch.sum(T_gt_ctw - data['true_pose'])
        # error2.append(e2)
        #
        # T_ptw = T_ctw
        T_ctw = torch.matmul(data['pose'], T_ctw)

        error.append(torch.sum(T_ctw - data['true_pose']))
        error2.append(index)

        if index>5000:
            break

    plt.plot(error)
    plt.plot(error2)
    plt.show()

if __name__=="__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter()

    cfg = Config(512)
    train(cfg, device, writer)
    # eval(cfg, device, writer, "/home/ngc/epoch_114.pth")
    writer.close()
