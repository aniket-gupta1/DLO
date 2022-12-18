import time
import torch
from torch.utils.data import Dataset
import os
import open3d as o3d
import numpy as np

class kitti(Dataset):
    def __init__(self, config, mode="training", inbetween_poses=False, form_transformation=False):
        super(kitti, self).__init__()
        self.root = config.root
        self.downsample = config.downsample
        self.mode = mode
        self.form_transformation = form_transformation
        self.inbetween_poses = inbetween_poses
        self.initial_pose = np.eye(4)

        if self.mode=="training":
            self.sequences = ['{:02d}'.format(i) for i in range(1) if i!=config.validation_seq]
        elif self.mode=="validation":
            self.sequences = [config.validation_seq]
        elif self.mode=="test":
            self.sequences = ['{:02d}'.format(i) for i in range(11, 22)]
        else:
            raise ValueError(f"Unknown mode{self.mode} (Correct modes: training, test, validation)")

        self.poses_wrt_world = {}
        self.poses_t2wt1 = {}
        self.data_list = []
        for seq in self.sequences:
            # 1: Get inbetween pose
            # 1.1: Read all the poses from the file
            if self.mode=="training" or self.mode=="validation":
                pose_path = os.path.join(self.root, 'poses') + f"/{seq}.txt"
                self.poses_wrt_world[seq] = self._read_pose(pose_path)

            # 1.2: Update the relative poses
            self.get_relative_pose()

            # 2. Get the pc pairs
            # 2.1: Read all the pc's
            velo_path = os.path.join(self.root, 'sequences', seq, 'velodyne')
            for i, vf in enumerate(sorted(os.listdir(velo_path))):
                if i == 0 and vf.endswith('.bin'):
                    vf_path1 = os.path.join(self.root, 'sequences', seq, 'velodyne', vf)
                elif vf.endswith('.bin'):
                    vf_path2 = os.path.join(self.root, 'sequences', seq, 'velodyne', vf)
                    data = [vf_path1, vf_path2]
                    vf_path1 = vf_path2

                    if self.mode=="training" or self.mode=="validation":
                        pose = self.poses_t2wt1[seq][i-1]
                        data.append(pose)

                    self.data_list.append(data)
    def get_relative_pose(self):
        initial_pose = np.eye(4)

        for seq in self.sequences:
            self.poses_t2wt1[seq] = []
            for i, pose in enumerate(self.poses_wrt_world[seq]):
                if i==0:
                    continue
                else:
                    pose_mat = np.reshape(pose, (4,4))
                    self.poses_t2wt1[seq].append(pose_mat @ np.linalg.inv(initial_pose))
                    initial_pose = pose_mat

    def _pcread(self, path):
        frame_points = np.fromfile(path, dtype=np.float32)
        return frame_points.reshape((-1,4))[:100000, 0:3]

    def _read_pose(self, file_path):
        pose_list = []
        with open(file_path) as file:
            while True:
                line = file.readline()
                if not line:
                    break
                T = np.fromstring(line, dtype=np.float64, sep=' ')

                if self.inbetween_poses or self.form_transformation:
                    T = np.append(T, [0,0,0,1])

                pose_list.append(T)
        return pose_list

    def _downsample(self, pts, vs1=0.1, vs2=0.1):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        # pcd = pcd.voxel_down_sample(voxel_size= vs1)

        return np.array(pcd.voxel_down_sample(voxel_size = vs2).points).astype(np.float32)

    def __getitem__(self, index):
        if self.mode=="training" or self.mode=="validation":
            pc1, pc2, pose = self.data_list[index]
        else:
            pc1, pc2 = self.data_list[index]
            pose = None

        data = {}

        if self.downsample:
            data['pc1'] = torch.from_numpy(self._downsample(self._pcread(pc1)))
            data['pc2'] = torch.from_numpy(self._downsample(self._pcread(pc2)))
        else:
            data['pc1'] = torch.from_numpy(self._pcread(pc1))
            data['pc2'] = torch.from_numpy(self._pcread(pc2))

        data['pose'] = pose
        return data

    def __len__(self):
        return len(self.data_list)
