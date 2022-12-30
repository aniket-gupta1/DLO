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

        self.frames = {}
        self.poses = {}
        self.data_list = []
        for seq in self.sequences:
            velo_path = os.path.join(self.root, 'sequences', seq, 'velodyne')
            self.frames[seq] = np.sort([vf[:-4] for vf in os.listdir(velo_path) if vf.endswith('.bin')])

            if self.mode=="training" or self.mode=="validation":
                pose_path = os.path.join(self.root, 'poses') + f"/{seq}.txt"
                self.poses[seq] = self._read_pose(pose_path)

            for i, vf in enumerate(sorted(os.listdir(velo_path))):
                if vf.endswith('.bin'):
                    vf_path = os.path.join(self.root, 'sequences', seq, 'velodyne', vf)
                    data = [seq, vf_path]

                    if self.mode=="training" or self.mode=="validation":
                        pose = self.poses[seq][i]
                        data.append(pose)

                    self.data_list.append(data)

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

    def _read_pose_mat(self, file_path):
        pose_list = []
        with open(file_path) as file:
            while True:
                line = file.readline()

                if not line:
                    break

                line = np.fromstring(line, dtype=np.float64, sep=' ')
                T = np.reshape(line, (3,4))
                pose_list.append(T)

        return pose_list

    def _inbetween_pose(self, pose):
        pose = np.reshape(pose, (4,4))
        inbetween_pose = pose @ np.linalg.inv(self.initial_pose)

        inbetween_pose = np.reshape(inbetween_pose, (16))

        self.initial_pose = pose

        return inbetween_pose

    def _downsample(self, pts, vs1=0.1, vs2=0.1):
        """
        Downsample using voxel grid. Although the downsampled point is the average of all the points in the voxel.
        :param pts:
        :param vs1:
        :param vs2:
        :return:
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        # pcd = pcd.voxel_down_sample(voxel_size= vs1)

        return np.array(pcd.voxel_down_sample(voxel_size = vs2).points).astype(np.float32)


    def __getitem__(self, index):
        if self.mode=="training" or self.mode=="validation":
            seq, pc_path, pose = self.data_list[index]
        else:
            seq, pc_path = self.data_list[index]
            pose = None

        data = {}

        if self.downsample:
            data['pointcloud'] = torch.from_numpy(self._downsample(self._pcread(pc_path)))
        else:
            data['pointcloud'] = torch.from_numpy(self._pcread(pc_path))
        data['seq'] = seq
        data['frame_num'] = int(pc_path[-10:-4])
        data['true_pose'] = np.reshape(pose, (4,4))


        if self.form_transformation and self.inbetween_poses:
            data['pose'] = np.reshape(self._inbetween_pose(pose), (4,4))
        elif self.form_transformation:
            data['pose'] = np.reshape(pose, (3,4))
        elif self.inbetween_poses:
            data['pose'] = self._inbetween_pose(pose)[:12]
        else:
            data['pose'] = pose

        return data

    def __len__(self):
        return len(self.data_list)-1
