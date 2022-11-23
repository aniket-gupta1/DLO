import time

import torch
from torch.utils.data import Dataset
import os
import numpy as np

class kitti(Dataset):
    def __init__(self, config, mode="training"):
        super(kitti, self).__init__()
        self.root = config.root
        self.mode = mode

        if self.mode=="training":
            self.sequences = ['{:02d}'.format(i) for i in range(11) if i!=config.validation_seq]
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
        return frame_points.reshape((-1,4))[:, 0:3]

    def _read_pose(self, file_path):
        pose_list = []
        with open(file_path) as file:
            while True:
                line = file.readline()
                if not line:
                    break
                T = np.fromstring(line, dtype=np.float64, sep=' ')
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

    def __getitem__(self, index):
        if self.mode=="training" or self.mode=="validation":
            seq, pc_path, pose = self.data_list[index]
        else:
            seq, pc_path = self.data_list[index]
            pose = None

        data = {}

        data['pointcloud'] = torch.from_numpy(self._pcread(pc_path))
        data['seq'] = seq
        data['pose'] = pose
        data['frame_num'] = int(pc_path[-10:-4])

        return data

    def __len__(self):
        return len(self.data_list)

