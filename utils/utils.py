from lietorch import SE3
from scipy.spatial.transform import Rotation
import numpy as np
import torch
import matplotlib.pyplot as plt

def pad_sequence(sequences, require_padding_mask=False, require_lens=False,
                 batch_first=False):
    """List of sequences to padded sequences

    Args:
        sequences: List of sequences (N, D)
        require_padding_mask:

    Returns:
        (padded_sequence, padding_mask), where
           padded sequence has shape (N_max, B, D)
           padding_mask will be none if require_padding_mask is False
    """
    padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=batch_first)
    padding_mask = None
    padding_lens = None

    if require_padding_mask:
        B = len(sequences)
        seq_lens = list(map(len, sequences))
        padding_mask = torch.zeros((B, padded.shape[1]), dtype=torch.bool, device=padded.device)
        for i, l in enumerate(seq_lens):
            padding_mask[i, l:] = True

    if require_lens:
        padding_lens = [seq.shape[0] for seq in sequences]

    return padded, padding_mask, padding_lens


def all_to_device(data, device):
    """Sends everything into a certain device """
    if isinstance(data, dict):
        for k in data:
            data[k] = all_to_device(data[k], device)
        return data
    elif isinstance(data, list):
        data = [all_to_device(d, device) for d in data]
        return data
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data

def mat_to_quat(mat):
    quat = Rotation.from_matrix(mat[:,:3,:3]).as_quat()
    t = mat[:,:3,3]
    pose = np.concatenate((t, quat), axis=-1)

    return pose


def plot_poses_realtime(poses, poses_gt, plot_gt=True):
    poses_gt = np.array(poses_gt)
    poses = np.array(poses)

    plt.cla()
    if plot_gt:
        plt.plot(poses_gt[:, 0], poses_gt[:, 1], label="Ground Truth")

    plt.plot(poses[:, 0], poses[:, 1], label="Estimated trajectory")
    # plt.xlim([-100,100])
    # plt.ylim([0,200])
    plt.legend()
    plt.title("Estimated Trajectory using VO")
    plt.xlabel("metres")
    plt.ylabel("metres")
    plt.grid()
    # plt.axis('equal')
    plt.pause(0.001)


def compute_rigid_transform(a: torch.Tensor, b: torch.Tensor, weights: torch.Tensor = None):
    """Compute rigid transforms between two point sets

    Args:
        a (torch.Tensor): ([*,] N, 3) points
        b (torch.Tensor): ([*,] N, 3) points
        weights (torch.Tensor): ([*, ] N)

    Returns:
        Transform T ([*,] 3, 4) to get from a to b, i.e. T*a = b
    """

    assert a.shape == b.shape
    assert a.shape[-1] == 3

    if weights is not None:
        assert a.shape[:-1] == weights.shape
        assert weights.min() >= 0 and weights.max() <= 1

        weights_normalized = weights[..., None] / \
                             torch.clamp_min(torch.sum(weights, dim=-1, keepdim=True)[..., None], 0.000001)
        centroid_a = torch.sum(a * weights_normalized, dim=-2)
        centroid_b = torch.sum(b * weights_normalized, dim=-2)
        a_centered = a - centroid_a[..., None, :]
        b_centered = b - centroid_b[..., None, :]
        cov = a_centered.transpose(-2, -1) @ (b_centered * weights_normalized)
    else:
        centroid_a = torch.mean(a, dim=-2)
        centroid_b = torch.mean(b, dim=-2)
        a_centered = a - centroid_a[..., None, :]
        b_centered = b - centroid_b[..., None, :]
        cov = a_centered.transpose(-2, -1) @ b_centered

    # Compute rotation using Kabsch algorithm. Will compute two copies with +/-V[:,:3]
    # and choose based on determinant to avoid flips
    u, s, v = torch.svd(cov, some=False, compute_uv=True)
    rot_mat_pos = v @ u.transpose(-1, -2)
    v_neg = v.clone()
    v_neg[..., 2] *= -1
    rot_mat_neg = v_neg @ u.transpose(-1, -2)
    rot_mat = torch.where(torch.det(rot_mat_pos)[..., None, None] > 0, rot_mat_pos, rot_mat_neg)

    # Compute translation (uncenter centroid)
    translation = -rot_mat @ centroid_a[..., :, None] + centroid_b[..., :, None]

    transform = torch.cat((rot_mat, translation), dim=-1)
    return transform

def loss_fn_se3(pred, gt):
    print(pred)
    print(gt)
    r, t = loss_fn2(pred, gt)
    # print(gt)
    gt = SE3(torch.Tensor(mat_to_quat(gt)).type(torch.float32).to(device))
    # print(gt.matrix())

    # print(pred)
    pred = SE3.exp(pred)
    # print(pred)
    # print(pred.matrix())

    error = (gt.inv() * pred).log()
    tr = error[:,0:3].norm(dim=-1)
    ro = error[:,3:6].norm(dim=-1)

    loss = tr.mean() + ro.mean()

    print(tr.mean())
    print(ro.mean())
    print(r)
    print(t)

    time.sleep(20000000)
    return loss