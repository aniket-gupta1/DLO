from lietorch import SE3
from scipy.spatial.transform import Rotation
import numpy as np
import torch

def mat_to_quat(mat):
    quat = Rotation.from_matrix(mat[:,:3,:3]).as_quat()
    t = mat[:,:3,3]
    pose = np.concatenate((t, quat), axis=-1)

    return pose

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