from lietorch import SE3
from scipy.spatial.transform import Rotation
import numpy as np

def mat_to_quat(mat):
    quat = Rotation.from_matrix(mat[:,:3,:3]).as_quat()
    t = mat[:,:3,3]
    pose = np.concatenate((t, quat), axis=-1)

    return pose

