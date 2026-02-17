"""
utils for making datasets
"""

import numpy as np
from datetime import datetime

def to_datestring(unixtime: float, fmt: str = '%Y-%m-%d_%H:%M:%S') -> str:
    """Convert a Unix timestamp to a formatted date string."""
    return datetime.utcfromtimestamp(unixtime).strftime(fmt)


def cap(eps: np.ndarray, max_val: float = 0.12, min_val: float = 0.01) -> np.ndarray:
    """Cap a vector's magnitude between min_val and max_val."""
    norm = np.linalg.norm(eps)
    if norm > max_val:
        return eps / norm * max_val
    if norm < min_val:
        return eps / norm * min_val
    return eps


def to_osc_pose(action: np.ndarray, n_grippers=1) -> np.ndarray:
    """Convert a 4D action to a 7D OSC pose action by inserting zero rotation components."""
    # action: [dx, dy, dz, gripper]
    # output: [dx, dy, dz, d_roll, d_pitch, d_yaw, gripper]
    return np.insert(action, 3, [0.0, 0.0, 0.0])


def quaternion_to_euler(quat: np.ndarray) -> np.ndarray:
    """Convert a quaternion (x, y, z, w) to Euler angles (roll, pitch, yaw)."""
    x, y, z, w = quat
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float32)