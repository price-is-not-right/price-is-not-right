"""Robot / gripper helpers shared by Panda and Kinova3 (Robotiq85) pipelines."""
from __future__ import annotations

import numpy as np

# OSC_POSITION eval / --ee demos. Joint-velocity defaults (default_kinova3) are not used.
SUPPORTED_ROBOTS = ("Panda", "Kinova3")

# EE home used by reset_gripper between place ops. Kinova value is a starting
# guess in the same table workspace — retune after a GT smoke test if needed.
RESET_GRIPPER_POS = {
    "Panda": np.array([-0.080193391, -0.03391656, 0.95828137], dtype=np.float64),
    "Kinova3": np.array([-0.080193391, -0.03391656, 0.95828137], dtype=np.float64),
}

# (left_body, right_body, open_aperture_threshold)
_FINGER_CANDIDATES = (
    ("gripper0_leftfinger", "gripper0_rightfinger", 0.055),       # PandaGripper
    ("gripper0_left_inner_finger", "gripper0_right_inner_finger", 0.13),  # Robotiq85
)


def resolve_finger_bodies(sim):
    """Return (left_name, right_name, open_thresh) for the mounted gripper."""
    for left, right, thresh in _FINGER_CANDIDATES:
        try:
            sim.model.body_name2id(left)
            sim.model.body_name2id(right)
            return left, right, thresh
        except ValueError:
            continue
    raise ValueError(
        "Could not resolve gripper finger bodies. Tried Panda "
        "(gripper0_leftfinger) and Robotiq85 (gripper0_left_inner_finger)."
    )


def gripper_finger_positions(sim):
    left, right, _ = resolve_finger_bodies(sim)
    left_pos = np.asarray(sim.data.body_xpos[sim.model.body_name2id(left)])
    right_pos = np.asarray(sim.data.body_xpos[sim.model.body_name2id(right)])
    return left_pos, right_pos


def gripper_aperture(sim):
    left_pos, right_pos = gripper_finger_positions(sim)
    return float(np.linalg.norm(left_pos - right_pos))


def gripper_is_open(sim, return_distance=False):
    _, _, thresh = resolve_finger_bodies(sim)
    aperture = gripper_aperture(sim)
    if return_distance:
        return aperture
    return bool(aperture > thresh)


def reset_gripper_target(robot: str) -> np.ndarray:
    if robot not in RESET_GRIPPER_POS:
        raise ValueError(f"Unknown robot {robot!r}; expected one of {SUPPORTED_ROBOTS}")
    return RESET_GRIPPER_POS[robot].copy()
