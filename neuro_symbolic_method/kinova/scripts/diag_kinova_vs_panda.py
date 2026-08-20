"""Compare Panda vs Kinova3 expert ReachPick geometry and OSC tracking."""
import os
import sys
os.environ.setdefault("MUJOCO_GL", "osmesa")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import robosuite as suite
from robosuite.utils.detector import HanoiDetector
from robot_utils import gripper_aperture, gripper_finger_positions


def make_env(robot):
    cfg = suite.load_controller_config(default_controller="OSC_POSITION")
    env = suite.make(
        "Hanoi",
        robots=robot,
        controller_configs=cfg,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        control_freq=20,
    )
    env.reset()
    return env


def sites_and_bodies(env, robot):
    sim = env.sim
    eef = np.asarray(sim.data.body_xpos[sim.model.body_name2id("gripper0_eef")])
    try:
        grip = np.asarray(sim.data.site_xpos[sim.model.site_name2id("gripper0_grip_site")])
    except ValueError:
        grip = None
    left, right = gripper_finger_positions(sim)
    mid = 0.5 * (left + right)
    cube = np.asarray(sim.data.body_xpos[env.obj_body_id["cube1"]])
    print(f"\n=== {robot} geometry at reset ===")
    print(f"  eef           {np.round(eef, 4)}")
    print(f"  grip_site     {None if grip is None else np.round(grip, 4)}")
    print(f"  finger_mid    {np.round(mid, 4)}")
    print(f"  cube1         {np.round(cube, 4)}")
    print(f"  eef-mid       {np.round(eef - mid, 4)}  |xy|={np.linalg.norm((eef-mid)[:2])*1000:.1f}mm")
    print(f"  eef-cube xy   {np.linalg.norm((eef-cube)[:2])*1000:.1f}mm  z={eef[2]-cube[2]:.3f}")
    print(f"  mid-cube xy   {np.linalg.norm((mid-cube)[:2])*1000:.1f}mm  z={mid[2]-cube[2]:.3f}")
    print(f"  aperture      {gripper_aperture(sim):.4f}")
    return env


def osc_step_response(env, robot):
    sim = env.sim
    p0 = np.asarray(sim.data.body_xpos[sim.model.body_name2id("gripper0_eef")]).copy()
    # command +X at full scale for 10 steps
    moved = []
    for i in range(10):
        env.step(np.array([1.0, 0.0, 0.0, -1.0]))
        p = np.asarray(sim.data.body_xpos[sim.model.body_name2id("gripper0_eef")])
        moved.append(p - p0)
    d = np.array(moved)
    print(f"\n=== {robot} OSC +X action=1.0 for 10 steps ===")
    print(f"  step1 delta {np.round(d[0], 4)}  |xy|={np.linalg.norm(d[0][:2])*1000:.1f}mm")
    print(f"  step10 delta {np.round(d[-1], 4)} |xy|={np.linalg.norm(d[-1][:2])*1000:.1f}mm")


def expert_reach_pick(env, robot, n_max=400):
    det = HanoiDetector(env)
    sim = env.sim
    gid = sim.model.body_name2id("gripper0_eef")
    cid = env.obj_body_id["cube1"]

    def cap(eps, max_val=0.12, min_val=0.01):
        n = np.linalg.norm(eps)
        if n > max_val:
            eps = eps / n * max_val
        if n < min_val and n > 0:
            eps = eps / n * min_val
        return eps

    # lift
    for _ in range(80):
        z = sim.data.body_xpos[gid][2]
        if z >= 1.1:
            break
        dz = cap(np.array([1.1 - z]))
        env.step(np.array([0.0, 0.0, 5.0 * dz[0], -1.0]))

    hist = []
    for t in range(n_max):
        g = np.asarray(sim.data.body_xpos[gid])
        c = np.asarray(sim.data.body_xpos[cid])
        xy = c[:2] - g[:2]
        dist = np.linalg.norm(xy)
        over = dist < 0.01
        hist.append(dist)
        if over:
            print(f"\n=== {robot} expert ReachPick over(cube1) at t={t}  xy={dist*1000:.2f}mm")
            left, right = gripper_finger_positions(sim)
            mid = 0.5 * (left + right)
            print(f"  eef-cube xy {dist*1000:.2f}mm  finger_mid-cube xy {np.linalg.norm(mid[:2]-c[:2])*1000:.2f}mm")
            print(f"  last 5 dist_mm {np.round(np.array(hist[-5:])*1000, 2)}")
            return dist
        cmd = cap(xy)
        env.step(np.array([5.0 * cmd[0], 5.0 * cmd[1], 0.0, -1.0]))
    print(f"\n=== {robot} FAILED to get over cube1 in {n_max} steps  last={hist[-1]*1000:.2f}mm")
    return hist[-1]


def expert_pick(env, robot, n_max=400):
    det = HanoiDetector(env)
    sim = env.sim
    gid = sim.model.body_name2id("gripper0_eef")
    cid = env.obj_body_id["cube1"]

    def cap(eps, max_val=0.12, min_val=0.01):
        n = np.linalg.norm(eps)
        if n > max_val:
            eps = eps / n * max_val
        if 0 < n < min_val:
            eps = eps / n * min_val
        return eps

    expert_reach_pick(env, robot, n_max=n_max)

    # open
    for t in range(80):
        st = det.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        if st["open_gripper(gripper)"]:
            print(f"  open_gripper at t={t} aperture={gripper_aperture(sim):.4f}")
            break
        env.step(np.array([0.0, 0.0, 0.0, -1.0]))
    else:
        print(f"  FAILED open_gripper aperture={gripper_aperture(sim):.4f}")
        return

    # descend
    for t in range(n_max):
        st = det.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        g = np.asarray(sim.data.body_xpos[gid])
        c = np.asarray(sim.data.body_xpos[cid])
        if st["at_grab_level(gripper,cube1)"]:
            print(f"  at_grab_level t={t} dz={abs(g[2]-c[2])*1000:.2f}mm xy={np.linalg.norm(g[:2]-c[:2])*1000:.2f}mm")
            break
        dz = cap(np.array([c[2] - g[2]]))
        env.step(np.array([0.0, 0.0, 5.0 * dz[0], -1.0]))
    else:
        print(f"  FAILED at_grab_level dz={abs(g[2]-c[2])*1000:.2f}mm")
        return

    # close
    for t in range(40):
        st = det.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        if st["grasped(cube1)"]:
            print(f"  grasped t={t} aperture={gripper_aperture(sim):.4f}")
            break
        env.step(np.array([0.0, 0.0, 0.0, 1.0]))
    else:
        print(f"  FAILED grasped aperture={gripper_aperture(sim):.4f} "
              f"open={st['open_gripper(gripper)']}")
        return

    # lift
    for t in range(80):
        st = det.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        if st["picked_up(cube1)"]:
            print(f"  picked_up t={t} cube_z={sim.data.body_xpos[cid][2]:.3f}")
            return
        env.step(np.array([0.0, 0.0, 0.4, 0.0]))
    print(f"  FAILED picked_up cube_z={sim.data.body_xpos[cid][2]:.3f} grasped={st['grasped(cube1)']}")


def main():
    for robot in ("Panda", "Kinova3"):
        env = make_env(robot)
        sites_and_bodies(env, robot)
        osc_step_response(env, robot)
        env.reset()
        print(f"\n=== {robot} expert pick cube1 ===")
        expert_pick(env, robot)
        env.close()


if __name__ == "__main__":
    main()
