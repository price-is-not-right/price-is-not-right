# main.py

import os
import time
import argparse
import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper
from dataset_making.utils import to_datestring
from dataset_making.record_demos import RecordDemos
from robosuite.utils.detector import KitchenDetector, NutAssemblyDetector
from dataset_making.panda_hanoi_detector import PandaHanoiDetector as HanoiDetector
import imageio


def make_env(args):
    """Create and return a wrapped robosuite environment."""
    cube_noise = getattr(args, 'cube_init_pos_noise_std', 0.01)
    ctrl_cfg = suite.load_controller_config(default_controller='OSC_POSE')
    env = suite.make(
        args.env,
        robots="Panda",
        controller_configs=ctrl_cfg,
        has_renderer=args.render,
        has_offscreen_renderer=True,
        horizon=1e8,
        use_camera_obs=True,
        use_object_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=256,
        camera_widths=256,
        random_block_placement=args.random_block_placement,
        random_block_selection=args.random_block_selection,
        cube_init_pos_noise_std=cube_noise
    )
    # Gym-compatible wrapper, proprio_obs toggles vision-based vs proprio inputs
    env = GymWrapper(env, proprio_obs=not args.vision)
    return env


def get_detector(env, env_name):
    """Instantiate the correct detector for the chosen environment."""
    if env_name == 'Hanoi':
        return HanoiDetector(env)
    if env_name == 'KitchenEnv':
        return KitchenDetector(env)
    if env_name == 'NutAssembly':
        return NutAssemblyDetector(env)
    raise ValueError(f"Unknown env {env_name}")


def record_episode_video(env, recorder, epi_idx, camera="agentview", fps=20, save_full_res_vid=False):
    if save_full_res_vid:
        frames = []
        wrist_frames = []

    # Hook into the recorder so that every time it records a step,
    # we grab a frame from both cameras
    original_env_record = getattr(recorder.env, "record_step", None)
    def hooked_env_record(obs, action):
        if save_full_res_vid:
            frames.append(env.sim.render(width=640, height=480, camera_name=camera))
            wrist_frames.append(env.sim.render(width=640, height=480, camera_name="robot0_eye_in_hand"))
        return recorder.record_step(obs, action)
    recorder.env.record_step = hooked_env_record

    # Reset and grab the initial frame from both cameras
    obs = recorder.reset()
    print(f"Obs: {obs}")
    print(f"Plan: {recorder.plan}")
    while recorder.plan is None or obs is None:
        obs = recorder.reset()
        print("Plan is None, resetting")
    
    if save_full_res_vid:
        frames.append(env.sim.render(width=640, height=480, camera_name=camera))
        wrist_frames.append(env.sim.render(width=640, height=480, camera_name="robot0_eye_in_hand"))

    # Run the whole trajectory (will call hooked_record_step under the hood)
    success = recorder.run_trajectory(obs)

    # Finally grab one more frame from both cameras
    if save_full_res_vid:
        frames.append(env.sim.render(width=640, height=480, camera_name=camera))
        wrist_frames.append(env.sim.render(width=640, height=480, camera_name="robot0_eye_in_hand"))


    if save_full_res_vid:
        # Write out both videos
        video_path = f"episode_{epi_idx:03d}_{camera}.mp4"
        wrist_video_path = f"episode_{epi_idx:03d}_robot0_eye_in_hand.mp4"
        
        # Flip frames vertically to fix upside-down issue
        flipped_frames = [np.flipud(frame) for frame in frames]
        flipped_wrist_frames = [np.flipud(frame) for frame in wrist_frames]
        imageio.mimwrite(video_path, flipped_frames, fps=fps, macro_block_size=None)
        imageio.mimwrite(wrist_video_path, flipped_wrist_frames, fps=fps, macro_block_size=None)
        
        print(f"Saved episode videos to {video_path} and {wrist_video_path}    (success={success}, frames={len(frames)})")

    # Restore original hook in case you want to reuse recorder
    if original_env_record:
        recorder.env.record_step = original_env_record
    else:
        del recorder.env.record_step

    return success


def main():
    parser = argparse.ArgumentParser(description="Record robot demos via symbolic planning and execution.")
    parser.add_argument('--env', type=str, choices=['Hanoi', 'KitchenEnv', 'NutAssembly', 'CubeSorting'], default='Hanoi')
    parser.add_argument('--dir', type=str, default='./datasets', help='Base directory for experiment outputs')
    parser.add_argument('--episodes', type=int, default=150, help='Number of episodes to record')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--name', type=str, default=None, help='Optional name override for experiment ID')
    parser.add_argument('--vision', action='store_true', help='Use vision-based observations')
    parser.add_argument('--relative_obs', action='store_true', help='Use relative gripper-goal features')
    parser.add_argument('--vla', action='store_true', help='Enable VLA-friendly formatting')
    parser.add_argument('--noise-std', type=float, default=0.03,
                        help='Std factor for Gaussian action noise (scaled by remaining distance).')
    parser.add_argument('--noisy-fraction', type=float, default=0.30,
                        help='Fraction of episodes that should use action noise (deterministic scheduling of the last fraction).')
    parser.add_argument('--cube-init-pos-noise-std', type=float, default=0.01,
                        help='Std dev (meters) for XY jitter of the initial tower position.')
    parser.add_argument('--random-block-placement', action='store_true', help='Place block on pegs randomly according to the rules of Towers of Hanoi')
    parser.add_argument('--random-block-selection', action='store_true', help='Randomly select 3 out of 4 blocks')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose debug output')
    parser.add_argument('--save-full-res-vid', action='store_true', help='Enable verbose debug output')

    # Currently not working
    parser.add_argument('--render', action='store_true', help='Render during execution')

    args = parser.parse_args()
    args.vla = True  # enforce VLA format

    # Seed everything
    np.random.seed(args.seed)

    # Build experiment directory structure
    exp_name = f"hanoi_random_block_selection_and_placement"
    timestamp = to_datestring(time.time())
    exp_id = args.name if args.name else timestamp
    args.env_dir = os.path.join(args.dir, exp_name, exp_id)
    os.makedirs(args.env_dir, exist_ok=True)
    traces_dir = os.path.join(args.env_dir, 'traces')
    os.makedirs(traces_dir, exist_ok=True)

    print(f"Starting experiment at {args.env_dir}")
    print(f"Environment settings: random_block_placement={args.random_block_placement}, random_block_selection={args.random_block_selection}")

    # 1) Make the env
    env = make_env(args)

    # Monkey‐patch Panda gripper names on the MjModel class
    model_cls = env.sim.model.__class__
    orig_body_name2id = model_cls.body_name2id

    def patched_body_name2id(self, name: str) -> int:
        try:
            return orig_body_name2id(self, name)
        except ValueError:
            # Panda names the fingers without "_inner_finger"
            alt = name.replace('_inner_finger', 'finger')
            return orig_body_name2id(self, alt)

    # Override the method on the class, so all instances use it
    model_cls.body_name2id = patched_body_name2id

    # 3) Now create the detector
    detector = get_detector(env, args.env)
    pddl_path = os.path.join('planning', 'PDDL', args.env.lower())

    # Wrap with recorder
    recorder = RecordDemos(
        env,
        vision_based=args.vision,
        detector=detector,
        pddl_path=pddl_path,
        args=args,
        render=args.render,
        randomize=True,
        noise_std_factor=args.noise_std
    )
    # Align recorder episode indexing so warmup reset in __init__ does not shift the schedule.
    # recorder.episode_idx = -1

    # Main loop: reset, run, save
    successes = 0
    epi = 0
    while successes < args.episodes:
        print(f"\n=== Episode {epi} ===")
        recorder.set_schedule(epi, args.episodes)
        print(f"[main] Scheduled ep {successes} / {args.episodes}: randomize={recorder.this_episode_randomize}")
        ok = record_episode_video(env, recorder, epi, args.save_full_res_vid)
        if not ok:
            print(f"Episode {epi} failed.")
        else:
            recorder.save_trajectory(epi)
            successes += 1
        # recorder.reset()
        epi += 1
        print(f"\n=== Successes {successes}/{args.episodes} ===")


if __name__ == '__main__':
    main()