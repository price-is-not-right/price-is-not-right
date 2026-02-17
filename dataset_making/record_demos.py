# record_demos.py

import os
import copy
import math
import gym
import cv2
import numpy as np
from planning.planner import add_predicates_to_pddl, call_planner
# from openpi.src.openpi.planning.hanoi_vlm_planner import query_model #COmmenting bc its breaking pi inference
from dataset_making.utils import to_datestring
from dataset_making.tasks import PickOperation, PlaceOperation, TurnOnOperation, TurnOffOperation

# Define which predicates to include per domain and planner mode
planning_predicates = {
    "Hanoi": ['on', 'clear', 'grasped', 'smaller'],
    "Hanoi4x3": ['on', 'clear', 'grasped', 'smaller'],
    "KitchenEnv": ['on', 'clear', 'grasped', 'stove_on'],
    "NutAssembly": ['on', 'clear', 'grasped'],
}
planning_mode = {"Hanoi": 0, "Hanoi4x3": 0, "KitchenEnv": 1, "NutAssembly": 0}

class RecordDemos(gym.Wrapper):
    """Gym wrapper to record demonstrations by symbolically planning and executing."""
    def __init__(
        self,
        env,
        vision_based: bool,
        detector,
        pddl_path: str,
        args,
        render: bool=False,
        randomize: bool=True,
        noise_std_factor: float=0.03
    ):
        super().__init__(env)
        self.env = env
        self.vision_based = vision_based
        self.detector = detector
        self.pddl_path = pddl_path
        # Ensure trailing slash for pddl path so planner joins files correctly
        if not self.pddl_path.endswith(os.sep):
            self.pddl_path += os.sep
        self.args = args
        self.render = render

        # Randomization / noise controls
        self.base_randomize_flag = randomize
        # Prefer CLI override if provided, else fall back to constructor default
        self.noise_std = float(getattr(args, "noise_std", noise_std_factor))
        # Fraction of episodes that should be noisy (deterministic schedule: last X%)
        self.noisy_fraction = float(getattr(args, "noisy_fraction", 0.30))
        # Total episodes (support either --episodes or --num_episodes)
        self.total_episodes = int(getattr(args, "episodes", getattr(args, "num_episodes", 0)) or 0)
        # Episode index (incremented in reset)
        self.episode_idx = -1
        # Will be computed per-episode in reset()
        self.this_episode_randomize = False
        # Deterministic scheduling state
        self.schedule_override = False
        self.scheduled_ep_index = -1
        self.reset_count = 0
        # Verbose mode flag
        self.verbose = getattr(args, "verbose", False)

        # Buffer for recording
        self.sequential_episode_buffer = []

        # Initialize symbolic plan
        # obs = self.reset()
        # The plan is already generated in reset() method
        if self.verbose:
            print("Task demonstrated:", self.plan)

    def set_schedule(self, ep_index: int, total_episodes: int):
        """Externally force the scheduling for this recording episode.
        This ensures deterministic application of noise regardless of how many resets occur."""
        self.total_episodes = int(total_episodes)
        self.scheduled_ep_index = int(ep_index)
        self.schedule_override = True
        # Compute and store the flag immediately
        if self.base_randomize_flag and self.total_episodes > 0:
            start_noisy_idx = max(0, int(math.floor((1.0 - self.noisy_fraction) * self.total_episodes)))
            self.this_episode_randomize = (self.scheduled_ep_index >= start_noisy_idx)
        else:
            # If totals unknown, fall back to probabilistic
            self.this_episode_randomize = self.base_randomize_flag and (np.random.rand() < self.noisy_fraction)

    def reset(self, seed=None) -> dict:
        """Reset environment and clear buffers."""
        self.reset_count += 1
        self.episode_idx += 1
        # Clear any previous trajectory data
        self.sequential_episode_buffer = []
        
        # Decide whether this episode uses action noise
        if self.schedule_override and self.total_episodes > 0:
            start_noisy_idx = max(0, int(math.floor((1.0 - self.noisy_fraction) * self.total_episodes)))
            # self.this_episode_randomize was already computed in set_schedule, but recompute for clarity
            self.this_episode_randomize = (self.scheduled_ep_index >= start_noisy_idx) and self.base_randomize_flag
            schedule_info = (
                f"deterministic override (record_ep={self.scheduled_ep_index}; noisy starts at {start_noisy_idx} of {self.total_episodes}; resets={self.reset_count})"
            )
        elif self.base_randomize_flag and self.total_episodes > 0:
            start_noisy_idx = max(0, int(math.floor((1.0 - self.noisy_fraction) * self.total_episodes)))
            self.this_episode_randomize = (self.episode_idx >= start_noisy_idx)
            schedule_info = f"deterministic (noisy episodes start at {start_noisy_idx} of {self.total_episodes}; resets={self.reset_count})"
        else:
            # Fallback: probabilistic if total not known
            self.this_episode_randomize = self.base_randomize_flag and (np.random.rand() < self.noisy_fraction)
            schedule_info = f"probabilistic (total episodes unknown; resets={self.reset_count})"
        if self.verbose:
            print(f"[RecordDemos] ep={self.episode_idx} randomize={self.this_episode_randomize} | {schedule_info}")
        
        # Reset the environment, handling varied return signatures
        try:
            raw = self.env.reset(seed=seed)
        except TypeError:
            raw = self.env.reset()
        
        # Extract observation if reset returns tuple or list
        if isinstance(raw, (tuple, list)):
            obs = raw[0]
        else:
            obs = raw
        
        # Ensure obs is a dict
        if not isinstance(obs, dict):
            obs = self.env._get_observations()

        # Forward the simulator state
        self.env.sim.forward()
        
        # Let the simulation settle for 50 timesteps
        action_space = self.env.action_space # Determine action shape for the robot
        neutral_action = np.zeros(action_space.shape, dtype=action_space.dtype)
        for _ in range(50):  # Let things settle for 50 timesteps
            obs, *_ = self.env.step(neutral_action)

        if self.verbose:
            print("After env.reset(), block positions:")
            for name in ["cube1_main", "cube2_main", "cube3_main"]:
                print(f"{name}: {self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(name)]}")

        # Detect init state
        state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        if self.verbose:
            print("Detector groundings:", state)
        
        # Include all relevant predicates, not just the True ones
        # The planner needs to know both what IS and what ISN'T true
        init_predicates = {}
        for predicate, value in state.items():
            if predicate.split('(')[0] in planning_predicates[self.args.env]:
                init_predicates[predicate] = value

        # After add_predicates_to_pddl:
        # Pass detected objects to dynamically generate PDDL
        # Filter predicates to only include those that involve active cubes
        active_cubes = []
        for cube in self.detector.objects:
            # Check if this cube is part of the tower by looking for 'on' predicates
            is_active = False
            for pred, value in init_predicates.items():
                if pred.startswith('on(') and value:
                    parts = pred.split('(')[1].split(',')
                    obj1 = parts[0]
                    obj2 = parts[1].rstrip(')')
                    if obj1 == cube or obj2 == cube:
                        is_active = True
                        break
            if is_active:
                active_cubes.append(cube)
        
        # Filter predicates to only include those involving active cubes or pegs
        filtered_predicates = {}
        for predicate, value in init_predicates.items():
            if value:
                # For smaller predicates, only include if both objects are active cubes or pegs
                if predicate.startswith('smaller('):
                    parts = predicate.split('(')[1].split(',')
                    obj1 = parts[0]
                    obj2 = parts[1].rstrip(')')
                    if (obj1 in active_cubes or obj1 in self.detector.object_areas) and \
                       (obj2 in active_cubes or obj2 in self.detector.object_areas):
                        filtered_predicates[predicate] = value
                else:
                    # Check if this predicate involves any active cubes
                    predicate_involves_active_cube = False
                    for cube in active_cubes:
                        if cube in predicate:
                            predicate_involves_active_cube = True
                            break
                    
                    # Also check if it involves any pegs (pegs are always valid)
                    if not predicate_involves_active_cube:
                        for peg in self.detector.object_areas:
                            if peg in predicate:
                                predicate_involves_active_cube = True
                                break
                    
                    if predicate_involves_active_cube:
                        filtered_predicates[predicate] = value
        
        detected_objects = {
            'cubes': active_cubes,  # Only include active cubes, not all detected cubes
            'pegs': self.detector.object_areas
        }
        
        add_predicates_to_pddl(self.pddl_path, filtered_predicates, detected_objects=detected_objects)
        
        # Generate a fresh plan for this episode based on current state
        # check 'planner' in selr.args.planner
        if self.args.planner == 'pddl':
            self.plan, _ = call_planner(self.pddl_path, problem="problem_save.pddl", mode=planning_mode[self.args.env])
        else:
            # call a vlm
            init_image = obs.get('agentview_image') #TODO: this obs does not have agentview_image. Figure out how to get the obs with agentview_image
            cv2.imshow("init_image", init_image)
            goal_image = self.args.goal_image_path #TODO: save the goal image in this path
            self.plan = query_model(init_image, goal_image, model=self.args.planner)
        print(f"Plan: {self.plan}")
        if not self.plan:
            print("There is no plan, resetting")
            return 
        
        # Convert PDDL plan to natural language
        natural_plan = self._convert_plan_to_natural_language(self.plan)
        print("Natural language plan:")
        for i, step in enumerate(natural_plan):
            print(f"  {i+1}. {step}")
        print()
        return obs

    def record_step(self, obs_dict: dict, action: np.ndarray):
        """Store a single step's data."""
        self.sequential_episode_buffer.append({
            'obs_dict': copy.deepcopy(obs_dict),
            'action': np.array(action, dtype=np.float32),
            'language_instruction': str(self.current_instruction)
        })

    def _convert_plan_to_natural_language(self, plan):
        """Convert PDDL plan to natural language commands."""
        natural_commands = []
        for op_str in plan:
            natural_commands.append(symbolic_to_natural_instruction(op_str))
        return natural_commands

    def _map_operator(self, op_str: str):
        """Map a PDDL operator string to a TaskOperation subclass and kwargs."""
        parts = op_str.lower().split()
        cmd = parts[0]
        if cmd == 'pick':
            return PickOperation, {'object_id': parts[1]}
        if cmd == 'place':
            return PlaceOperation, {
                'object_id': parts[1],
                'placement_id': parts[2]
            }
        if cmd == 'turn-on':
            return TurnOnOperation, {'object_id': parts[1]}
        if cmd == 'turn-off':
            return TurnOffOperation, {'object_id': parts[1]}
        raise ValueError(f"Unsupported operator: {op_str}")

    def run_trajectory(self, obs: dict) -> bool:
        """Execute the full symbolic plan, recording each step."""
        for op_str in self.plan:
            OpClass, params = self._map_operator(op_str)
            op = OpClass(
                self.env,
                self.detector,
                self.this_episode_randomize,
                self.noise_std,
                **params
            )
            self.current_instruction = symbolic_to_natural_instruction(op_str)
            success, obs = op.execute(obs)
            if not success:
                print(f"Failed to perform task: {op_str}")
                return False
        return True

    def save_trajectory(self, episode_idx: int) -> bool:
        """Format and save the recorded trajectory in RLDS .npy format."""
        from dataset_making.utils import quaternion_to_euler  # Ensure this is imported

        steps = []
        buffer = self.sequential_episode_buffer
        num_steps = len(buffer)

        if not buffer:
            print(f"No data to save for episode {episode_idx}.")
            return False

        for i, step_data in enumerate(buffer):
            obs = step_data['obs_dict']
            action = step_data['action']
            instr = step_data['language_instruction']

            # MAIN IMAGE
            agent_img = obs.get('agentview_image')
            if agent_img is None:
                print("Agent image is none")
                agent_img = np.zeros((256,256,3), dtype=np.uint8)
            else:
                agent_img = np.ascontiguousarray(agent_img[::-1, ::-1], dtype=np.uint8)
                # agent_img = np.asarray(agent_img, dtype=np.uint8)

            # WRIST IMAGE
            wrist_img = obs.get('robot0_eye_in_hand_image')
            if wrist_img is None:
                print("Wrist image is none")
                wrist_img = np.zeros((256,256,3), dtype=np.uint8)
            else:
                # wrist_img = np.asarray(wrist_img, dtype=np.uint8)
                wrist_img = np.ascontiguousarray(wrist_img[::-1, ::-1], dtype=np.uint8)

            # JOINT STATE
            joint_state = np.asarray(obs.get('robot0_joint_pos', np.zeros(7)), dtype=np.float32)

            # STATE (END EFFECTOR)
            eef_pos = obs.get('robot0_eef_pos', np.zeros(3, dtype=np.float32))
            eef_quat = obs.get('robot0_eef_quat', np.array([0., 0., 0., 1.], dtype=np.float32))
            eef_gripper = obs.get('robot0_gripper_qpos', np.zeros(2, dtype=np.float32))
            # Convert quaternion to axis angle
            eef_axis_angle = _quat2axisangle(eef_quat)

            eef_state = np.concatenate((eef_pos, eef_axis_angle, eef_gripper)).astype(np.float32)

            # RLDS step dict
            step_dict = {
                "action": action.astype(np.float32),
                "is_terminal": (i == num_steps - 1),
                "is_last": (i == num_steps - 1),
                "is_first": (i == 0),
                "reward": 1.0 if (i == num_steps - 1) else 0.0,  # or set as appropriate
                "discount": 1.0,
                "language_instruction": instr,
                "observation": {
                    "wrist_image": wrist_img,
                    "image": agent_img,
                    "state": eef_state,        
                    "joint_state": joint_state,
                }
            }
            steps.append(step_dict)

        # Compose RLDS episode dict
        out_dict = {
            "steps": steps,
            "episode_metadata": {
                "file_path": f"episode_{episode_idx}.npy"
            }
        }

        out_dir = os.path.join(self.args.env_dir, 'data')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'episode_{episode_idx}.npy')
        try:
            np.save(out_path, out_dict)
            print(f"Saved RLDS episode {episode_idx} ({len(steps)} steps) to {out_path}")
            self.sequential_episode_buffer = []
            return True
        except Exception as e:
            print(f"Error saving episode {episode_idx}: {e}")
            return False

def symbolic_to_natural_instruction(op_str):
    # Check if op_str is already in natural language. Natural language instructions start with 'pick the' or 'place the'
    if op_str.lower().startswith("pick the") or op_str.lower().startswith("place the"):
        return op_str  # Already in natural language
    # Example: "pick cube1" or "place cube1 cube3" etc.
    colors = {"cube1": "blue block", "cube2": "red block", "cube3": "green block", "cube4": "yellow block"}
    areas  = {"peg1": "left area", "peg2": "middle area", "peg3": "right area"}
    op = op_str.lower().split()
    if not op: return ""
    if op[0] == "pick":
        block = colors.get(op[1], op[1])
        if len(op) > 2:
            # PICK CUBE1 CUBE2 means pick cube1 from cube2
            from_obj = colors.get(op[2], op[2])
            return f"Pick the {block}."
        else:
            return f"Pick up the {block}."
    if op[0] == "place":
        block = colors.get(op[1], op[1])
        # Place target can be area or another block
        if op[2].startswith("cube"):
            target = colors.get(op[2], op[2])
            return f"Place the {block} on top of the {target}."
        else:
            area = areas.get(op[2], op[2])
            return f"Place the {block} in the {area}."
    # Add any other op cases as needed
    return op_str  # fallback

def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den