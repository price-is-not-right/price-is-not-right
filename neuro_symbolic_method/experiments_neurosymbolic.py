import warnings
warnings.filterwarnings("ignore")

import os
import argparse
import copy
import gym
import joblib
import yaml
import robosuite as suite
import numpy as np
from statistics import mean
from robosuite.wrappers import GymWrapper
from robosuite.utils.detector import (
    HanoiDetector, KitchenDetector, NutAssemblyDetector,
    CubeSortingDetector, HeightStackingDetector,
    AssemblyLineSortingDetector, PatternReplicationDetector,
)
from planning.planner import (
    add_predicates_to_pddl, define_goal_in_pddl, call_planner
)
from planning.executor import Executor_Diffusion
from ultralytics import YOLO
from robot_utils import SUPPORTED_ROBOTS, reset_gripper_target

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DETECTOR_REGISTRY = {
    "HanoiDetector": HanoiDetector,
    "KitchenDetector": KitchenDetector,
    "NutAssemblyDetector": NutAssemblyDetector,
    "CubeSortingDetector": CubeSortingDetector,
    "HeightStackingDetector": HeightStackingDetector,
    "AssemblyLineSortingDetector": AssemblyLineSortingDetector,
    "PatternReplicationDetector": PatternReplicationDetector,
}

CONFIG_FILE_REGISTRY = {
    "Hanoi": "configs/hanoi.yaml",
    "KitchenEnv": "configs/kitchenenv.yaml",
    "NutAssembly": "configs/nutassembly.yaml",
    "CubeSorting": "configs/cubesorting.yaml",
    "HeightStacking": "configs/heightstacking.yaml",
    "AssemblyLineSorting": "configs/assemblylinesort.yaml",
    "PatternReplication": "configs/patternreplication.yaml",
}

# Per-robot config overrides: --robot alone selects the matching yaml (policies,
# regressors and horizons are robot-specific). Falls back to CONFIG_FILE_REGISTRY.
ROBOT_CONFIG_REGISTRY = {
    "Kinova3": {
        "Hanoi": "kinova/configs/hanoi.yaml",
    },
}

TERMINATION_CONDITIONS = {
    "pick":       lambda state, symgoal: state[f"grasped({symgoal[0]})"],
    "drop":       lambda state, symgoal: state[f"on({symgoal[0]},{symgoal[1]})"] and not state[f"grasped({symgoal[0]})"],
    "reach_pick": lambda state, symgoal: state[f"over(gripper,{symgoal[0]})"],
    "reach_drop": lambda state, symgoal: state[f"over(gripper,{symgoal[1]})"],
    "turnon":     lambda state, symgoal: state["stove_on()"],
    "turnoff":    lambda state, symgoal: not state["stove_on()"],
    "default":    lambda state, symgoal: False,
}

RESET_GRIPPER_POS = reset_gripper_target("Panda")

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def resolve_config_path(env_name: str, robot: str = None, config_path: str = None) -> str:
    if config_path:
        return config_path
    robot_configs = ROBOT_CONFIG_REGISTRY.get(robot or "Panda", {})
    return robot_configs.get(env_name, CONFIG_FILE_REGISTRY[env_name])


def load_env_config(env_name: str, robot: str = None, config_path: str = None) -> dict:
    path = resolve_config_path(env_name, robot=robot, config_path=config_path)
    print(f"Config: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Executor building
# ---------------------------------------------------------------------------

def resolve_policies(cfg: dict, use_yolo: bool) -> dict:
    return dict(cfg["policies"])


def build_executors(cfg: dict, n_act: int, debug: bool, use_yolo: bool = False) -> list:
    policies = resolve_policies(cfg, use_yolo=use_yolo)
    executors = []
    for spec in cfg["executors"]:
        horizon = eval(spec["horizon_formula"], {"n_act": n_act})
        print(f"Building executor {spec['id']} with horizon {horizon} "
              f"(oracle={spec['oracle']}, use_yolo={use_yolo})")
        executor = Executor_Diffusion(
            id=spec["id"],
            policy=policies[spec["policy_key"]],
            Beta=TERMINATION_CONDITIONS[spec["termination_condition"]],
            nulified_action_indexes=spec["nulified_action_indexes"],
            nulified_action_values=spec.get("nulified_action_values"),
            oracle=spec["oracle"],
            use_yolo=use_yolo,
            horizon=horizon,
            debug=debug,
        )
        executors.append(executor)
    return executors


# ---------------------------------------------------------------------------
# Goal generation
# ---------------------------------------------------------------------------

def build_goal_predicates(cfg: dict, state: dict, detector) -> list:
    strategy = cfg["goal_strategy"]
    params = cfg.get("goal_params", {})

    if strategy == "default":
        return []

    elif strategy == "cube_sorting":
        small_pred = params["small_predicate"]
        goal_predicates = []
        for predicate, value in state.items():
            if small_pred in predicate:
                obj = predicate[predicate.find("(") + 1:predicate.find(")")].split(", ")[0]
                target = params["target_true"] if value else params["target_false"]
                goal_predicates.append(f"on {obj} {target}")
        return goal_predicates

    elif strategy == "height_stacking":
        size_pred = params["size_predicate"]
        base = params["base_location"]
        sizes = {}
        for predicate, value in state.items():
            if size_pred in predicate and value:
                objs = predicate[predicate.find("(") + 1:predicate.find(")")].split(",")
                sizes[objs[0]] = objs[1]
        sorted_sizes = sorted(sizes.items(), key=lambda x: x[1])
        goal_predicates = [
            f"on {sorted_sizes[i][0]} {sorted_sizes[i + 1][0]}"
            for i in range(len(sorted_sizes) - 1)
        ]
        goal_predicates.append(f"on {sorted_sizes[-1][0]} {base}")
        return goal_predicates

    elif strategy == "type_match":
        match_pred = params["match_predicate"]
        goal_predicates = []
        for predicate, value in state.items():
            if match_pred in predicate and value:
                objs = predicate[predicate.find("(") + 1:predicate.find(")")].split(",")
                goal_predicates.append(f"on {objs[0]} {objs[1]}")
        return goal_predicates

    elif strategy == "pattern_replication":
        return detector.get_pattern_replication_goal()

    else:
        raise ValueError(f"Unknown goal strategy: {strategy}")


def get_plan(state: dict, cfg: dict, detector) -> tuple:
    planning_predicates = cfg["planning_predicates"]
    pddl_path = cfg["pddl_path"]
    mode = cfg["planning_mode"]
    init_predicates = {
        pred: True
        for pred, val in state.items()
        if val and pred.split("(")[0] in planning_predicates and "ref" not in pred
    }
    print(f"Initial predicates for planning: {init_predicates}")
    add_predicates_to_pddl(pddl_path, init_predicates)

    goal_predicates = build_goal_predicates(cfg, state, detector)
    if len(goal_predicates) > 0:
        define_goal_in_pddl(pddl_path, goal_predicates)
    plan, _ = call_planner(pddl_path, mode=mode)
    return plan, goal_predicates


# ---------------------------------------------------------------------------
# Environment wrapper
# ---------------------------------------------------------------------------

class DictObs(gym.Env):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self):
        self.env.reset()
        return self.env._get_observations()

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        return self.env._get_observations(), reward, terminated or truncated, info

    def render(self, mode="human", *args, **kwargs):
        self.env.render()

    def _get_observations(self):
        return self.env._get_observations()

    def __getattr__(self, name):
        return getattr(self.env, name)


# ---------------------------------------------------------------------------
# Gripper reset
# ---------------------------------------------------------------------------

def reset_gripper(env, detector, render: bool, max_open_steps: int = 80, max_home_steps: int = 200):
    print("Resetting gripper")
    state = detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)

    for step_i in range(max_open_steps):
        if state.get("open_gripper(gripper)", False):
            break
        env.step(np.array([0.0, 0.0, 0.35, -1.0]))
        state = detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        if render:
            env.render()
    else:
        ap = detector.open("gripper", return_distance=True)
        print(f"[reset_gripper] open timeout after {max_open_steps} steps "
              f"(aperture={ap}); continuing")

    for _ in range(10):
        env.step(np.array([0.0, 0.0, 0.5, -1.0]))
        if render:
            env.render()

    gripper_pos = np.asarray(env._get_observations()["robot0_eef_pos"], dtype=float)
    for step_i in range(max_home_steps):
        delta = RESET_GRIPPER_POS - gripper_pos
        dist = float(np.linalg.norm(delta))
        if dist <= 0.01:
            break
        action = np.clip(4.0 * delta, -1.0, 1.0)
        env.step(np.array([action[0], action[1], action[2], -1.0]))
        if render:
            env.render()
        gripper_pos = np.asarray(env._get_observations()["robot0_eef_pos"], dtype=float)
    else:
        dist = float(np.linalg.norm(RESET_GRIPPER_POS - gripper_pos))
        print(f"[reset_gripper] home timeout after {max_home_steps} steps "
              f"(dist={dist:.4f}m); continuing")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Hanoi",
                        choices=list(CONFIG_FILE_REGISTRY.keys()))
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--rnd_reset", action='store_true')
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use_yolo", action="store_true",
                        help="Use YOLO + regressor perception instead of sim ground truth.")
    parser.add_argument("--robot", type=str, default=None, choices=list(SUPPORTED_ROBOTS),
                        help="Manipulator model (default: Panda). Selects the matching "
                             "config from ROBOT_CONFIG_REGISTRY unless --config is given.")
    parser.add_argument("--config", type=str, default=None,
                        help="Override the env yaml resolved from --env/--robot.")
    parser.add_argument("--n_act", type=int, default=None,
                        help="Actions executed per policy step (default: cfg['n_act'] or 8).")
    parser.add_argument("--n_obs", type=int, default=None,
                        help="Observation horizon (default: cfg['n_obs'] or 16).")
    parser.add_argument("--n_ep", type=int, default=100)
    args = parser.parse_args()

    np.random.seed(args.seed)


    cfg = load_env_config(args.env, robot=args.robot, config_path=args.config)
    robot = args.robot or cfg.get("robot", "Panda")
    if args.n_act is None:
        args.n_act = int(cfg.get("n_act", 8))
    if args.n_obs is None:
        args.n_obs = int(cfg.get("n_obs", 16))
    if robot not in SUPPORTED_ROBOTS:
        raise ValueError(f"Unsupported robot {robot!r}; expected one of {SUPPORTED_ROBOTS}")
    global RESET_GRIPPER_POS
    RESET_GRIPPER_POS = np.asarray(
        cfg.get("reset_gripper_pos", reset_gripper_target(robot)), dtype=np.float64
    )
    print(f"Robot: {robot}  |  reset_gripper_pos={np.round(RESET_GRIPPER_POS, 4)}  "
          f"|  n_act={args.n_act} n_obs={args.n_obs}")

    actions = build_executors(cfg, args.n_act, args.debug, use_yolo=args.use_yolo)

    if args.use_yolo:
        print(f"Loading YOLO model: {cfg['yolo_model']}")
        yolo_model = YOLO(cfg["yolo_model"])
        print(f"Loading regressor model: {cfg['regressor_model']}")
        regressor_model = joblib.load(cfg["regressor_model"])
    else:
        yolo_model = None
        regressor_model = None

    controller_config = suite.load_controller_config(default_controller="OSC_POSITION")
    env = suite.make(
        env_name=args.env,
        robots=robot,
        controller_configs=controller_config,
        has_renderer=args.render,
        has_offscreen_renderer=True,
        reward_shaping=True,
        control_freq=20,
        horizon=20000,
        use_camera_obs=True,
        use_object_obs=False,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=args.size,
        camera_widths=args.size,
        random_block_placement=args.rnd_reset,
    )
    detector = DETECTOR_REGISTRY[cfg["detector"]](env)
    env = DictObs(GymWrapper(env))

    episode_successes = 0
    num_valid_pick_place_queries = 0
    valid_pick_place_success = 0
    pick_place_successes = []
    percentage_advancement = []
    retry_reset = False

    for i in range(args.n_ep):
        if retry_reset:
            i -= 1
        print(f"Episode: {i}")
        plan = False
        goal_reached = False
        np.random.seed(args.seed + i)

        while plan is False:
            env.reset()
            if args.render:
                env.render()

            observations = []
            for _ in range(args.n_obs):
                env.step(np.zeros(env.action_space.shape))
                obs = env._get_observations()
                obs["objects_pos"] = detector.get_all_objects_pos()
                observations.append(obs)
            observations = observations[-args.n_obs:]

            state = detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            plan, goal_predicates = get_plan(state, cfg, detector)

        print(f"Plan: {plan}")
        print(f"Goal predicates: {goal_predicates}")

        pick_place_success = 0
        n_ops = len(plan) / 2
        perception_tracking = {}

        def wipe_all_perception(preserve_last_known=False):
            """Reset YOLO tracking on every skill executor."""
            nonlocal perception_tracking
            if not (args.use_yolo and actions):
                perception_tracking = {}
                return
            actions[0].reset_tracking(preserve_last_known=preserve_last_known)
            perception_tracking = copy.deepcopy(actions[0].get_tracking_data())
            for ex in actions:
                ex.set_tracking_data(copy.deepcopy(perception_tracking))

        def refresh_yolo_poses(prefer_stereo=True, min_conf2=0.35, force_ids=None,
                               skip_grasped=True):
            """Re-detect visible cubes and update last_known from the cameras.

            - Stereo (conf2 >= min_conf2): always update.
            - Mono / weak wrist: only fill missing cubes, never overwrite a prior.
            - force_ids: always update these PDDL ids when detected.
            - skip_grasped: skip cubes currently in the gripper, whose mid-air
              position says nothing about where they will come to rest.
            """
            nonlocal perception_tracking, observations
            if not (args.use_yolo and actions):
                return
            force_ids = set(force_ids or [])
            grasped = set()
            if skip_grasped:
                try:
                    g = detector.get_groundings(as_dict=True, binary_to_float=False,
                                                return_distance=False)
                    for pred, val in (g or {}).items():
                        if val and pred.startswith("grasped("):
                            grasped.add(pred[len("grasped("):-1])
                except Exception:
                    grasped = set()
            obs = env._get_observations()
            obs["objects_pos"] = detector.get_all_objects_pos()
            if observations:
                observations[-1] = obs
            else:
                observations = [obs]
            ee = obs["robot0_eef_pos"]
            img1 = np.array(obs["agentview_image"].reshape((args.size, args.size, 3)), dtype=np.uint8)
            img2 = np.array(obs["robot0_eye_in_hand_image"].reshape((args.size, args.size, 3)), dtype=np.uint8)
            ex0 = actions[0]
            if not getattr(ex0, "yolo_model", None):
                ex0.load_policy(detector=detector, yolo_model=yolo_model,
                                regressor_model=regressor_model, image_size=args.size)
            pred, rel = ex0.detect_cubes_simple(
                img1, img2, ee, conf_threshold=0.75,
                sim=getattr(env, "sim", None), render=False,
            )
            if rel:
                ex0.relations = rel
                ex0.map_id_semantic = {y: p for p, y in rel.items()}
            if not hasattr(ex0, "last_known_semantic_positions"):
                ex0.last_known_semantic_positions = {}
            updated = []
            for pddl_id, yolo_id in (rel or {}).items():
                if yolo_id not in pred:
                    continue
                if skip_grasped and pddl_id in grasped and pddl_id not in force_ids:
                    continue
                conf2 = float(getattr(ex0, "_last_det_conf2", {}).get(yolo_id, 0.0))
                new = np.asarray(pred[yolo_id], dtype=np.float64)
                z = float(new[2])
                if z < 0.80 or z > 0.96:
                    continue
                prev = ex0.last_known_semantic_positions.get(pddl_id)
                forced = pddl_id in force_ids
                stereo_ok = conf2 >= min_conf2
                if prev is not None and not stereo_ok and not forced:
                    # Keep a prior fix over a weak mono refresh.
                    continue
                if prev is None and not stereo_ok and not forced:
                    if prefer_stereo or conf2 < 0.2:
                        continue
                if prev is not None:
                    prev = np.asarray(prev, dtype=np.float64)
                    if float(np.linalg.norm(new - prev)) > 0.10:
                        continue
                ex0.last_known_semantic_positions[pddl_id] = new
                updated.append(f"{pddl_id}:c2={conf2:.2f}")
            perception_tracking = copy.deepcopy(ex0.get_tracking_data())
            for ex in actions:
                ex.set_tracking_data(copy.deepcopy(perception_tracking))
            if updated:
                print(f"[YOLO] pose refresh updated [{', '.join(updated)}]")

        def reset_gripper_and_perception():
            nonlocal perception_tracking
            reset_gripper(env, detector, args.render)
            if args.use_yolo and actions:
                wipe_all_perception(preserve_last_known=True)
                print("[YOLO] tracking reset after gripper reset "
                      f"(kept last_known={list(actions[0].last_known_semantic_positions.keys())})")
                # Home camera view: re-localize every visible cube (incl. ones
                # shifted when a neighbor was lifted).
                refresh_yolo_poses(prefer_stereo=False, min_conf2=0.35)

        # Fresh episode: never carry poses from a previous episode's final layout.
        wipe_all_perception(preserve_last_known=False)
        if args.use_yolo:
            print("[YOLO] episode tracking wipe (no last_known carry-over)")

        reset_gripper_and_perception()

        for j, operator in enumerate(plan):
            print(f"\nExecuting operator: {operator}")
            parts = operator.split()
            op_name = parts[0].lower()

            if op_name == "pick":
                if len(parts) < 3:
                    raise ValueError(f"Malformed PICK operator: {operator}")
                obj_to_pick = parts[1].lower()
                obj_to_drop = parts[2].lower()
                skill = actions[:2]
                sub_goal = (obj_to_pick, obj_to_drop)
                print(f"Picking: {obj_to_pick}")
            elif op_name == "place":
                if len(parts) < 3:
                    raise ValueError(f"Malformed PLACE operator: {operator}")
                obj_to_pick = parts[1].lower()
                obj_to_drop = parts[2].lower()
                skill = actions[2:4]
                sub_goal = (obj_to_pick, obj_to_drop)
                num_valid_pick_place_queries += 1
                print(f"Placing: {obj_to_pick} on {obj_to_drop}")
            elif op_name == "turnon":
                skill = [actions[-1]]
                sub_goal = (None, None)
                print("Turn On action")
            elif op_name == "turnoff":
                skill = [actions[-2]]
                sub_goal = (None, None)
                print("Turn Off action")
            elif op_name == "wait":
                skill = []
                sub_goal = (None, None)
                print("Wait action")
            else:
                raise ValueError(f"Unknown operator: {operator}")

            reach_failed = False
            reach_ok = False
            reach_drop_ok = False
            skill_retries = 0
            place_op_retries = 0
            pick_op_retries = 0
            while True:
              for action_step in skill:
                action_step.load_policy(
                    detector=detector,
                    yolo_model=yolo_model,
                    regressor_model=regressor_model,
                    image_size=args.size,
                )
                if args.use_yolo:
                    action_step.set_tracking_data(copy.deepcopy(perception_tracking))

                if args.debug:
                    print(f"\tExecuting action: {action_step.id}")
                    print(len(observations))
                observations, success, goal_reached = action_step.execute(
                    env, observations, args.n_act, sub_goal,
                    goal_predicates.copy(), args.render,
                )
                print(f"\t{action_step.id}: {'Success' if success else 'Failed'}")
                if not success and sub_goal:
                    _st = detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
                    o0 = sub_goal[0]
                    o1 = sub_goal[1] if len(sub_goal) > 1 else None
                    _pos = detector.get_all_objects_pos()
                    g = np.asarray(_pos.get("gripper", [0, 0, 0]), dtype=float)
                    t = np.asarray(_pos.get(o0, [0, 0, 0]), dtype=float)
                    xy = float(np.linalg.norm(g[:2] - t[:2]))
                    dz = float(abs(g[2] - t[2]))
                    print(
                        f"\t  fail-state grasped({o0})={_st.get(f'grasped({o0})')} "
                        f"over(g,{o0})={_st.get(f'over(gripper,{o0})')} "
                        f"over(g,{o1})={_st.get(f'over(gripper,{o1})')} "
                        f"on({o0},{o1})={_st.get(f'on({o0},{o1})')} "
                        f"open={_st.get('open_gripper(gripper)')} "
                        f"at_grab={_st.get(f'at_grab_level(gripper,{o0})')} "
                        f"picked_up({o0})={_st.get(f'picked_up({o0})')} "
                        f"xy={xy*1000:.1f}mm dz={dz*1000:.1f}mm "
                        f"g={np.round(g, 3)} t={np.round(t, 3)}"
                    )
                if args.use_yolo:
                    perception_tracking = copy.deepcopy(action_step.get_tracking_data())
                state = detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)

                # Retry failed ReachPick up to 2x with fresh snapshots.
                while (args.use_yolo and not success and op_name == "pick"
                       and action_step.id == "ReachPick" and skill_retries < 2):
                    skill_retries += 1
                    print(f"[YOLO] ReachPick failed; retry {skill_retries}/2 with fresh snapshot")
                    if actions:
                        for ex in actions:
                            ex._skill_snapshot_pos = None
                        if sub_goal and sub_goal[0]:
                            for ex in actions:
                                lk = getattr(ex, "last_known_semantic_positions", None)
                                if lk is not None:
                                    lk.pop(sub_goal[0], None)
                        refresh_yolo_poses(prefer_stereo=False, min_conf2=0.35)
                    action_step.set_tracking_data(copy.deepcopy(perception_tracking))
                    observations, success, goal_reached = action_step.execute(
                        env, observations, args.n_act, sub_goal,
                        goal_predicates.copy(), args.render,
                    )
                    print(f"\t{action_step.id} retry: {'Success' if success else 'Failed'}")
                    if args.use_yolo:
                        perception_tracking = copy.deepcopy(action_step.get_tracking_data())
                    state = detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)

                if success and op_name == "pick" and action_step.id == "ReachPick":
                    reach_ok = True
                if success and op_name == "place" and action_step.id == "ReachDrop":
                    reach_drop_ok = True

                # Retry failed ReachDrop once.
                if (args.use_yolo and not success and op_name == "place"
                        and action_step.id == "ReachDrop" and skill_retries < 1):
                    skill_retries += 1
                    print("[YOLO] ReachDrop failed; retrying once with fresh snapshot")
                    if actions:
                        for ex in actions:
                            ex._skill_snapshot_pos = None
                        if sub_goal and sub_goal[1] and str(sub_goal[1]).startswith("cube"):
                            for ex in actions:
                                lk = getattr(ex, "last_known_semantic_positions", None)
                                if lk is not None:
                                    lk.pop(sub_goal[1], None)
                        refresh_yolo_poses(prefer_stereo=False, min_conf2=0.35)
                    action_step.set_tracking_data(copy.deepcopy(perception_tracking))
                    observations, success, goal_reached = action_step.execute(
                        env, observations, args.n_act, sub_goal,
                        goal_predicates.copy(), args.render,
                    )
                    print(f"\t{action_step.id} retry: {'Success' if success else 'Failed'}")
                    if args.use_yolo:
                        perception_tracking = copy.deepcopy(action_step.get_tracking_data())
                    state = detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)

                # Retry Pick once if reach succeeded but grasp failed.
                if (not success and op_name == "pick"
                        and action_step.id == "Pick" and reach_ok and skill_retries < 3):
                    skill_retries += 1
                    print("Pick failed after ReachPick; retrying grasp")
                    action_step.set_tracking_data(copy.deepcopy(perception_tracking))
                    observations, success, goal_reached = action_step.execute(
                        env, observations, args.n_act, sub_goal,
                        goal_predicates.copy(), args.render,
                    )
                    print(f"\t{action_step.id} retry: {'Success' if success else 'Failed'}")
                    if args.use_yolo:
                        perception_tracking = copy.deepcopy(action_step.get_tracking_data())
                    state = detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)

                # Retry Drop once if reach succeeded but release failed.
                if (args.use_yolo and not success and op_name == "place"
                        and action_step.id == "Drop" and reach_drop_ok and skill_retries < 3):
                    skill_retries += 1
                    print("[YOLO] Drop failed after ReachDrop; retrying drop once")
                    action_step.set_tracking_data(copy.deepcopy(perception_tracking))
                    observations, success, goal_reached = action_step.execute(
                        env, observations, args.n_act, sub_goal,
                        goal_predicates.copy(), args.render,
                    )
                    print(f"\t{action_step.id} retry: {'Success' if success else 'Failed'}")
                    if args.use_yolo:
                        perception_tracking = copy.deepcopy(action_step.get_tracking_data())
                    state = detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)

                if (not success and op_name == "pick"
                        and action_step.id == "ReachPick"):
                    reach_failed = True
                    break
                if (not success and op_name == "place"
                        and action_step.id == "ReachDrop"):
                    break
              # Full place-operator retry: Drop sometimes needs a fresh ReachDrop
              # approach after a near-miss release (peg or cube support).
              if (op_name == "place" and not success
                      and place_op_retries < 1 and not reach_failed):
                  place_op_retries += 1
                  skill_retries = 0
                  reach_drop_ok = False
                  print("Place operator failed; retrying ReachDrop+Drop once")
                  if actions:
                      for ex in actions:
                          ex._skill_snapshot_pos = None
                      if args.use_yolo and sub_goal and sub_goal[1] and str(sub_goal[1]).startswith("cube"):
                          refresh_yolo_poses(
                              prefer_stereo=False, min_conf2=0.35,
                              force_ids=[sub_goal[1]])
                  continue
              # Full pick-operator retry: ReachPick→Pick miss often needs a fresh
              # approach, not just re-grasp (gripper may have closed off-target).
              if (op_name == "pick" and not success
                      and pick_op_retries < 1 and not reach_failed):
                  pick_op_retries += 1
                  skill_retries = 0
                  reach_ok = False
                  print("Pick operator failed; retrying ReachPick+Pick once")
                  if actions:
                      for ex in actions:
                          ex._skill_snapshot_pos = None
                      if args.use_yolo and sub_goal and sub_goal[0]:
                          for ex in actions:
                              lk = getattr(ex, "last_known_semantic_positions", None)
                              if lk is not None:
                                  lk.pop(sub_goal[0], None)
                          refresh_yolo_poses(
                              prefer_stereo=False, min_conf2=0.35,
                              force_ids=[sub_goal[0]])
                  continue
              # If ReachPick itself failed after its retries, allow one full
              # pick-operator re-approach with wiped pick-target memory.
              if (op_name == "pick" and reach_failed
                      and pick_op_retries < 1):
                  pick_op_retries += 1
                  skill_retries = 0
                  reach_failed = False
                  reach_ok = False
                  print("ReachPick failed; retrying full pick approach once")
                  if actions:
                      for ex in actions:
                          ex._skill_snapshot_pos = None
                      if args.use_yolo and sub_goal and sub_goal[0]:
                          for ex in actions:
                              lk = getattr(ex, "last_known_semantic_positions", None)
                              if lk is not None:
                                  lk.pop(sub_goal[0], None)
                          refresh_yolo_poses(prefer_stereo=False, min_conf2=0.35)
                  continue
              break
            if reach_failed:
                success = False
            if op_name == "pick":
                if not success:
                    print(f"--- Failed PICK. {pick_place_success}/{n_ops} ({pick_place_success/n_ops:.0%})")
                    print("Aborting episode; moving to next.")
                    break
                elif args.use_yolo and actions:
                    # Support may have shifted — refresh all visible cubes from cameras
                    # instead of blindly deleting the support's last_known.
                    refresh_yolo_poses(prefer_stereo=False, min_conf2=0.35)
            elif op_name == "place":
                if success:
                    pick_place_success += 1
                    valid_pick_place_success += 1
                    print(f"+++ Picked and placed. {pick_place_success}/{n_ops} ({pick_place_success/n_ops:.0%})")
                    if j != len(plan) - 1:
                        try:
                            reset_gripper_and_perception()
                        except ValueError:
                            retry_reset = True
                            break
                else:
                    print(f"--- Failed. {pick_place_success}/{n_ops} ({pick_place_success/n_ops:.0%})")
                    print("Aborting episode; moving to next.")
                    break
        n_ops = len(plan) / 2
        pick_place_successes.append(pick_place_success)
        percentage_advancement.append(pick_place_success / n_ops)

        if goal_reached or pick_place_success / n_ops == 1.0:
            episode_successes += 1
            print("Episode succeeded.")

        ep = i + 1
        pick_place_rate = (
            valid_pick_place_success / num_valid_pick_place_queries
            if num_valid_pick_place_queries > 0
            else 0.0
        )
        print(f"Success rate: {episode_successes / ep:.3f}")
        print(f"Successful pick_place: {pick_place_successes}")
        print(f"Mean successful pick_place: {mean(pick_place_successes):.2f}")
        print(f"Mean percentage advancement: {mean(percentage_advancement):.2%}")
        print(f"Pick-place success rate: {pick_place_rate:.3f}\n")

        os.makedirs("results", exist_ok=True)
        with open(f"results/results_neurosym_seed_{args.seed}.txt", "w") as f:
            f.write(f"Success rate: {episode_successes / 100}\n")
            f.write(f"Mean successful pick_place: {mean(pick_place_successes)}\n")
            f.write(f"Mean percentage advancement: {mean(percentage_advancement)}\n")
            f.write(f"Pick-place success rate: {pick_place_rate}\n")


if __name__ == "__main__":
    main()
