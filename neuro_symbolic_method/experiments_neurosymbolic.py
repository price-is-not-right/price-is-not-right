import warnings
warnings.filterwarnings("ignore")

import os
import argparse
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
# import wandb

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

# Termination conditions registry — logic stays in Python, keyed by name
TERMINATION_CONDITIONS = {
    "pick":       lambda state, symgoal: state[f"grasped({symgoal[0]})"],
    "drop":       lambda state, symgoal: state[f"on({symgoal[0]},{symgoal[1]})"] and not state[f"grasped({symgoal[0]})"],
    "reach_pick": lambda state, symgoal: state[f"over(gripper,{symgoal[0]})"],
    "reach_drop": lambda state, symgoal: state[f"over(gripper,{symgoal[1]})"],
    "turnon":     lambda state, symgoal: state["stove_on()"],
    "turnoff":    lambda state, symgoal: not state["stove_on()"],
    "default":    lambda state, symgoal: False,
}

RESET_GRIPPER_POS = np.array([-0.080193391, -0.03391656, 0.95828137])

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_env_config(env_name: str) -> dict:
    config_path = CONFIG_FILE_REGISTRY[env_name]
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Executor building
# ---------------------------------------------------------------------------

def build_executors(cfg: dict, n_act: int, debug: bool) -> list:
    policies = cfg["policies"]
    executors = []
    for spec in cfg["executors"]:
        horizon = eval(spec["horizon_formula"], {"n_act": n_act})
        print(f"Building executor {spec['id']} with horizon {horizon}")
        executor = Executor_Diffusion(
            id=spec["id"],
            policy=policies[spec["policy_key"]],
            Beta=TERMINATION_CONDITIONS[spec["termination_condition"]],
            nulified_action_indexes=spec["nulified_action_indexes"],
            oracle=spec["oracle"],
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

def reset_gripper(env, detector, render: bool):
    print("Resetting gripper")
    state = detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
    while not state["open_gripper(gripper)"]:
        env.step(np.array([0, 0, 0, -1]))
        state = detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        if render:
            env.render()

    for _ in range(5):
        env.step(np.array([0, 0, 0.5, 0]))
        if render:
            env.render()

    gripper_pos = env._get_observations()["robot0_eef_pos"]
    delta = RESET_GRIPPER_POS - gripper_pos
    while np.linalg.norm(delta) > 0.01:
        action = 5 * np.array([delta[0], delta[1], delta[2], 0]) * 0.9
        env.step(action)
        if render:
            env.render()
        gripper_pos = env._get_observations()["robot0_eef_pos"]
        delta = RESET_GRIPPER_POS - gripper_pos


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
    parser.add_argument("--n_act", type=int, default=4)
    parser.add_argument("--n_obs", type=int, default=16)
    parser.add_argument("--n_ep", type=int, default=100)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # # WandB
    # wandb.init(
    #     project="vlas-cycliclxm",
    #     name="neurosymbolic-experiment-hs3",
    #     settings=wandb.Settings(x_stats_sampling_interval=0.5),
    # )

    # Load config
    cfg = load_env_config(args.env)

    # Build executors
    actions = build_executors(cfg, args.n_act, args.debug)

    # Load models
    # yolo_model = YOLO(cfg["yolo_model"])
    # regressor_model = joblib.load(cfg["regressor_model"])
    yolo_model = None
    regressor_model = None

    # Build robosuite environment
    controller_config = suite.load_controller_config(default_controller="OSC_POSITION")
    env = suite.make(
        env_name=args.env,
        robots="Panda",
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

        # Reset until a valid plan is found
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
        reset_gripper(env, detector, args.render)
        tracking_data = {}

        for j, operator in enumerate(plan):
            print(f"\nExecuting operator: {operator}")

            # Operators come in reach/act pairs; collect obj_to_pick on even steps
            if j % 2 == 0:
                obj_to_pick = operator.split(" ")[-2].lower()
                continue
            else:
                obj_to_drop = operator.split(" ")[-1].lower()

            num_valid_pick_place_queries += 1
            op_name = operator.split(" ")[0].lower()

            if op_name == "turnon":
                skill = actions[-1]
                print("Turn On action")
            elif op_name == "turnoff":
                skill = actions[-2]
                print("Turn Off action")
            elif op_name == "wait":
                skill = []
                print("Wait action")
            else:
                skill = actions[:4]
                print(f"Picking: {obj_to_pick}, Dropping on: {obj_to_drop}")

            for action_step in skill:
                action_step.load_policy(
                    detector=detector,
                    yolo_model=yolo_model,
                    regressor_model=regressor_model,
                    image_size=args.size,
                )
                if tracking_data:
                    action_step.set_tracking_data(tracking_data)

                print(f"\tExecuting action: {action_step.id}")
                sub_goal = (obj_to_pick, obj_to_drop)
                print(len(observations))
                observations, success, goal_reached = action_step.execute(
                    env, observations, args.n_act, sub_goal,
                    goal_predicates.copy(), args.render,
                )
                tracking_data = action_step.get_tracking_data()
                state = detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)

            n_ops = len(plan) / 2
            if success:
                pick_place_success += 1
                valid_pick_place_success += 1
                print(f"+++ Picked and placed. {pick_place_success}/{n_ops} ({pick_place_success/n_ops:.0%})")
                if operator == plan[-1]:
                    continue
                try:
                    reset_gripper(env, detector, args.render)
                except ValueError:
                    retry_reset = True
                    break
            else:
                print(f"--- Failed. {pick_place_success}/{n_ops} ({pick_place_success/n_ops:.0%})")
                try:
                    reset_gripper(env, detector, args.render)
                except ValueError:
                    retry_reset = True
                    break

        n_ops = len(plan) / 2
        pick_place_successes.append(pick_place_success)
        percentage_advancement.append(pick_place_success / n_ops)

        if goal_reached or pick_place_success / n_ops == 1.0:
            episode_successes += 1
            print("Episode succeeded.")

        ep = i + 1
        print(f"Success rate: {episode_successes / ep:.3f}")
        print(f"Successful pick_place: {pick_place_successes}")
        print(f"Mean successful pick_place: {mean(pick_place_successes):.2f}")
        print(f"Mean percentage advancement: {mean(percentage_advancement):.2%}")
        print(f"Pick-place success rate: {valid_pick_place_success / num_valid_pick_place_queries:.3f}\n")

        os.makedirs("results", exist_ok=True)
        with open(f"results/results_neurosym_seed_{args.seed}.txt", "w") as f:
            f.write(f"Success rate: {episode_successes / 100}\n")
            f.write(f"Mean successful pick_place: {mean(pick_place_successes)}\n")
            f.write(f"Mean percentage advancement: {mean(percentage_advancement)}\n")
            f.write(f"Pick-place success rate: {valid_pick_place_success / num_valid_pick_place_queries}\n")

    # wandb.finish()


if __name__ == "__main__":
    main()