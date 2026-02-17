import warnings
warnings.filterwarnings("ignore")


import argparse
import gym
import joblib
import robosuite as suite
import numpy as np
from statistics import mean 
from robosuite.wrappers import GymWrapper
from robosuite.utils.detector import HanoiDetector, KitchenDetector, NutAssemblyDetector, CubeSortingDetector, HeightStackingDetector, AssemblyLineSortingDetector, PatternReplicationDetector
from planning.planner import *
from planning.executor import *
from ultralytics import YOLO


env_detectors = {
    "Hanoi": HanoiDetector,
    "KitchenEnv": KitchenDetector,
    "NutAssembly": NutAssemblyDetector,
    "CubeSorting": CubeSortingDetector,
    "HeightStacking": HeightStackingDetector,
    "AssemblyLineSorting": AssemblyLineSortingDetector,
    "PatternReplication": PatternReplicationDetector
}
planning_predicates = {
    "Hanoi": ['on', 'clear', 'grasped'],
    "KitchenEnv": ['on', 'clear', 'grasped', 'stove_on'],
    "NutAssembly": ['on', 'clear', 'grasped'],
    "CubeSorting": ['on', 'clear', 'grasped', 'type_match'],
    "HeightStacking": ['on', 'clear', 'grasped', 'smaller'],
    "AssemblyLineSorting": ['on', 'clear', 'grasped', 'type_match'],
    "PatternReplication": ['on', 'clear', 'grasped']}
planning_mode = {
    "Hanoi": 0,
    "KitchenEnv": 1,
    "NutAssembly": 0,
    "CubeSorting": 0,
    "HeightStacking": 0,
    "AssemblyLineSorting": 0,
    "PatternReplication": 0}
env_detectors = {
    "Hanoi": HanoiDetector,
    "KitchenEnv": KitchenDetector,
    "NutAssembly": NutAssemblyDetector,
    "CubeSorting": CubeSortingDetector,
    "HeightStacking": HeightStackingDetector,
    "AssemblyLineSorting": AssemblyLineSortingDetector,
    "PatternReplication": PatternReplicationDetector
}
pddl_paths = {
    "Hanoi": "planning/PDDL/hanoi/",
    "KitchenEnv": "planning/PDDL/kitchen.pddl",
    "NutAssembly": "planning/PDDL/nut_assembly.pddl",
    "CubeSorting": "planning/PDDL/cubesorting/",
    "HeightStacking": "planning/PDDL/heightstacking/",
    "AssemblyLineSorting": "planning/PDDL/assemblyline/",
    "PatternReplication": "planning/PDDL/patternreplication/"
}
yolo_model_paths = {
    "Hanoi": "models/yolo/hanoi_yolo.pt",
    "KitchenEnv": "models/yolo/kitchen_yolo.pt",
    "NutAssembly": "models/yolo/nut_assembly_yolo.pt",
    "CubeSorting": "models/yolo/hanoi_yolo.pt",
    "HeightStacking": "models/yolo/hanoi_yolo.pt",
    "AssemblyLineSorting": "models/yolo/hanoi_yolo.pt",
    "PatternReplication": "models/yolo/hanoi_yolo.pt"
}
regressor_model_paths = {
    "Hanoi": "models/regressors/hanoi_regressor.pkl",
    "KitchenEnv": "models/regressors/kitchen_regressor.pkl",
    "NutAssembly": "models/regressors/nut_assembly_regressor.pkl",
    "CubeSorting": "models/regressors/hanoi_regressor.pkl",
    "HeightStacking": "models/regressors/hanoi_regressor.pkl",
    "AssemblyLineSorting": "models/regressors/hanoi_regressor.pkl",
    "PatternReplication": "models/regressors/hanoi_regressor.pkl"
}
env_pddl_mapping = {
    "Hanoi": {'cube1': 'o1', 'cube2': 'o2', 'cube3': 'o3', 'peg1': 'o4', 'peg2': 'o5', 'peg3': 'o6'},
    "KitchenEnv": {'bread': 'o1', 'pot': 'o2', 'stove': 'o3', 'serving': 'o4', 'table': 'o5'},
    "NutAssembly": {'nut1': 'o1', 'nut2': 'o2', 'bolt1': 'o3', 'bolt2': 'o4', 'plate': 'o5'},
    "CubeSorting": {'cube1': 'o1', 'cube2': 'o2', 'cube3': 'o3', 'peg1': 'o4', 'peg2': 'o5', 'peg3': 'o6'},
    "HeightStacking": {'cube1': 'o1', 'cube2': 'o2', 'cube3': 'o3', 'peg1': 'o4', 'peg2': 'o5', 'peg3': 'o6'},
    "AssemblyLineSorting": {'cube1': 'o1', 'cube2': 'o2', 'cube3': 'o3', 'peg1': 'o4', 'peg2': 'o5', 'peg3': 'o6'},
    "PatternReplication": {'cube1': 'o1', 'cube2': 'o2', 'cube3': 'o3', 'peg1': 'o4', 'peg2': 'o5', 'peg3': 'o6'}
}
policies_paths = {
    "Hanoi": {"grasp": "policies/hanoi/grasp.ckpt",
              "drop": "policies/hanoi/drop.ckpt",
              "reach_pick": "policies/hanoi/reach_pick.ckpt",
              "reach_place": "policies/hanoi/reach_drop.ckpt"},
    "CubeSorting": {"grasp": "policies/hanoi/grasp.ckpt",
                     "drop": "policies/hanoi/drop.ckpt",
                     "reach_pick": "policies/hanoi/reach_pick.ckpt",
                     "reach_place": "policies/hanoi/reach_drop.ckpt"},
    "HeightStacking": {"grasp": "policies/hanoi/grasp.ckpt",
                        "drop": "policies/hanoi/drop.ckpt",
                        "reach_pick": "policies/hanoi/reach_pick.ckpt",
                        "reach_place": "policies/hanoi/reach_drop.ckpt"},
    "AssemblyLineSorting": {"grasp": "policies/hanoi/grasp.ckpt",
                             "drop": "policies/hanoi/drop.ckpt",
                             "reach_pick": "policies/hanoi/reach_pick.ckpt",
                             "reach_place": "policies/hanoi/reach_drop.ckpt"},
    "PatternReplication": {"grasp": "policies/hanoi/grasp.ckpt",
                            "drop": "policies/hanoi/drop.ckpt",
                            "reach_pick": "policies/hanoi/reach_pick.ckpt",
                            "reach_place": "policies/hanoi/reach_drop.ckpt"},
    "NutAssembly": {"grasp": "policies/nut_assembly/grasp.ckpt",
                      "drop": "policies/nut_assembly/drop.ckpt",
                      "reach_pick": "policies/nut_assembly/reach_pick.ckpt",
                      "reach_place": "policies/nut_assembly/reach_drop.ckpt"},
    "KitchenEnv": {"grasp": "policies/kitchen/grasp.ckpt",
                    "drop": "policies/kitchen/drop.ckpt",
                    "reach_pick": "policies/kitchen/reach_pick.ckpt",
                    "reach_place": "policies/kitchen/reach_drop.ckpt"}}

def get_plan(state, pddl_path, mode):
    # Generate the plan
    init_predicates = {predicate: True for predicate in state.keys() if state[predicate] and predicate.split("(")[0] in planning_predicates[args.env]}
    # Remove all predicates regarding reference objects from the PDDL file
    copy_init_predicates = init_predicates.copy()
    for predicate in copy_init_predicates.keys():
        if "ref" in predicate:
            init_predicates.pop(predicate)
    add_predicates_to_pddl(pddl_path, init_predicates)
    # Get goal from initial state
    # For Hanoi, KitchenEnv and NutAssembly, do nothing
    # For CubeSorting, find all small cubes and write the goal as on(cube, target_zone)
    if args.env == "CubeSorting":
        goal_predicates = []
        for predicate in state.keys():
            if "small" in predicate and state[predicate]:
                objs = predicate[predicate.find("(")+1:predicate.find(")")].split(", ")
                goal_predicates.append(f'on {objs[0]} platform1')
            elif "small" in predicate and not state[predicate]:
                objs = predicate[predicate.find("(")+1:predicate.find(")")].split(", ")
                goal_predicates.append(f'on {objs[0]} platform2')
        goal_str = "\n".join(goal_predicates)
        print("Goal predicates: ", goal_str)
    elif args.env == "HeightStacking":
        goal_predicates = []
        sizes = {}
        for predicate in state.keys():
            if "smaller" in predicate and state[predicate]:
                objs = predicate[predicate.find("(")+1:predicate.find(")")].split(",")
                sizes[objs[0]] = objs[1]
        # Create stacking order based on sizes
        sorted_sizes = sorted(sizes.items(), key=lambda x: x[1])
        for i in range(len(sorted_sizes)-1):
            goal_predicates.append(f'on {sorted_sizes[i][0]} {sorted_sizes[i+1][0]}')
        goal_str = "\n".join(goal_predicates)
        # Add largest cube on platform
        goal_predicates.append(f'on {sorted_sizes[-1][0]} platform')
        print("Goal predicates: ", goal_str)
    elif args.env == "AssemblyLineSorting":
        goal_predicates = []
        types = {}
        for predicate in state.keys():
            if "type_match" in predicate and state[predicate]:
                objs = predicate[predicate.find("(")+1:predicate.find(")")].split(",")
                types[objs[0]] = objs[1]
        for obj, type_ in types.items():
            goal_predicates.append(f'on {obj} {type_}')
        goal_str = "\n".join(goal_predicates)
        print("Goal predicates: ", goal_str)
    elif args.env == "PatternReplication":
        goal_predicates = detector.get_pattern_replication_goal()
        goal_str = "\n".join(goal_predicates)
        print("Goal predicates: ", goal_str)
    define_goal_in_pddl(pddl_path, goal_predicates)
    plan, _ = call_planner(pddl_path, mode=mode)
    return plan, goal_predicates

if __name__ == "__main__":

    # Define the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Hanoi', choices=['Hanoi', 'KitchenEnv', 'NutAssembly', 'CubeSorting', 'AssemblyLineSorting', 'HeightStacking', 'PatternReplication'], help='Name of the environment to run the experiment in')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--size', type=int, default=256, help='Size of the rendered images')
    parser.add_argument('--rnd_reset', action='store_true', help='Randomize the object positions at reset')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--n_act', type=int, default=4, help='Number of actions to execute per policy call')
    args = parser.parse_args()
    np.random.seed(args.seed)

    def termination_indicator(operator):
        if operator == 'pick':
            def Beta(state, symgoal):
                condition = state[f"grasped({symgoal[0]})"]
                return condition
        elif operator == 'drop':
            def Beta(state, symgoal):
            # print("State in drop Beta: ", state)
            # print("Symgoal in drop Beta: ", symgoal)
            # print("Checking condition: ", f"on({symgoal[0]},{symgoal[1]})", state[f"on({symgoal[0]},{symgoal[1]})"])
            # print("Checking condition: ", f"grasped({symgoal[0]})", state[f"grasped({symgoal[0]})"])
            # print("Condition value: ", state[f"on({symgoal[0]},{symgoal[1]})"] and not state[f"grasped({symgoal[0]})"])
            # print("-----------------------------------")
                condition = state[f"in({symgoal[0]},{symgoal[1]})"] and not state[f"grasped({symgoal[0]})"]
                return condition
        elif operator == 'reach_pick':
            def Beta(state, symgoal):
                condition = state[f"over(gripper,{symgoal[0]})"]
                return condition
        elif operator == 'reach_drop':
            def Beta(state, symgoal):
                condition = state[f"over(gripper,{symgoal[1]})"]
                return condition
        elif operator == 'turnon':
            def Beta(state, symgoal):
                condition = state[f'stove_on()']
                return condition
        elif operator == 'turnoff':
            def Beta(state, symgoal):
                condition = not(state[f'stove_on()'])
                return condition
        else:
            def Beta(state, symgoal):
                condition = False
                return condition
        return Beta

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

        def render(self, mode='human', *args, **kwargs):
            self.env.render()

        def _get_observations(self):
            return self.env._get_observations()
        
        def __getattr__(self, name):
            return getattr(self.env, name)


    # # Load executors
    pick = Executor_Diffusion(id='Pick', 
                       policy=policies_paths[args.env]['grasp'],
                       Beta=termination_indicator('pick'),
                       nulified_action_indexes=[0, 1],
                       oracle=True,
                       horizon=8/args.n_act*25,
                       debug=args.debug)
    reach_pick = Executor_Diffusion(id='ReachPick', 
                            policy=policies_paths[args.env]['reach_pick'],
                            Beta=termination_indicator('reach_pick'),
                            nulified_action_indexes=[3],
                            oracle=True,
                            horizon=8/args.n_act*25,
                            debug=args.debug)
    reach_place = Executor_Diffusion(id='ReachDrop', 
                            policy=policies_paths[args.env]['reach_place'],
                            Beta=termination_indicator('reach_drop'),
                            nulified_action_indexes=[3],
                            oracle=True,
                            horizon=8/args.n_act*35,
                            debug=args.debug)
    drop = Executor_Diffusion(id='Drop', 
                    policy=policies_paths[args.env]['drop'],
                    Beta=termination_indicator('drop'),
                    nulified_action_indexes=[0, 1],
                    oracle=True,
                    horizon=8/args.n_act*25,
                    debug=args.debug)

    Move_action = [reach_pick, pick, reach_place, drop]

    # Load the controller config
    controller_config = suite.load_controller_config(default_controller='OSC_POSITION')

    # Create the environment
    env = suite.make(
        env_name=args.env,
        robots="Kinova3",
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
        #random_block_placement=args.rnd_reset
    )
    detector = env_detectors[args.env](env)
    # Wrap the environment with the GymWrapper
    env = GymWrapper(env)
    env = DictObs(env)
    pddl_path = pddl_paths[args.env]
    obj_mapping = env_pddl_mapping[args.env]
    yolo_model = YOLO(yolo_model_paths[args.env])
    regressor_model = joblib.load(regressor_model_paths[args.env])
    n_obs = 4

    env.reset()
    state = detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
    print("Initial state: ", state)

    # Create a lambda function that maps "on(cube1,peg1)" to "p1(o1,o3)"
    def map_predicate(predicate):
        # Extract the objects from the predicate
        objects = predicate.split('(')[1].split(')')[0].split(',')
        return f"p1({obj_mapping[objects[0]]},{obj_mapping[objects[1]]})"
    def change_predicate(predicate):
        # Extract the objects from the predicate
        objects = predicate.split('(')[1].split(')')[0].split(',')
        return f"p1({obj_mapping[objects[0]]},{obj_mapping[objects[0]]})"
    # Filter and keep only the predicates that are "on" and are True and map them to the PDDL format
    #init_predicates = {map_predicate(predicate): True for predicate in state.keys() if predicate[:3] == "on(" and state[predicate]}
    # Filter and keep only the predicates that are "clear" and are True and map them to the PDDL format
    #init_predicates.update({change_predicate(predicate): True for predicate in state.keys() if 'clear' in predicate and state[predicate]})
    init_predicates = {predicate: True for predicate in state.keys() if state[predicate] and predicate.split("(")[0] in planning_predicates[args.env]}
    #print("Initial predicates: ", init_predicates)
    # Usage
    # Remove all predicates regarding reference objects from the PDDL file
    copy_init_predicates = init_predicates.copy()
    for predicate in copy_init_predicates.keys():
        if "ref" in predicate:
            init_predicates.pop(predicate)
    print("Initial predicates: ", init_predicates)

    reset_gripper_pos = np.array([-0.080193391, -0.03391656,  0.95828137])
    episode_successes = 0
    num_valid_pick_place_queries = 0
    pick_place_success = 0
    pick_place_successes = []
    percentage_advancement = []
    valid_pick_place_success = 0

    def reset_gripper(env):
        print("Resetting gripper")
        # First open the gripper
        state = detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        while not(state["open_gripper(gripper)"]):
            #print(state["open_gripper(gripper)"])
            action = np.array([0, 0, 0, -1])
            env.step(action)
            state = detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            if args.render:
                env.render()
        # Then move the gripper to the initial position
        for _ in range(50):
            action = np.array([0, 0, 0.5, 0])
            env.step(action)
            if args.render:
                env.render()
        gripper_pos = env._get_observations()["robot0_eef_pos"]
        delta = reset_gripper_pos - gripper_pos
        action = 5*np.array([delta[0], delta[1], delta[2], 0])
        while np.linalg.norm(delta) > 0.01:
            #print("Curent pos: ", gripper_pos)
            #print("Reset pos: ", reset_gripper_pos)
            action = 5*np.array([delta[0], delta[1], delta[2], 0])
            action = action * 0.9
            env.step(action)
            if args.render:
                env.render()
            gripper_pos = env._get_observations()["robot0_eef_pos"]
            delta = reset_gripper_pos - gripper_pos
            #print(f"Delta: {delta}, Current pos: {gripper_pos}, Reset pos: {reset_gripper_pos}, Action: {action}")
    retry_reset = False
    for i in range(100):
        if retry_reset:
            i -= 1
        print("Episode: ", i)
        success = False
        valid_state = False
        plan = False
        goal_reached = False
        np.random.seed(args.seed + i)
        # Reset the environment until a valid state is reached
        while plan == False:
            # Reset the environment
            env.reset()
            if args.render:
                env.render()
            observations = []
            for _ in range(args.n_act):
                env.step(np.zeros(env.action_space.shape))
                obs = env._get_observations()
                objects_pos = detector.get_all_objects_pos()
                obs['objects_pos'] = objects_pos
                observations.append(obs)
            # Get only the last n_obs observations
            observations = observations[-n_obs:]
            state = detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            #print("Initial state: ", state)
            # Generate a plan
            plan, goal_predicates = get_plan(state, pddl_path, mode=planning_mode[args.env])
        print("Plan: ", plan)
        print("Goal predicates: ", goal_predicates)

        pick_place_success = 0
        # Execute the first operator in the plan
        reset_gripper(env)
        tracking_data = {}
        for j, operator in enumerate(plan):
            print("\nExecuting operator: ", operator)
            # Concatenate the observations with the operator effects
            #obj_to_pick = obj_mapping[operator.split(' ')[2].lower()]
            #obj_to_drop = obj_mapping[operator.split(' ')[3].lower()]
            if j%2 == 0:
                obj_to_pick = operator.split(' ')[-2].lower()
                continue
            else:
                obj_to_drop = operator.split(' ')[-1].lower()

            num_valid_pick_place_queries += 1
            if operator.split(' ')[0].lower() == "turnon":
                skill = Turnon_action
                print("Turn On action")
            elif operator.split(' ')[0].lower() == "turnoff":
                skill = Turnoff_action
                print("Turn Off action")
            elif operator.split(' ')[0].lower() == "wait":
                skill = []
                print("Wait action")
            else:
                skill = Move_action
                print("Picking object: {}, Dropping object: {}".format(obj_to_pick, obj_to_drop))
            for action_step in skill:
                action_step.load_policy(detector=detector, 
                                        yolo_model=yolo_model,
                                        regressor_model=regressor_model, 
                                        image_size=args.size)
                if tracking_data:
                    action_step.set_tracking_data(tracking_data)
                print("\tExecuting action: ", action_step.id)
                sub_goal = (obj_to_pick, obj_to_drop)
                task_goals = goal_predicates.copy()
                observations, success, goal_reached = action_step.execute(env, observations, args.n_act, sub_goal, task_goals, args.render)
                tracking_data = action_step.get_tracking_data()

                state = detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            if success:
                pick_place_success += 1
                valid_pick_place_success += 1
                print("+++ Object successfully picked and placed.")
                print(f"Successfull operations: {pick_place_success}, Out of: {len(plan)/2}, Percentage advancement: {pick_place_success/(len(plan)/2)}")
                if operator == plan[-1]:
                    continue
                try:
                    reset_gripper(env)
                except ValueError:
                    retry_reset = True
                    break
            else:
                # Print the number of operators that were successfully executed out of the total number of operators in the plan
                print("--- Object not picked and placed.")
                print(f"Successfull operations: {pick_place_success}, Out of: {len(plan)/2}, Percentage advancement: {pick_place_success/(len(plan)/2)}")
                try:
                    reset_gripper(env)
                except ValueError:
                    retry_reset = True
                    break
                continue
            
        pick_place_successes.append(pick_place_success)
        percentage_advancement.append(pick_place_success/(len(plan)/2))
        if goal_reached or pick_place_success/(len(plan)/2) == 1.0:
            episode_successes += 1
            print("Episode Execution succeeded.\n")
        print("Success rate: ", episode_successes/(i+1))
        print("\n\n")

        pick_place_successes.append(pick_place_success)
        percentage_advancement.append(pick_place_success/(len(plan)/2))

        print("Successfull pick_place: ", pick_place_successes)
        print("Percentage advancement: ", percentage_advancement)
        print("Mean Successful pick_place: ", mean(pick_place_successes))
        print("Mean Percentage advancement: ", mean(percentage_advancement))
        print("Pick placce success rate: ", valid_pick_place_success/num_valid_pick_place_queries)

        print("Success rate: ", episode_successes/(i+1))
        # Write the results to a file results_seed_{args.seed}.txt
        os.makedirs("results", exist_ok=True)
        with open(f"results/results_neurosym_seed_{args.seed}.txt", 'w') as file:
            file.write("Success rate: {}\n".format(episode_successes/(100)))
            file.write("Mean Successful pick_place: {}\n".format(mean(pick_place_successes)))
            file.write("Mean Percentage advancement: {}\n".format(mean(percentage_advancement)))
            file.write("Pick placce success rate: {}\n".format(valid_pick_place_success/num_valid_pick_place_queries))

