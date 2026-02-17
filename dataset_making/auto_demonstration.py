import os, argparse, time, zipfile, pickle, copy
import numpy as np
import robosuite as suite
import robosuite_task_zoo
from datetime import datetime
import gymnasium as gym
# import gym
import cv2
import sys
from PIL import Image
#import gym

from robosuite.wrappers import GymWrapper
from robosuite.wrappers.visualization_wrapper import VisualizationWrapper
from robosuite.utils.detector import HanoiDetector, KitchenDetector, NutAssemblyDetector
from robosuite.wrappers.nutassembly.object_state import AssembleStateWrapper
from robosuite.wrappers.kitchen.object_state import KitchenStateWrapper
from robosuite.wrappers.hanoi.object_state import HanoiStateWrapper
from robosuite.wrappers.nutassembly.vision import AssembleVisionWrapper
from robosuite.wrappers.kitchen.vision import KitchenVisionWrapper
from robosuite.wrappers.hanoi.vision import HanoiVisionWrapper

from planning.planner import *
from planning.executor import *

object_state_wrapper = {"Hanoi": HanoiStateWrapper, "KitchenEnv": KitchenStateWrapper, "NutAssembly": AssembleStateWrapper}
vision_wrapper = {"Hanoi": HanoiVisionWrapper, "KitchenEnv": KitchenVisionWrapper, "NutAssembly": AssembleVisionWrapper}

def to_datestring(unixtime: int, format='%Y-%m-%d_%H:%M:%S'):
	return datetime.utcfromtimestamp(unixtime).strftime(format)

planning_predicates = {"Hanoi": ['on', 'clear', 'grasped'],
                          "KitchenEnv": ['on', 'clear', 'grasped', 'stove_on'],
                          "NutAssembly": ['on', 'clear', 'grasped']}  
planning_mode = {"Hanoi": 0,
                 "KitchenEnv": 1,
                 "NutAssembly": 0}

class RecordDemos(gym.Wrapper):
    def __init__(self, 
                 env,
                 vision_based,
                 detector,
                 pddl_path,
                 args, 
                 render=False,
                 randomize=True,
                 noise_std_factor=0.5):
        # Run super method
        super().__init__(env=env)
        
        # Set args
        self.env = env
        self.vision_based = vision_based
        self.detector = detector
        self.pddl_path = pddl_path
        self.args = args
        self.render = render
        self.randomize = randomize
        self.noise_std_factor = noise_std_factor

        # Set up the environment
        self.gripper_body = self.env.sim.model.body_name2id('gripper0_eef')
        self.recorded_eps = 0

        # Init buffer
        self.data_buffer = dict()
        self.action_steps = []
        self.checkpoint = None
        if args.checkpoints > 0:
            self.checkpoint = 0

        # Detect init state
        self.reset()

    def get_plan(self):
        """
        Returns the plan
        """
        # Detect init state
        state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        # Filter and keep only the predicates that are in planning_predicates[args.env] and are True and map them to the PDDL format
        init_predicates = {predicate: True for predicate in state.keys() if state[predicate] and predicate.split("(")[0] in planning_predicates[self.args.env]}
        print("Initial predicates: ", init_predicates)
        # Usage

        add_predicates_to_pddl(pddl_path, init_predicates)
        # Generate a plan
        self.plan, _ = call_planner(pddl_path, mode=planning_mode[self.args.env])
        print("Task demonstrated: ", self.plan)

    def get_task(self):
        """
        Returns the task
        """
        return self.plan[self.operator_step]

    def operator_to_function(self, operation):
        """
        A function that maps the operation to the corresponding function
        returns the function, the semantic description of the operation and the goal
        """
        map_color = {"cube1": "blue", "cube2": "red", "cube3": "green",}
        operation = operation.lower().split(' ')
        print("Operation: ", operation[0])
        if not self.args.vla:
            if len(operation) == 3:
                self.env.set_task((operation[1], operation[2]))
            elif 'turn' in operation[0]:
                self.env.set_task((None, None))
        if 'pick' in operation[0]:
            if self.args.env == "Hanoi":
                return self.pick, f'pick {operation[1]} from {operation[2]}', operation[1]
            else:
                return self.pick, f'pick {operation[1]} from {operation[2]}', operation[1]
        elif 'place' in operation[0]:
            if self.args.env == "Hanoi":
                return self.place, f'place {operation[1]} on {operation[2]}', (operation[1], operation[2])
            else:
                return self.place, f'place {operation[1]} on {operation[2]}', (operation[1], operation[2])
        elif 'turn-on' in operation[0]:
            return self.turn_on_button, f'switch on button', "button"
        elif 'turn-off' in operation[0]:
            return self.turn_off_button, f'switch off button', "button"
        else:
            return None

    def reset(self, seed=None):
        """
        The reset function that resets the environment
        """
        if self.args.checkpoints > 0:
            if self.recorded_eps // self.args.checkpoints > self.checkpoint:
                print("\nSAVING CHECKPOINT: ", self.checkpoint)
                self.data_buffer = dict()
                self.action_steps = []
                self.checkpoint += 1
        self.operator_step = 0
        self.episode_buffer = dict() # 1 episode here consists of a trajectory between 2 symbolic nodes
        self.task_buffer = list()
        try:
            obs, _ = self.env.reset()
        except:
            obs = self.env.env.reset()
        self.env.sim.forward()
        self.get_plan()
        return obs

    def run_trajectory(self, obs):
        """
        Runs the trajectory
        """
        done = False
        for operation in self.plan:
            #try:
            function, self.task, goal = self.operator_to_function(operation)
            #except:
            #    continue
            print(f'Performing task: {self.task}, with goal: {goal}')
            done, obs = function(obs, goal)
            if not(done):
                print("Failed to perform task")
                return False
            print("Successful task")
        print("Successful episode?: ", done)
        return done

    def save_trajectory(self):
        """
        Saves the trajectory to the buffer
        """
        if self.args.vla:
            episode = []
            for step in self.action_steps:
                if step in self.episode_buffer.keys():
                    for i in range(0,len(self.episode_buffer[step]),2): 
                        # Convert action from dx,dy,dz,gripper to dx,dy,dz,d_roll,d_pitch,d_yaw,gripper
                        action_4dim = self.episode_buffer[step][i]
                        action = np.concatenate((action_4dim[:3], np.zeros(3), action_4dim[3:]))

                        raw_image = self.episode_buffer[step][i+1]
                        image = np.array(raw_image).reshape((128, 128, 3))                        
                        episode.append({
                            'image': image,
                            # 'wrist_image': np.asarray(np.random.rand(64, 64, 3) * 255, dtype=np.uint8),
                            # 'state': np.asarray(np.random.rand(10), dtype=np.float32),
                            'action': action,
                            'language_instruction': step,
                        })
            np.save(f'data/kitchen_env/episode_{self.recorded_eps}.npy', episode)
            np.save(f'data/hanoi_dataset/data/episode_{num_recorded_eps}.npy', episode)
        
        else:
            for step in self.action_steps:
                print(step)
                print("\n\n\n")
                if step in self.episode_buffer.keys():
                    if step not in self.data_buffer.keys():
                        self.data_buffer[step] = [(self.episode_buffer[step], self.task_buffer)]
                    else:
                        self.data_buffer[step].append((self.episode_buffer[step], self.task_buffer))
            self.zip_buffer(self.args.traces)
        self.recorded_eps += 1
        obs = self.reset()
        return obs

    def zip_buffer(self, dir_path):
        # Decompose the data buffer into action steps
        for step in self.action_steps:
            if step not in self.data_buffer.keys():
                continue
            # Convert the data buffer to bytes
            data_bytes = pickle.dumps(self.data_buffer[step])
            if self.args.checkpoints > 0:
                file_path = dir_path + str(self.checkpoint) + '/' + step + '.zip'
                os.makedirs(dir_path + str(self.checkpoint), exist_ok=True)
            else:
                file_path = dir_path + step + '.zip'
            # Write the bytes to a zip file
            with zipfile.ZipFile(file_path, 'w') as zip_file:
                with zip_file.open('data.pkl', 'w', force_zip64=True) as file:
                    file.write(data_bytes)

    def record_step(self, obs, action, next_obs, state_memory, new_state, sym_action="MOVE", action_step="main", reward=-1.0, done=False, info=None, goal=None):
        """
        Records the step
        """
        if self.args.render and self.vision_based:
            # display the image (obs)
            cv2.imshow('image', obs)
            cv2.waitKey(1)
        if self.vision_based or self.args.vla:
            # convert obs type to np.uint8
            obs = np.array(obs, dtype=np.uint8)
            next_obs = np.array(next_obs, dtype=np.uint8)
        if obs.shape != next_obs.shape:
            return state_memory
        if self.args.vla:
            action_step = self.task
        if action_step not in self.action_steps:
            self.action_steps.append(action_step)
        if action_step not in self.episode_buffer.keys():
            # self.episode_buffer[action_step] = [obs, action, next_obs] # Why 3 things?
            self.episode_buffer[action_step] = [action, next_obs]
        else:
            self.episode_buffer[action_step] += [action, next_obs]
        return state_memory
        # else:
        #     keypoint = self.relative_obs_mapping(goal)
        #     transition = (obs, action, next_obs, keypoint, reward, done)
        #     if action_step not in self.action_steps:
        #         self.action_steps.append(action_step)
        #     if action_step not in self.episode_buffer.keys():
        #         self.episode_buffer[action_step] = [transition]
        #     else:
        #         self.episode_buffer[action_step].append(transition)
        #     self.task_buffer.append(self.task)
        #     return state_memory

    def cap(self, eps, max_val=0.12, min_val=0.01):
        """
        Caps the displacement
        """
        # If the displacement is greater than the max value, cap it
        if np.linalg.norm(eps) > max_val:
            eps = eps / np.linalg.norm(eps) * max_val
        # If the displacement is smaller than the min value, cap it
        if np.linalg.norm(eps) < min_val:
            eps = eps / np.linalg.norm(eps) * min_val
        return eps

    def to_osc_pose(self, action):
        """
        Converts the action to the OSC pose
        """
        # Add [0, 0, 0] to the action to make it a 6D action and the gripper aperture at the end
        # Insert [0, 0, 0] from action[3] to action[5]
        action = np.insert(action, 3, [0, 0, 0])
        return action

    def relative_obs_mapping(self, goal):
        """
        Get the relative observation between the gripper and the goal + the gripper aperture
        """
        goal_str = goal
        goal = self.env.sim.model.body_name2id(self.detector.object_id[goal_str])
        gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
        object_pos = np.asarray(self.env.sim.data.body_xpos[goal])
        dist = np.linalg.norm(object_pos - gripper_pos)

        gripper_quat = self.env.sim.data.body_xquat[self.gripper_body]
        object_quat = self.env.sim.data.body_xquat[goal]
        angle = self.quaternion_to_euler(gripper_quat)[2] - self.quaternion_to_euler(object_quat)[2]

        left_finger_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_left_inner_finger")])
        right_finger_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_right_inner_finger")])
        aperture = np.linalg.norm(left_finger_pos - right_finger_pos)

        return np.array([dist, angle, aperture])
        
    def quaternion_to_euler(self, quat):
        """
        Converts a quaternion to euler angles
        """
        x = quat[0]
        y = quat[1]
        z = quat[2]
        w = quat[3]
        # roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        # pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(sinp)
        # yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return np.array([roll, pitch, yaw])

    def pick(self, obs, goal):
        """
        Transitons the environment to a state where the gripper is has picked the object
        """
        state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        self.state_memory = state
        #print("\n\n\t\t---------------------PICK GOAL IS: ", goal)
        #print("\n\n")
        reset_step_count = 0

        goal_str = goal
        goal = self.env.sim.model.body_name2id(self.detector.object_id[goal_str])

        # Moving gripper 10 cm above the object
        ref_z = 1.1
        z_pos = 0
        while z_pos < ref_z:
            z_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])[2]
            dist_z = abs(ref_z - z_pos)
            dist_z = self.cap([dist_z])
            action = 5*np.concatenate([[0, 0], dist_z, [0]]) if not(self.randomize) else 5*np.concatenate([[0, 0], dist_z, [0]]) + np.concatenate([[0, 0], np.random.normal(0, self.noise_std_factor*np.linalg.norm(dist_z), 1), [0]])
            action = self.to_osc_pose(action)
            next_obs, _, _, _, _  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="pick", goal=goal_str)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 500:
                return False, obs
            dist_z = dist_z[0]
        reset_step_count = 0
        self.env.time_step = 0

        #print("Moving gripper over object...")
        gripper_goal = "pot_handle" if goal_str == "pot" else goal_str
    
        while not state['over(gripper,{})'.format(gripper_goal)]:
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
            if goal_str == 'pot':
                object_pos = np.asarray(self.env.sim.data.body_xpos[goal])+np.array([0, -0.09, 0])
            elif self.args.env == "NutAssembly":
                object_pos = np.asarray(self.env.sim.data.body_xpos[goal])+np.array([0, 0.05, 0])
            else:
                object_pos = np.asarray(self.env.sim.data.body_xpos[goal])
            dist_xy_plan = object_pos[:2] - gripper_pos[:2]
            dist_xy_plan = self.cap(dist_xy_plan)
            action = 5*np.concatenate([dist_xy_plan, [0, 0]]) if not(self.randomize) else 5*np.concatenate([dist_xy_plan, [0, 0]]) + np.concatenate([np.random.normal(0, self.noise_std_factor*np.linalg.norm(dist_xy_plan), 3), [0]])
            action = self.to_osc_pose(action)
            next_obs, _, _, _, _  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="pick", goal=goal_str)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 500:
                return False, obs
        reset_step_count = 0
        self.env.time_step = 0

        #print("Opening gripper...")
        while not state['open_gripper(gripper)']:
            action = np.asarray([0,0,0,-1])
            action = self.to_osc_pose(action)
            next_obs, _, _, _, _  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="pick", goal=goal_str)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 100:
                return False, obs
        reset_step_count = 0
        self.env.time_step = 0

        #print("Moving down gripper to grab level...")
        while not state['at_grab_level(gripper,{})'.format(goal_str)]:
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
            object_pos = np.asarray(self.env.sim.data.body_xpos[goal])
            dist_z_axis = [object_pos[2] - gripper_pos[2]]
            dist_z_axis = self.cap(dist_z_axis)
            action = 5*np.concatenate([[0, 0], dist_z_axis, [0]]) if not(self.randomize) else 5*np.concatenate([[0, 0], dist_z_axis, [0]]) + np.concatenate([[0, 0], np.random.normal(0, self.noise_std_factor*np.linalg.norm(dist_z_axis), 1), [0]])
            action = self.to_osc_pose(action)
            next_obs, _, _, _, _  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="pick", goal=goal_str)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 400:
                return False, obs
        reset_step_count = 0
        self.env.time_step = 0

        #print("Closing gripper...")
        while not state['grasped({})'.format(goal_str)]:
            action = np.asarray([0,0,0,1])
            action = self.to_osc_pose(action)
            next_obs, _, _, _, _  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="pick", goal=goal_str)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 30:
                return False, obs
        reset_step_count = 0
        self.env.time_step = 0

        #print("Lifting object...")
        while not state['picked_up({})'.format(goal_str)]:
            action = np.asarray([0,0,0.4,0]) if not(self.randomize) else [0,0,0.5,0] + np.concatenate([np.random.normal(0, 0.1, 3), [0]])
            action = 5*self.cap(action)
            action = self.to_osc_pose(action)
            next_obs, _, _, _, _  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="place", goal=goal_str)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 300:
                return False, obs
        reset_step_count = 0
        self.env.time_step = 0

        return True, obs

    def place(self, obs, goal):
        """
        Transitons the environment to a state where the gripper is object is placed at the place to drop
        """
        state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        self.state_memory = state
        #print("\n\n\t\t---------------------PICK GOAL IS: ", goal)
        #print("\n\n")
        reset_step_count = 0

        goal_str = goal[1]
        pick_str = goal[0]
        goal = self.env.sim.model.body_name2id(self.detector.object_id[goal_str])
        pick_body = self.env.sim.model.body_name2id(self.detector.object_id[pick_str])

        goal_pos = self.env.sim.data.body_xpos[goal][:3]
        if 'peg' in goal_str and self.args.env == "Hanoi":
            goal_pos = self.env.sim.data.body_xpos[goal][:3] - np.array([0.1, 0.04, 0])
        else:
            goal_pos = self.env.sim.data.body_xpos[goal][:3]
        goal_quat = self.env.sim.data.body_xquat[goal]
        self.keypoint = np.concatenate([goal_pos, goal_quat])

        print("Moving gripper over place to drop...")
        while not state['over(gripper,{})'.format(goal_str)]:
            #distance = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=True)['over(gripper,stove)']
            #print("Distance: ", distance)
            if pick_str == 'pot':
                gripper_pos = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("PotObject_root")]
            elif self.args.env == "NutAssembly":
                if goal_str == "roundpeg":
                    gripper_pos = np.asarray(self.env.sim.data.body_xpos[pick_body]) +np.array([0, -0.01, 0]) #+np.array([0, -0.05, 0])
                elif goal_str == "squarepeg":
                    gripper_pos = np.asarray(self.env.sim.data.body_xpos[pick_body]) +np.array([+0.015, +0.01, 0])#+ np.array([0.01, -0.025, 0])
            else:
                gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
            if 'peg' in goal_str and self.args.env == "Hanoi":
                object_pos = self.detector.area_pos[goal_str]
            elif "serving" in goal_str:
                object_pos = np.asarray(self.env.sim.data.body_xpos[goal])+np.array([0.002, -0.05, 0])
            # elif self.args.env == "NutAssembly" and goal_str == "roundpeg":
            #     object_pos = np.asarray(self.env.sim.data.body_xpos[goal])+np.array([0, 0.05, 0])
            # elif self.args.env == "NutAssembly" and goal_str == "squarepeg":
            #     object_pos = np.asarray(self.env.sim.data.body_xpos[goal])+ np.array([-0.01, 0.025, 0])
            #elif "stove" in goal_str:
            #    object_pos = np.asarray(self.env.sim.data.body_xpos[goal])+np.array([0.01, 0, 0])
            #elif "pot" in goal_str:
            #    object_pos = np.asarray(self.env.sim.data.body_xpos[goal])+np.array([0, -0.01, 0])
            else:
                object_pos = np.asarray(self.env.sim.data.body_xpos[goal])
            dist_xy_plan = object_pos[:2] - gripper_pos[:2] #+ 0.05
            dist_xy_plan = self.cap(dist_xy_plan)
            action = 5*np.concatenate([dist_xy_plan, [0, 0]]) if not(self.randomize) else 5*np.concatenate([dist_xy_plan, [0, 0]]) + np.concatenate([np.random.normal(0, self.noise_std_factor*np.linalg.norm(dist_xy_plan), 3), [0]])
            action = self.to_osc_pose(action)
            next_obs, _, _, _, _  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="reach_place", goal=goal_str)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 500:
                return False, obs
        reset_step_count = 0
        self.env.time_step = 0

        print("Moving down picked object on place to drop...")
        while not state['on({},{})'.format(pick_str, goal_str)]:
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
            #object_pos = np.asarray(self.env.sim.data.body_xpos[goal])
            place_pos = np.asarray(self.env.sim.data.body_xpos[goal])
            dist_z_axis = [- (gripper_pos[2] - place_pos[2])]
            dist_z_axis = self.cap(dist_z_axis)
            action = 5*np.concatenate([[0, 0], dist_z_axis, [0]]) if not(self.randomize) else 5*np.concatenate([[0, 0], dist_z_axis, [0]]) + np.concatenate([[0, 0], np.random.normal(0, self.noise_std_factor*np.linalg.norm(dist_z_axis), 1), [0]])
            action = self.to_osc_pose(action)
            next_obs, _, _, _, _  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="place", goal=goal_str)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 200:
                return False, obs
        reset_step_count = 0
        self.env.time_step = 0

        print("dropping object...")
        while not(state['open_gripper(gripper)']):#state['grasped({})'.format(goal_str)]:
            action = np.asarray([0,0,0,-1])
            action = self.to_osc_pose(action)
            next_obs, _, _, _, _  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="place", goal=goal_str)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 30:
                return False, obs
        reset_step_count = 0
        self.env.time_step = 0

        print("Resetting gripper")
        # First move up
        for _ in range(15):
            action = np.array([0, 0, 1, 0])
            action = self.to_osc_pose(action)
            next_obs, _, _, _, _  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="place", goal=goal_str)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1

        return True, obs
    
    def turn_on_button(self, obs, goal):
        """
        Transitons the environment to a state where button is turned on
        """
        state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        self.state_memory = state
        #print("\n\n\t\t---------------------PICK GOAL IS: ", goal)
        #print("\n\n")

        goal_str = goal
        goal = self.env.sim.model.body_name2id(self.detector.object_id[goal_str])

        goal_pos = self.env.sim.data.body_xpos[goal][:3]
        goal_quat = self.env.sim.data.body_xquat[goal]
        self.keypoint = np.concatenate([goal_pos, goal_quat])

        reset_step_count = 0

        #print("Moving gripper over button...")
        while not state['over(gripper,{})'.format(goal_str)]:
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
            object_pos = np.asarray(self.env.sim.data.body_xpos[goal])
            dist_xy_plan = object_pos[:2] - gripper_pos[:2]
            dist_xy_plan = self.cap(dist_xy_plan)
            action = 5*np.concatenate([dist_xy_plan, [0, 0]]) if not(self.randomize) else 5*np.concatenate([dist_xy_plan, [0, 0]]) + np.concatenate([np.random.normal(0, self.noise_std_factor*np.linalg.norm(dist_xy_plan), 3), [0]])
            action = self.to_osc_pose(action)
            next_obs, _, _, _, _  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="turn_on", goal=goal_str)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 500:
                return False, obs
        reset_step_count = 0
        self.env.time_step = 0

        #print("Moving down gripper to swith level ...")
        while not state['at_grab_level(gripper,{})'.format(goal_str)]:
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
            object_pos = np.asarray(self.env.sim.data.body_xpos[goal])
            dist_z_axis = [object_pos[2] - gripper_pos[2]]
            dist_z_axis = self.cap(dist_z_axis)
            action = 5*np.concatenate([[0, 0], dist_z_axis, [0]]) if not(self.randomize) else 5*np.concatenate([[0, 0], dist_z_axis, [0]]) + np.concatenate([[0, 0], np.random.normal(0, self.noise_std_factor*np.linalg.norm(dist_z_axis), 1), [0]])
            action = self.to_osc_pose(action)
            next_obs, _, _, _, _  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="turn_on", goal=goal_str)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 400:
                return False, obs
        reset_step_count = 0
        self.env.time_step = 0

        #print("Turning on button...")
        while not state['stove_on()']:
            action = np.asarray([0,0.3,0,0])
            action = self.to_osc_pose(action)
            next_obs, _, _, _, _  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="turn_on", goal=goal_str)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 50:
                return False, obs
        reset_step_count = 0
        self.env.time_step = 0

        for _ in range(15):
            action = np.array([0, 0, 1, 0])
            action = self.to_osc_pose(action)
            next_obs, _, _, _, _  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="turn_on", goal=goal_str)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1

        return True, obs
    
    def turn_off_button(self, obs, goal):
        """
        Transitons the environment to a state where button is turned off
        """
        state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        self.state_memory = state
        #print("\n\n\t\t---------------------PICK GOAL IS: ", goal)
        #print("\n\n")

        goal_str = goal
        goal = self.env.sim.model.body_name2id(self.detector.object_id[goal_str])

        goal_pos = self.env.sim.data.body_xpos[goal][:3]
        goal_quat = self.env.sim.data.body_xquat[goal]
        self.keypoint = np.concatenate([goal_pos, goal_quat])

        reset_step_count = 0

        #print("Moving gripper over button...")
        while not state['over(gripper,{})'.format(goal_str)]:
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
            object_pos = np.asarray(self.env.sim.data.body_xpos[goal])
            dist_xy_plan = object_pos[:2] - gripper_pos[:2]
            dist_xy_plan = self.cap(dist_xy_plan)
            action = 5*np.concatenate([dist_xy_plan, [0, 0]]) if not(self.randomize) else 5*np.concatenate([dist_xy_plan, [0, 0]]) + np.concatenate([np.random.normal(0, self.noise_std_factor*np.linalg.norm(dist_xy_plan), 3), [0]])
            action = self.to_osc_pose(action)
            next_obs, _, _, _, _  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="turn_off", goal=goal_str)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 500:
                return False, obs
        reset_step_count = 0
        self.env.time_step = 0

        #print("Moving down gripper to swith level ...")
        while not state['at_grab_level(gripper,{})'.format(goal_str)]:
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
            object_pos = np.asarray(self.env.sim.data.body_xpos[goal])
            dist_z_axis = [object_pos[2] - gripper_pos[2]]
            dist_z_axis = self.cap(dist_z_axis)
            action = 5*np.concatenate([[0, 0], dist_z_axis, [0]]) if not(self.randomize) else 5*np.concatenate([[0, 0], dist_z_axis, [0]]) + np.concatenate([[0, 0], np.random.normal(0, self.noise_std_factor*np.linalg.norm(dist_z_axis), 1), [0]])
            action = self.to_osc_pose(action)
            next_obs, _, _, _, _  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="turn_off", goal=goal_str)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 400:
                return False, obs
        reset_step_count = 0
        self.env.time_step = 0

        #print("Turning off button...")
        while state['stove_on()']:
            action = np.asarray([0,-0.3,0,0])
            action = self.to_osc_pose(action)
            next_obs, _, _, _, _  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="turn_off", goal=goal_str)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 50:
                return False, obs
        reset_step_count = 0
        self.env.time_step = 0

        for _ in range(15):
            action = np.array([0, 0, 1, 0])
            action = self.to_osc_pose(action)
            next_obs, _, _, _, _  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="turn_off", goal=goal_str)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1

        return True, obs

if __name__ == "__main__":
    # Define the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Hanoi', choices=['Hanoi', 'KitchenEnv', 'NutAssembly'], help='Name of the environment to run the experiment in')
    parser.add_argument('--dir', type=str, default='./data/', help='Path to the data folder')
    parser.add_argument('--episodes', type=int, default=int(200), help='Number of trajectories to record')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--name', type=str, default=None, help='Name of the experiment')
    parser.add_argument('--render', action='store_true', help='Render the recording')
    parser.add_argument('--vision', action='store_true', help='Use vision based observation')
    parser.add_argument('--relative_obs', action='store_true', help='Use relative gripper-goal observation')
    parser.add_argument('--vla', action='store_true', help='Store the data in VLA friendly format')
    parser.add_argument('--size', type=int, default=224, help='Size of the observation')
    parser.add_argument('--checkpoints', type=int, default=0, help='Saves the data every n episodes, and resets the buffer')
    parser.add_argument('--rnd_reset', action='store_true', help='Random reset for Hanoi env')

    args = parser.parse_args()
    # Set the random seed
    np.random.seed(args.seed)
    # Define the directories
    dir = args.dir
    experiment_name = args.env + '_seed_' + str(args.seed)
    experiment_id = f"{to_datestring(time.time())}"#self.hashid 
    if args.name is not None:
        experiment_id = args.name
    args.env_dir = os.path.join(dir, experiment_name, experiment_id)

    print("Starting experiment {}.".format(os.path.join(experiment_name, experiment_id)))

    # Create the directories
    args.traces = args.env_dir + '/traces/'
    os.makedirs(args.env_dir, exist_ok=True)
    os.makedirs(args.traces, exist_ok=True)

    # Load the controller config
    controller_config = suite.load_controller_config(default_controller='OSC_POSE')
    # Create the environment
    if args.env == 'Hanoi':
        env = suite.make(
            args.env,
            robots="Kinova3",
            controller_configs=controller_config,
            has_renderer=args.render,
            has_offscreen_renderer=True,
            horizon=100000000,
            use_camera_obs=args.vision,
            use_object_obs=not(args.vision),
            camera_heights=args.size,
            camera_widths=args.size,
            random_reset = args.rnd_reset
        )
    else:
        env = suite.make(
            args.env,
            robots="Kinova3",
            controller_configs=controller_config,
            has_renderer=args.render,
            has_offscreen_renderer=True,
            horizon=100000000,
            use_camera_obs=args.vision,
            use_object_obs=not(args.vision),
            camera_heights=args.size,
            camera_widths=args.size,
        )

    env.reset()

    # Wrap the environment
    if args.env == 'Hanoi':
        detector = HanoiDetector(env)
        pddl_path = './planning/PDDL/hanoi/'
    elif args.env == 'KitchenEnv':
        detector = KitchenDetector(env)
        pddl_path = './planning/PDDL/kitchen/'
    elif args.env == 'NutAssembly':
        detector = NutAssemblyDetector(env)
        pddl_path = './planning/PDDL/nut_assembly/'
    env = GymWrapper(env, proprio_obs=not(args.vision))
    if args.vla:
        print("Using VLA friendly format")
        env = VisualizationWrapper(env)
    elif not args.vision:
        print("Using object based observation")
        env = object_state_wrapper[args.env](env)
    else:
        print("Using vision based observation")
        env = VisualizationWrapper(env)
        env = vision_wrapper[args.env](env, image_size=args.size)

    env = RecordDemos(env, args.vision, detector, pddl_path, args, render=args.render, randomize=True)

    os.makedirs(f'data/hanoi_dataset/data', exist_ok=True)

    # Run the recording of the demonstrations
    episode = 1
    while env.recorded_eps < args.episodes: 
        obs = env.reset()
        print("Episode: {}".format(episode))
        keys = list(env.data_buffer.keys())
        done = env.run_trajectory(obs)
        if done:
            obs = env.save_trajectory()
        episode += 1
        print("Number of recorded episodes: {}".format(env.recorded_eps))
        print("\n\n")