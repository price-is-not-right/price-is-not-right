import os, argparse, time, zipfile, pickle, copy
import numpy as np
import robosuite as suite
import robosuite_task_zoo
from datetime import datetime
import gymnasium as gym
import cv2

from robosuite.wrappers import GymWrapper
from robosuite.wrappers.visualization_wrapper import VisualizationWrapper
from robosuite.utils.detector import HanoiDetector, KitchenDetector, NutAssemblyDetector, CubeSortingDetector, HeightStackingDetector, AssemblyLineSortingDetector, PatternReplicationDetector

from planning.planner import *
from planning.executor import *

from ultralytics import YOLO
import joblib
import pandas as pd


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*env.*is deprecated.*")

def to_datestring(unixtime: int, format='%Y-%m-%d_%H:%M:%S'):
	return datetime.utcfromtimestamp(unixtime).strftime(format)

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

yolo_id_mappings = {
    "Hanoi": {"blue cube": "cube1", "red cube": "cube2", "green cube": "cube3"},
    "KitchenEnv": {"pot": "pot", "bread": "bread", "serving bowl": "serving", "stove": "stove", "button": "button"},
    "NutAssembly": {"roundnut": "roundnut", "squarenut": "squarenut", "roundpeg": "roundpeg", "squarepeg": "squarepeg"},
    "CubeSorting": {"green cube": "cube1", "red cube": "cube2", "blue cube": "cube3"},
    "HeightStacking": {"green cube": "cube1", "red cube": "cube2", "blue cube": "cube3"},
    "AssemblyLineSorting": {"green cube": "cube1", "red cube": "cube2", "blue cube": "cube3"},
    "PatternReplication": {"green cube": "cube1", "red cube": "cube2", "blue cube": "cube3"}
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

def to_float(x):
    if torch.is_tensor(x):
        return x.detach().cpu().item()
    return float(x)


class RecordDemos(gym.Wrapper):
    def __init__(self, 
                 env,
                 detector,
                 pddl_path,
                 args, 
                 vision_based = True,
                 yolo_model=None,
                 regressor_model=None,
                 yolo_id_mapping=None,
                 obj_mapping=None,
                 render=False,
                 randomize=True,
                 noise_std_factor=0.1):
        super().__init__(env=env)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.env = env
        self.vision_based = vision_based
        self.detector = detector
        self.pddl_path = pddl_path
        self.args = args
        self.yolo_model = yolo_model
        self.regressor_model = regressor_model
        self.render = render
        self.randomize = randomize
        self.noise_std_factor = noise_std_factor
        self.obj_mapping = obj_mapping

        self.gripper_body = self.env.sim.model.body_name2id('gripper0_eef')
        self.recorded_eps = 0

        self.data_buffer = dict()
        self.action_steps = []
        self.checkpoint = None
        if args.checkpoints > 0:
            self.checkpoint = 0
        self.df_metrics = pd.DataFrame(columns=['wx', 'wy', 'wz', 'pred_x', 'pred_y', 'pred_z', 'error'])
        self.bboxes_centers = []
        self.yolo_id_mapping = yolo_id_mapping
        self.perception = None
        if yolo_model is not None and regressor_model is not None:
            self.perception = Executor_Diffusion(
                id="demo_perception",
                policy=None,
                Beta=lambda *a, **k: False,
                use_yolo=True,
                oracle=True,
                debug=False,
            )
            self.perception.yolo_model = yolo_model
            self.perception.regressor_model = regressor_model
            self.perception.detector = detector
            self.perception.image_size = getattr(args, "size", 256)
            self.perception.device = self.device
            self.perception.warnings = {"obj_to_pick": True, "place_to_drop": True}
            self.perception._skill_snapshot_pos = None
            print("[auto_demo] Using executor vision pipeline "
                  "(detect_cubes_simple + stereo/residual/regressor)")
        self.reset()

    def reset_perception_snapshot(self):
        if self.perception is not None:
            self.perception.reset_tracking()
            self.perception.warnings = {"obj_to_pick": True, "place_to_drop": True}

    def get_plan(self):
        state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        print("Detected state: ", state)
        init_predicates = {predicate: True for predicate in state.keys() if state[predicate] and predicate.split("(")[0] in planning_predicates[self.args.env]}
        print("Initial predicates: ", init_predicates)
        copy_init_predicates = init_predicates.copy()
        for predicate in copy_init_predicates.keys():
            if "ref" in predicate:
                init_predicates.pop(predicate)
        add_predicates_to_pddl(self.pddl_path, init_predicates)
        if self.args.env == "CubeSorting":
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
            define_goal_in_pddl(self.pddl_path, goal_predicates)
        elif self.args.env == "HeightStacking":
            goal_predicates = []
            sizes = {}
            for predicate in state.keys():
                if "smaller" in predicate and state[predicate]:
                    objs = predicate[predicate.find("(")+1:predicate.find(")")].split(",")
                    sizes[objs[0]] = objs[1]
            sorted_sizes = sorted(sizes.items(), key=lambda x: x[1])
            for i in range(len(sorted_sizes)-1):
                goal_predicates.append(f'on {sorted_sizes[i][0]} {sorted_sizes[i+1][0]}')
            goal_str = "\n".join(goal_predicates)
            goal_predicates.append(f'on {sorted_sizes[-1][0]} platform')
            print("Goal predicates: ", goal_str)
            define_goal_in_pddl(self.pddl_path, goal_predicates)
        elif self.args.env == "AssemblyLineSorting":
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
            define_goal_in_pddl(self.pddl_path, goal_predicates)
        elif self.args.env == "PatternReplication":
            goal_predicates = self.detector.get_pattern_replication_goal()
            goal_str = "\n".join(goal_predicates)
            print("Goal predicates: ", goal_str)
            define_goal_in_pddl(self.pddl_path, goal_predicates)

        
        self.plan, _ = call_planner(self.pddl_path, mode=planning_mode[self.args.env])
        if not self.plan:
            # Planner returns False on failure / already-solved goals (common with
            # --rnd_reset when the random tower already matches the PDDL goal peg).
            self.plan = []
        if self.args.one_operation and len(self.plan) > 1:
            self.plan = self.plan[:2]
        print("Task demonstrated: ", self.plan)


    def _get_lift_drop_target(self, op_idx):
        op = self.plan[op_idx].lower().split(" ")
        if not op or "pick" not in op[0]:
            return None
        picked_obj = op[1] if len(op) > 1 else None
        for next_op in self.plan[op_idx + 1:]:
            parts = next_op.lower().split(" ")
            if "place" in parts[0] and len(parts) == 3 and parts[1] == picked_obj:
                return parts[2]
        return None

    def operator_to_function(self, operation):
        operation = operation.lower().split(' ')
        print("Operation: ", operation[0])
        if len(operation) == 3:
            self.obj_to_pick = operation[1]
            self.place_to_drop = operation[2]
        elif 'turn' in operation[0]:
            self.obj_to_pick = None
            self.place_to_drop = None
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

    def pixel_to_world_pos(self, cls_id, px1, py1, w1, h1, conf1, px2, py2, w2, h2, conf2, ee_x, ee_y, ee_z):
        models_dual = self.regressor_model
        reg_x_dual, reg_y_dual, reg_z_dual = models_dual["reg_x"], models_dual["reg_y"], models_dual["reg_z"]

        try:
            features = np.array([[
                        to_float(cls_id),
                        to_float(px1), to_float(py1), to_float(w1), to_float(h1), to_float(conf1),
                        to_float(px2), to_float(py2), to_float(w2), to_float(h2), to_float(conf2),
                        to_float(ee_x), to_float(ee_y), to_float(ee_z)
                    ]], dtype=np.float64)
            x = reg_x_dual.predict(features)[0]
            y = reg_y_dual.predict(features)[0]
            z = reg_z_dual.predict(features)[0]
        except ValueError:
            features = np.array([[
                        to_float(px1), to_float(py1), to_float(w1), to_float(h1), to_float(conf1),
                        to_float(px2), to_float(py2), to_float(w2), to_float(h2), to_float(conf2),
                        to_float(ee_x), to_float(ee_y), to_float(ee_z)
                    ]], dtype=np.float64)

            x = reg_x_dual.predict(features)[0]
            y = reg_y_dual.predict(features)[0]
            z = reg_z_dual.predict(features)[0]
        return x, y, z

    def _camera_ray(self, cam_name, row_disp, col_disp, image_h, image_w):
        sim = self.env.sim
        cam_id = sim.model.camera_name2id(cam_name)
        cam_pos = sim.data.cam_xpos[cam_id].copy()
        cam_rot = sim.data.cam_xmat[cam_id].reshape(3, 3).copy()
        fovy = sim.model.cam_fovy[cam_id]
        f = 0.5 * image_h / np.tan(fovy * np.pi / 360)
        v = image_h - 1 - row_disp
        u = col_disp
        x_c = u - image_w / 2.0
        y_c = -(v - image_h / 2.0)
        z_c = -f
        dir_cam = np.array([x_c, y_c, z_c], dtype=np.float64)
        dir_cam /= np.linalg.norm(dir_cam)
        return cam_pos, cam_rot @ dir_cam

    def triangulate_dual(self, px1, py1, w1, h1, px2, py2, w2, h2, image_h=256, image_w=256):
        if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
            return None
        row1 = image_h - 1 - py1
        row2 = image_h - 1 - py2
        o1, d1 = self._camera_ray("agentview", row1, px1, image_h, image_w)
        o2, d2 = self._camera_ray("robot0_eye_in_hand", row2, px2, image_h, image_w)
        A = np.stack([d1, -d2], axis=1)
        ATA = A.T @ A
        if np.linalg.det(ATA) < 1e-12:
            return None
        ts = np.linalg.solve(ATA, A.T @ (o2 - o1))
        p1 = o1 + ts[0] * d1
        p2 = o2 + ts[1] * d2
        return (p1 + p2) / 2.0

    def yolo_estimate(self, obs_step):

        cubes_predicted_xyz = {}
        confidence_agent = {}
        if "agentview" not in obs_step or "wrist_image" not in obs_step:
            print("No images provided for YOLO estimation.")
            return cubes_predicted_xyz
        agentview = obs_step["agentview"]
        wrist_image = obs_step["wrist_image"]
        agentview_results = self.yolo_model(agentview, verbose=False, device=self.device)[0]
        wrist_results = self.yolo_model(wrist_image, verbose=False, device=self.device)[0]
        objects_info = obs_step["objects_pos"]
        for pred in agentview_results.boxes:
            confidence_wrist = {}
            cls_id = int(pred.cls)
            cls = self.yolo_model.names[cls_id]
            cls = self.yolo_id_mapping[cls] if cls in self.yolo_id_mapping else cls
            x, y, w, h = pred.xywhn.tolist()[0]
            conf = pred.conf
            x = int(x * agentview.shape[1])
            y = int(y * agentview.shape[0])
            w = int(w * agentview.shape[1])
            h = int(h * agentview.shape[0])

            if cls_id not in confidence_agent:
                confidence_agent[cls_id] = conf
            else:
                if conf > confidence_agent[cls_id]:
                    confidence_agent[cls_id] = conf
                else:
                    continue

            if self.args.render:
                cv2.rectangle(agentview, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 255, 0), 2)
                cv2.putText(agentview, f"{cls} {float(conf):.2f}", (x - w // 2, y - h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imshow('Agentview YOLO Detections', agentview)
                cv2.waitKey(1)

            if cls in objects_info:
                ground_truth_xyz = objects_info[cls]
                ee_pos = objects_info["gripper"]

                found_match = False
                for wrist_pred in wrist_results.boxes:
                    cls_id2 = int(wrist_pred.cls)
                    cls2 = self.yolo_model.names[cls_id2]
                    if cls_id2 not in confidence_wrist:
                        confidence_wrist[cls_id2] = wrist_pred.conf
                    else:
                        if wrist_pred.conf > confidence_wrist[cls_id2]:
                            confidence_wrist[cls_id2] = wrist_pred.conf
                        else:
                            continue
                    if cls_id2 == cls_id:
                            found_match = True
                            x_cam2, y_cam2, w_cam2, h_cam2 = wrist_pred.xywhn.tolist()[0]
                            conf_cam2 = wrist_pred.conf
                            x_cam2 = int(x_cam2 * wrist_image.shape[1])
                            y_cam2 = int(y_cam2 * wrist_image.shape[0])
                            w_cam2 = int(w_cam2 * wrist_image.shape[1])
                            h_cam2 = int(h_cam2 * wrist_image.shape[0])

                            tri = self.triangulate_dual(x, y, w, h, x_cam2, y_cam2, w_cam2, h_cam2,
                                                         image_h=agentview.shape[0], image_w=agentview.shape[1])
                            self.bboxes_centers.append({
                                "px_cam1": x,
                                "py_cam1": y,
                                "w_cam1": w,
                                "h_cam1": h,
                                "conf_cam1": float(conf),
                                "cls": cls,
                                "px_cam2": x_cam2,
                                "py_cam2": y_cam2,
                                "w_cam2": w_cam2,
                                "h_cam2": h_cam2,
                                "conf_cam2": float(conf_cam2),
                                "ee_x": ee_pos[0] if ee_pos is not None else None,
                                "ee_y": ee_pos[1] if ee_pos is not None else None,
                                "ee_z": ee_pos[2] if ee_pos is not None else None,
                                "tri_x": tri[0] if tri is not None else None,
                                "tri_y": tri[1] if tri is not None else None,
                                "tri_z": tri[2] if tri is not None else None,
                                "world_x": ground_truth_xyz[0],
                                "world_y": ground_truth_xyz[1],
                                "world_z": ground_truth_xyz[2],
                            })
                if not found_match:
                    x_cam2, y_cam2, w_cam2, h_cam2, conf_cam2 = 0, 0, 0, 0, 0
                    self.bboxes_centers.append({
                        "px_cam1": x,
                        "py_cam1": y,
                        "w_cam1": w,
                        "h_cam1": h,
                        "conf_cam1": float(conf),
                        "cls": cls,
                        "px_cam2": 0,
                        "py_cam2": 0,
                        "w_cam2": 0,
                        "h_cam2": 0,
                        "conf_cam2": 0,
                        "ee_x": ee_pos[0] if ee_pos is not None else None,
                        "ee_y": ee_pos[1] if ee_pos is not None else None,
                        "ee_z": ee_pos[2] if ee_pos is not None else None,
                        "tri_x": None,
                        "tri_y": None,
                        "tri_z": None,
                        "world_x": ground_truth_xyz[0],
                        "world_y": ground_truth_xyz[1],
                        "world_z": ground_truth_xyz[2],
                        })
                predicted_xyz = self.pixel_to_world_pos(cls_id, x, y, w, h, conf, x_cam2, y_cam2, w_cam2, h_cam2, conf_cam2, ee_pos[0], ee_pos[1], ee_pos[2])
                cubes_predicted_xyz.update({cls: predicted_xyz})
            else:
                print("Object detected, but no mapping to ground truth position found for class: ", cls, " in objects_info with keys: ", objects_info.keys())

        return cubes_predicted_xyz

    def reset(self, seed=None):
        if self.args.checkpoints > 0:
            if self.recorded_eps // self.args.checkpoints > self.checkpoint:
                print("\nSAVING CHECKPOINT: ", self.checkpoint)
                self.data_buffer = dict()
                self.action_steps = []
                self.checkpoint += 1
        self.operator_step = 0
        self.episode_buffer = dict()
        self.task_buffer = list()
        self.success_reset = False
        while not self.success_reset:
            try:
                try:
                    obs = self.env.reset()
                except AttributeError:
                    obs = self.env.env.reset()
                except AttributeError:
                    obs = self.env.reset()
                self.env.sim.forward()
                if not self.valid_state(self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)):
                    print("Invalid state detected after reset, resetting again...")
                    self.success_reset = False
                    continue
                self.get_plan()
            except Exception as e:
                print("Reset failed with exception: ", e)
                self.success_reset = False
                continue
            if not self.plan or len(self.plan) == 0:
                print("Empty plan, resetting again...")
                self.success_reset = False
            else:
                self.success_reset = True
        return obs
    
    def valid_state(self, state):
        on_predicates = [predicate for predicate in state.keys() if "on" in predicate and state[predicate]]
        if len(on_predicates) != 3:
            return False
        objects_on = {}
        for predicate in on_predicates:
            objs = predicate[predicate.find("(")+1:predicate.find(")")].split(", ")[0].split(",")
            if objs[1] not in objects_on:
                objects_on[objs[1]] = objs[0]
            else:
                return False
        return True

    def run_trajectory(self, obs):
        done = False
        for op_idx, operation in enumerate(self.plan):
            self.lift_drop_target = self._get_lift_drop_target(op_idx)
            function, self.task, goal = self.operator_to_function(operation)
            self.reset_perception_snapshot()
            print(f'Performing task: {self.task}, with goal: {goal}')
            done, obs = function(obs, goal)
            if not(done):
                print("Failed to perform task")
                return False
            print("Successful task")
        if not self.valid_state(self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)):
            print("Invalid state detected after operation: ", operation)
            done = False
        print("Successful episode?: ", done)
        return done

    def save_trajectory(self):
        if self.args.vla:
            episode = []
            for step in self.action_steps:
                if step in self.episode_buffer.keys():
                    for i in range(0,len(self.episode_buffer[step]),2): 
                        action_4dim = self.episode_buffer[step][i]
                        action = np.concatenate((action_4dim[:3], np.zeros(3), action_4dim[3:]))

                        raw_image = self.episode_buffer[step][i+1]
                        image = np.array(raw_image).reshape((256, 256, 3))
                        
                        episode.append({
                            'image': image,
                            'action': action,
                            'language_instruction': step,
                        })
            np.save(f'data/kitchen_env/episode_{self.recorded_eps}.npy', episode)
        
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
        self.save_csv_yolo(output_path=os.path.join(self.args.yolo_data, "yolo_data_{}.csv".format(self.recorded_eps)))
        obs = self.reset()
        return obs

    def save_csv_yolo(self, output_path="yolo_data.csv"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if not self.bboxes_centers:
            print("No bounding boxes data to save.")
            return
        pd.DataFrame(self.bboxes_centers).to_csv(output_path, index=False)
        print(f"YOLO data saved at {output_path}")

    def zip_buffer(self, dir_path):
        for step in self.action_steps:
            if step not in self.data_buffer.keys():
                continue
            data_bytes = pickle.dumps(self.data_buffer[step])
            if self.args.checkpoints > 0:
                file_path = dir_path + str(self.checkpoint) + '/' + step + '.zip'
                os.makedirs(dir_path + str(self.checkpoint), exist_ok=True)
            else:
                file_path = dir_path + step + '.zip'
            with zipfile.ZipFile(file_path, 'w') as zip_file:
                with zip_file.open('data.pkl', 'w', force_zip64=True) as file:
                    file.write(data_bytes)

    def get_object_obs(self, objects_pos, predicted_pos, relative_obs=True):
        if self.perception is not None and self.args.use_yolo:
            self.perception.warnings.setdefault("obj_to_pick", True)
            self.perception.warnings.setdefault("place_to_drop", True)
            return self.perception.get_object_obs(
                self.env,
                objects_pos,
                predicted_pos,
                self.obj_to_pick,
                self.place_to_drop,
                relative_obs=relative_obs,
            )

        gripper_pos = objects_pos["gripper"]
        try:
            from robot_utils import gripper_aperture
            aperture = gripper_aperture(self.env.sim)
        except Exception:
            left_finger_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_leftfinger")])
            right_finger_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_rightfinger")])
            aperture = np.linalg.norm(left_finger_pos - right_finger_pos)
        
        try:
            obj_to_pick_pos = predicted_pos[self.obj_to_pick] if self.obj_to_pick in predicted_pos else objects_pos[self.obj_to_pick]
        except Exception:
            obj_to_pick_pos = np.array([0.0, 0.0, 0.0])
        try:
            place_to_drop_pos = predicted_pos[self.place_to_drop] if self.place_to_drop in predicted_pos else objects_pos[self.place_to_drop]
        except Exception:
            place_to_drop_pos = np.array([0.0, 0.0, 0.0])
        if relative_obs:
            rel_obj_to_pick_pos = gripper_pos - obj_to_pick_pos
            rel_place_to_drop_pos = gripper_pos - place_to_drop_pos
            obs = np.concatenate([gripper_pos, [aperture], rel_obj_to_pick_pos, rel_place_to_drop_pos])
        else:
            obs = np.concatenate([gripper_pos, [aperture], obj_to_pick_pos, place_to_drop_pos])
        return obs

    def filter_obs(self, obs, action_step="main"):
        if action_step == "main":
            return obs
        elif action_step == "pick":
            return np.concatenate([[obs[6]], [obs[3]]])
        elif action_step == "place":
            return np.concatenate([[obs[-1]], [obs[3]]])
        elif action_step == "turn_on":
            return obs[:3]
        elif action_step == "turn_off":
            return obs[:3]
        elif action_step == "reach_pick":
            return obs[4:7]
        elif action_step == "reach_place":
            return obs[-3:]

    def record_step(self, obs, action, next_obs, state_memory, next_state=None, action_step="main", goal=None, info={}):
        if not(self.args.action_split):
            action_step = "main"
        full_obs = self.env.env._get_observations()
        objects_pos = self.detector.get_all_objects_pos()
        agentview_raw = np.array(full_obs["agentview_image"].reshape((self.args.size, self.args.size, 3)), dtype=np.uint8)
        wrist_raw = np.array(full_obs["robot0_eye_in_hand_image"].reshape((self.args.size, self.args.size, 3)), dtype=np.uint8)
        next_obs, obs = {}, {}
        next_obs['objects_pos'] = {k: np.asarray(v, dtype=np.float32).copy() for k, v in objects_pos.items()}
        next_obs["agentview"] = cv2.cvtColor(cv2.flip(agentview_raw.reshape(self.args.size, self.args.size, 3), 0), cv2.COLOR_RGB2BGR)
        next_obs["wrist_image"] = cv2.cvtColor(cv2.flip(wrist_raw.reshape(self.args.size, self.args.size, 3), 0), cv2.COLOR_RGB2BGR)
        next_obs['proprio'] = np.asarray(full_obs["robot0_proprio-state"], dtype=np.float32).copy()
        if self.args.render:
            cv2.imshow('agentview', next_obs["agentview"])
            cv2.waitKey(1)
        if self.args.use_yolo:
            if self.detector is None:
                print("Detector is None")
                return None
            if self.perception is None:
                print("Perception pipeline is None (need YOLO + regressor)")
                return None
            ee_pos = objects_pos["gripper"]
            if getattr(self.perception, "_skill_snapshot_pos", None) is None:
                predicted_cubes_xyz, relations = self.perception.detect_cubes_simple(
                    agentview_raw,
                    wrist_raw,
                    ee_pos,
                    conf_threshold=0.8,
                    sim=self.env.sim,
                    render=self.args.render,
                )
                self.perception.relations = relations
                self.perception.map_id_semantic = {
                    y: p for p, y in relations.items() if y is not None
                }
                self.perception._skill_snapshot_pos = copy.deepcopy(predicted_cubes_xyz)
            else:
                predicted_cubes_xyz = copy.deepcopy(self.perception._skill_snapshot_pos or {})
                if self.args.render and getattr(self.perception, "_last_yolo_viz", None) is not None:
                    cv2.imshow("YOLO Detections", self.perception._last_yolo_viz)
                    cv2.waitKey(1)

            predicted_cubes_xyz = self.perception.correct_grasped_object_positions(
                predicted_cubes_xyz,
                ee_pos,
                image_shape=agentview_raw.shape,
            )
            if self.args.train_yolo:
                self._append_yolo_csv_from_snapshot(
                    predicted_cubes_xyz, objects_pos, ee_pos
                )
            self.df_metrics = self.compute_metrics(
                self.df_metrics, objects_pos, predicted_cubes_xyz
            )
            next_obs = self.get_object_obs(
                objects_pos, predicted_cubes_xyz, relative_obs=True
            )
        elif self.args.object_centric:
            if self.detector is None:
                print("Detector is None")
                return None
            next_obs = self.get_object_obs(objects_pos, {}, relative_obs=True)
        elif self.args.train_yolo:
            cubes_predicted_xyz = self.yolo_estimate(next_obs)
            self.df_metrics = self.compute_metrics(
                self.df_metrics, objects_pos, cubes_predicted_xyz
            )
            next_obs = self.get_object_obs(objects_pos, {}, relative_obs=True)
        if self.args.action_split:
            next_obs = self.filter_obs(next_obs, action_step)
        if action_step not in self.action_steps:
            self.action_steps.append(action_step)
        if action_step not in self.episode_buffer.keys():
            self.episode_buffer[action_step] = [action, next_obs]
        else:
            self.episode_buffer[action_step] += [action, next_obs]
        return state_memory

    def _append_yolo_csv_from_snapshot(self, predicted_pos, objects_pos, ee_pos):
        relations = getattr(self.perception, "relations", {}) or {}
        for pddl_id, yolo_id in relations.items():
            if yolo_id not in predicted_pos or pddl_id not in objects_pos:
                continue
            pred = predicted_pos[yolo_id]
            gt = objects_pos[pddl_id]
            self.bboxes_centers.append({
                "px_cam1": 0, "py_cam1": 0, "w_cam1": 0, "h_cam1": 0, "conf_cam1": 0.0,
                "cls": yolo_id,
                "px_cam2": 0, "py_cam2": 0, "w_cam2": 0, "h_cam2": 0, "conf_cam2": 0.0,
                "ee_x": ee_pos[0], "ee_y": ee_pos[1], "ee_z": ee_pos[2],
                "tri_x": pred[0], "tri_y": pred[1], "tri_z": pred[2],
                "world_x": gt[0], "world_y": gt[1], "world_z": gt[2],
            })

    def compute_metrics(self, df, objects_pos, predicted_pos):
        relations = {}
        if self.perception is not None:
            relations = getattr(self.perception, "relations", {}) or {}

        targets = [self.obj_to_pick, self.place_to_drop]
        for obj in targets:
            if obj is None:
                continue
            pred_key = relations.get(obj, obj)
            if pred_key in predicted_pos and obj in objects_pos:
                predicted_pos_obj = predicted_pos[pred_key]
                world_pos_obj = objects_pos[obj]
                df = pd.concat(
                    [
                        df,
                        pd.DataFrame([{
                            'wx': world_pos_obj[0],
                            'wy': world_pos_obj[1],
                            'wz': world_pos_obj[2],
                            'pred_x': predicted_pos_obj[0],
                            'pred_y': predicted_pos_obj[1],
                            'pred_z': predicted_pos_obj[2],
                            'error': np.linalg.norm(np.array(predicted_pos_obj) - np.array(world_pos_obj))
                        }])
                    ],
                    ignore_index=True
                )
                df["diff_x"] = df["wx"] - df["pred_x"]
                df["diff_y"] = df["wy"] - df["pred_y"]
                df["diff_z"] = df["wz"] - df["pred_z"]
        return df

    def cap(self, eps, max_val=0.12, min_val=0.01):
        if np.linalg.norm(eps) > max_val:
            eps = eps / np.linalg.norm(eps) * max_val
        if np.linalg.norm(eps) < min_val:
            eps = eps / np.linalg.norm(eps) * min_val
        return eps

    def to_osc_pose(self, action):
        if not self.args.ee:
            action = np.insert(action, 3, [0, 0, 0])
        return action

    def pick(self, obs, goal):
        state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        self.state_memory = state
        reset_step_count = 0

        goal_str = goal
        goal = self.env.sim.model.body_name2id(self.detector.object_id[goal_str])

        ref_z = 1.1
        z_pos = 0
        while z_pos < ref_z:
            z_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])[2]
            dist_z = abs(ref_z - z_pos)
            dist_z = self.cap([dist_z])
            action = 5*np.concatenate([[0, 0], dist_z, [0]]) if not(self.randomize) else 5*np.concatenate([[0, 0], dist_z, [0]]) + np.concatenate([[0, 0], np.random.normal(0, self.noise_std_factor*np.linalg.norm(dist_z), 1), [0]])
            action = self.to_osc_pose(action)
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="reach_pick", goal=goal_str, info=info)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 500:
                return False, obs
            dist_z = dist_z[0]
        reset_step_count = 0
        self.env.time_step = 0

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
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="reach_pick", goal=goal_str, info=info)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 500:
                return False, obs
        reset_step_count = 0
        self.env.time_step = 0

        while not state['open_gripper(gripper)']:
            action = np.asarray([0,0,0,-1])
            action = self.to_osc_pose(action)
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="pick", goal=goal_str, info=info)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 100:
                return False, obs
        reset_step_count = 0
        self.env.time_step = 0

        while not state['at_grab_level(gripper,{})'.format(goal_str)]:
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
            object_pos = np.asarray(self.env.sim.data.body_xpos[goal])
            dist_z_axis = [object_pos[2] - gripper_pos[2]]
            dist_z_axis = self.cap(dist_z_axis)
            action = 5*np.concatenate([[0, 0], dist_z_axis, [0]]) if not(self.randomize) else 5*np.concatenate([[0, 0], dist_z_axis, [0]]) + np.concatenate([[0, 0], np.random.normal(0, self.noise_std_factor*np.linalg.norm(dist_z_axis), 1), [0]])
            action = self.to_osc_pose(action)
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="pick", goal=goal_str, info=info)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 400:
                return False, obs
        reset_step_count = 0
        self.env.time_step = 0

        while not state['grasped({})'.format(goal_str)]:
            action = np.asarray([0,0,0,1])
            action = self.to_osc_pose(action)
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="pick", goal=goal_str, info=info)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 30:
                return False, obs
        reset_step_count = 0
        self.env.time_step = 0

        original_place_to_drop = self.place_to_drop
        if getattr(self, "lift_drop_target", None) is not None:
            self.place_to_drop = self.lift_drop_target
        try:
            while not state['picked_up({})'.format(goal_str)]:
                action = np.asarray([0,0,0.4,0]) if not(self.randomize) else [0,0,0.5,0] + np.concatenate([np.random.normal(0, 0.1, 3), [0]])
                action = 5*self.cap(action)
                action = self.to_osc_pose(action)
                next_obs, _, _, _, info  = self.env.step(action)
                self.env.render() if self.render else None
                next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
                self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="reach_place", goal=goal_str, info=info)
                if self.state_memory is None:
                    return False, obs
                obs, state = next_obs, next_state
                reset_step_count += 1
                if reset_step_count > 300:
                    return False, obs
        finally:
            self.place_to_drop = original_place_to_drop
        reset_step_count = 0
        self.env.time_step = 0

        return True, obs

    def place(self, obs, goal):
        state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        self.state_memory = state
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
            if pick_str == 'pot':
                gripper_pos = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("PotObject_root")]
            elif self.args.env == "NutAssembly":
                if goal_str == "roundpeg":
                    gripper_pos = np.asarray(self.env.sim.data.body_xpos[pick_body]) +np.array([0, -0.01, 0])
                elif goal_str == "squarepeg":
                    gripper_pos = np.asarray(self.env.sim.data.body_xpos[pick_body]) +np.array([+0.015, +0.01, 0])
            else:
                gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
            if 'peg' in goal_str and self.args.env == "Hanoi":
                object_pos = self.detector.area_pos[goal_str]
            elif "serving" in goal_str:
                object_pos = np.asarray(self.env.sim.data.body_xpos[goal])+np.array([0.002, -0.05, 0])
            else:
                object_pos = np.asarray(self.env.sim.data.body_xpos[goal])
            dist_xy_plan = object_pos[:2] - gripper_pos[:2]
            dist_xy_plan = self.cap(dist_xy_plan)
            action = 5*np.concatenate([dist_xy_plan, [0, 0]]) if not(self.randomize) else 5*np.concatenate([dist_xy_plan, [0, 0]]) + np.concatenate([np.random.normal(0, self.noise_std_factor*np.linalg.norm(dist_xy_plan), 3), [0]])
            action = self.to_osc_pose(action)
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="reach_place", goal=goal_str, info=info)
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
            place_pos = np.asarray(self.env.sim.data.body_xpos[goal])
            dist_z_axis = [- (gripper_pos[2] - place_pos[2])]
            dist_z_axis = self.cap(dist_z_axis)
            action = 5*np.concatenate([[0, 0], dist_z_axis, [0]]) if not(self.randomize) else 5*np.concatenate([[0, 0], dist_z_axis, [0]]) + np.concatenate([[0, 0], np.random.normal(0, self.noise_std_factor*np.linalg.norm(dist_z_axis), 1), [0]])
            action = self.to_osc_pose(action)
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="place", goal=goal_str, info=info)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 200:
                return False, obs
        reset_step_count = 0
        self.env.time_step = 0

        print("dropping object...")
        while not(state['open_gripper(gripper)']):
            action = np.asarray([0,0,0,-1])
            action = self.to_osc_pose(action)
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="place", goal=goal_str, info=info)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 30:
                return False, obs
        reset_step_count = 0
        self.env.time_step = 0

        for _ in range(15):
            action = np.array([0, 0, 1, 0])
            action = self.to_osc_pose(action)
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="place", goal=goal_str, info=info)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1

        return True, obs
    
    def turn_on_button(self, obs, goal):
        state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        self.state_memory = state

        goal_str = goal
        goal = self.env.sim.model.body_name2id(self.detector.object_id[goal_str])

        goal_pos = self.env.sim.data.body_xpos[goal][:3]
        goal_quat = self.env.sim.data.body_xquat[goal]
        self.keypoint = np.concatenate([goal_pos, goal_quat])

        reset_step_count = 0

        while not state['over(gripper,{})'.format(goal_str)]:
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
            object_pos = np.asarray(self.env.sim.data.body_xpos[goal])
            dist_xy_plan = object_pos[:2] - gripper_pos[:2]
            dist_xy_plan = self.cap(dist_xy_plan)
            action = 5*np.concatenate([dist_xy_plan, [0, 0]]) if not(self.randomize) else 5*np.concatenate([dist_xy_plan, [0, 0]]) + np.concatenate([np.random.normal(0, self.noise_std_factor*np.linalg.norm(dist_xy_plan), 3), [0]])
            action = self.to_osc_pose(action)
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="turn_on", goal=goal_str, info=info)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 500:
                return False, obs
        reset_step_count = 0
        self.env.time_step = 0

        while not state['at_grab_level(gripper,{})'.format(goal_str)]:
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
            object_pos = np.asarray(self.env.sim.data.body_xpos[goal])
            dist_z_axis = [object_pos[2] - gripper_pos[2]]
            dist_z_axis = self.cap(dist_z_axis)
            action = 5*np.concatenate([[0, 0], dist_z_axis, [0]]) if not(self.randomize) else 5*np.concatenate([[0, 0], dist_z_axis, [0]]) + np.concatenate([[0, 0], np.random.normal(0, self.noise_std_factor*np.linalg.norm(dist_z_axis), 1), [0]])
            action = self.to_osc_pose(action)
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="turn_on", goal=goal_str, info=info)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 400:
                return False, obs
        reset_step_count = 0
        self.env.time_step = 0

        while not state['stove_on()']:
            action = np.asarray([0,0.3,0,0])
            action = self.to_osc_pose(action)
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="turn_on", goal=goal_str, info=info)
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
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="turn_on", goal=goal_str, info=info)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1

        return True, obs
    
    def turn_off_button(self, obs, goal):
        state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        self.state_memory = state

        goal_str = goal
        goal = self.env.sim.model.body_name2id(self.detector.object_id[goal_str])

        goal_pos = self.env.sim.data.body_xpos[goal][:3]
        goal_quat = self.env.sim.data.body_xquat[goal]
        self.keypoint = np.concatenate([goal_pos, goal_quat])

        reset_step_count = 0

        while not state['over(gripper,{})'.format(goal_str)]:
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
            object_pos = np.asarray(self.env.sim.data.body_xpos[goal])
            dist_xy_plan = object_pos[:2] - gripper_pos[:2]
            dist_xy_plan = self.cap(dist_xy_plan)
            action = 5*np.concatenate([dist_xy_plan, [0, 0]]) if not(self.randomize) else 5*np.concatenate([dist_xy_plan, [0, 0]]) + np.concatenate([np.random.normal(0, self.noise_std_factor*np.linalg.norm(dist_xy_plan), 3), [0]])
            action = self.to_osc_pose(action)
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="turn_off", goal=goal_str, info=info)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 500:
                return False, obs
        reset_step_count = 0
        self.env.time_step = 0

        while not state['at_grab_level(gripper,{})'.format(goal_str)]:
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
            object_pos = np.asarray(self.env.sim.data.body_xpos[goal])
            dist_z_axis = [object_pos[2] - gripper_pos[2]]
            dist_z_axis = self.cap(dist_z_axis)
            action = 5*np.concatenate([[0, 0], dist_z_axis, [0]]) if not(self.randomize) else 5*np.concatenate([[0, 0], dist_z_axis, [0]]) + np.concatenate([[0, 0], np.random.normal(0, self.noise_std_factor*np.linalg.norm(dist_z_axis), 1), [0]])
            action = self.to_osc_pose(action)
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="turn_off", goal=goal_str, info=info)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 400:
                return False, obs
        reset_step_count = 0
        self.env.time_step = 0

        while state['stove_on()']:
            action = np.asarray([0,-0.3,0,0])
            action = self.to_osc_pose(action)
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="turn_off", goal=goal_str, info=info)
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
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="turn_off", goal=goal_str, info=info)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1

        return True, obs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Hanoi', choices=['Hanoi', 'KitchenEnv', 'NutAssembly', 'CubeSorting', 'AssemblyLineSorting', 'HeightStacking', 'PatternReplication'], help='Name of the environment to run the experiment in')
    parser.add_argument('--dir', type=str, default='./data/', help='Path to the data folder')
    parser.add_argument('--episodes', type=int, default=int(30), help='Number of trajectories to record')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--name', type=str, default=None, help='Name of the experiment')
    parser.add_argument('--render', action='store_true', help='Render the recording')
    parser.add_argument('--train_yolo', action='store_true', help='Store observation to train train_yolo+regressor format')
    parser.add_argument('--object_centric', action='store_true', help='Use object centric observations, with the object at the center of the image')
    parser.add_argument('--noise_std_factor', type=float, default=0.1, help='Standard deviation of the noise to add to the actions, as a factor of the action magnitude')
    parser.add_argument('--use_yolo', action='store_true', help='Use executor vision pipeline (detect_cubes_simple + stereo/residual/regressor) for object pose observations')
    parser.add_argument('--vla', action='store_true', help='Store the data in VLA friendly format')
    parser.add_argument('--size', type=int, default=256, help='Size of the observation')
    parser.add_argument('--checkpoints', type=int, default=0, help='Saves the data every n episodes, and resets the buffer')
    parser.add_argument('--ee', action='store_true', help='Use end effector observation, without rotations')
    parser.add_argument('--rnd_reset', action='store_true', help='Random reset for Hanoi env')
    parser.add_argument('--one_operation', action='store_true', help='Only one operation per episode')
    parser.add_argument('--action_split', action='store_true', help='For testing, directly splits the trajectories')
    parser.add_argument('--robot', type=str, default='Panda', choices=['Panda', 'Kinova3'],
                        help='Manipulator model (Kinova3 uses Robotiq85Gripper by default)')
    parser.add_argument('--yolo_model', type=str, default=None,
                        help='Override YOLO weights path (default: env registry / kinova paths)')
    parser.add_argument('--regressor_model', type=str, default=None,
                        help='Override dual regressor path for --use_yolo demos')


    args = parser.parse_args()
    np.random.seed(args.seed)
    dir = args.dir
    experiment_name = args.env + '_seed_' + str(args.seed)
    experiment_id = f"{to_datestring(time.time())}"
    if args.name is not None:
        experiment_id = args.name
    args.env_dir = os.path.join(dir, experiment_name, experiment_id)

    print("Starting experiment {}.".format(os.path.join(experiment_name, experiment_id)))
    print(f"Robot: {args.robot}")

    args.traces = args.env_dir + '/traces/'
    args.yolo_data = args.env_dir + '/yolo_data/'
    os.makedirs(args.env_dir, exist_ok=True)
    os.makedirs(args.traces, exist_ok=True)
    os.makedirs(args.yolo_data, exist_ok=True)
    

    obj_mapping = env_pddl_mapping[args.env]

    if not args.ee:
        controller_config = suite.load_controller_config(default_controller='OSC_POSE')
    else:
        controller_config = suite.load_controller_config(default_controller='OSC_POSITION')
    if args.env == 'Hanoi':
        env = suite.make(
            args.env,
            robots=args.robot,
            controller_configs=controller_config,
            has_renderer=args.render,
            has_offscreen_renderer=True,
            horizon=100000000,
            use_camera_obs=True,
            use_object_obs=False,
            camera_names=["agentview", "robot0_eye_in_hand"],
            camera_heights=args.size,
            camera_widths=args.size,
            random_block_placement=args.rnd_reset,
            cube_init_pos_noise_std=0.025,
            peg_xy_jitter=0.025,
        )
    else:
        env = suite.make(
            args.env,
            robots=args.robot,
            controller_configs=controller_config,
            has_renderer=args.render,
            has_offscreen_renderer=True,
            horizon=100000000,
            use_camera_obs=True,
            use_object_obs=False,
            camera_names=["agentview", "robot0_eye_in_hand"],
            camera_heights=args.size,
            camera_widths=args.size,
            cube_placement_noise=0.025,
        )


    env.reset()

    detector = env_detectors[args.env](env)
    pddl_path = pddl_paths[args.env]
    yolo_path = args.yolo_model or yolo_model_paths[args.env]
    reg_path = args.regressor_model or regressor_model_paths[args.env]
    yolo_model = YOLO(yolo_path) if args.use_yolo or args.train_yolo else None
    yolo_id_mapping = yolo_id_mappings[args.env] if args.use_yolo or args.train_yolo else None
    regressor_model = joblib.load(reg_path) if args.use_yolo or args.train_yolo else None
    env = GymWrapper(env, proprio_obs=True, flatten_obs=False)
    env = RecordDemos(env, 
                      detector, 
                      pddl_path, 
                      args, 
                      yolo_model=yolo_model, 
                      regressor_model=regressor_model, 
                      yolo_id_mapping=yolo_id_mapping,
                      obj_mapping=obj_mapping,
                      render=args.render, 
                      randomize=True,
                      noise_std_factor=args.noise_std_factor)

    os.makedirs(f'data/', exist_ok=True)

    episode = 1
    while env.recorded_eps < args.episodes: 
        obs = env.reset()
        print("Episode: {}".format(episode))
        done = env.run_trajectory(obs)
        if done:
            obs = env.save_trajectory()
        episode += 1
        print("Number of recorded episodes: {}".format(env.recorded_eps))
        if args.use_yolo:
            print("Metric df: \n", env.df_metrics.describe())
            mean_error_x = env.df_metrics["diff_x"].mean()
            mean_error_y = env.df_metrics["diff_y"].mean()
            mean_error_z = env.df_metrics["diff_z"].mean()
            std_error_x = env.df_metrics["diff_x"].std()
            std_error_y = env.df_metrics["diff_y"].std()
            std_error_z = env.df_metrics["diff_z"].std()
            print(f"Mean Error X: {mean_error_x}, Std Error X: {std_error_x}")
            print(f"Mean Error Y: {mean_error_y}, Std Error Y: {std_error_y}")
            print(f"Mean Error Z: {mean_error_z}, Std Error Z: {std_error_z}")
            print(f"Overall Mean Error: {(mean_error_x**2 + mean_error_y**2 + mean_error_z**2)**0.5}")
            print("\n\n")
        print("\n\n")
