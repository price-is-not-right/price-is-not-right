import os
import dill
import joblib
import torch
import numpy as np
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.workspace.train_diffusion_transformer_lowdim_workspace import TrainDiffusionTransformerLowdimWorkspace
from scipy.optimize import linear_sum_assignment
from planning.object_metadata import *
import copy

import cv2
cv2.destroyAllWindows = lambda: None

class Executor():
	def __init__(self, id, mode, Beta=None):
		super().__init__()
		self.id = id
		self.Beta = Beta
		self.mode = mode
		self.policy = None

	def path_to_json(self):
		return {self.id:self.policy}


class Executor_Diffusion(Executor):
    def __init__(self, 
                 id, 
                 policy, 
                 Beta, 
                 count=0,
                 nulified_action_indexes=[],
                 nulified_action_values=None,
                 oracle=False,
                 horizon=None, 
                 use_yolo=False, 
                 save_data=False,
                 instances_per_label=None,
                 particle_filter_particles_2d=100,
                 particle_filter_particles_3d=100,
                 max_position_jump=0.10,
                 max_bbox_jump=20,
                 debug=False
                 ):
        super().__init__(id, "RL", Beta)
        self.debug = debug
        self.policy = policy
        self.model = None
        self.nulified_action_indexes = nulified_action_indexes
        # Value inserted for action dims the policy does not emit. Defaults to 0
        # (hold); a Robotiq needs an explicit open/close to hold against load.
        self.nulified_action_values = {
            int(k): float(v) for k, v in (nulified_action_values or {}).items()
        }
        self.horizon = horizon
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.oracle = oracle
        self.use_yolo = use_yolo
        self.save_data = save_data
        self.image_buffer = []
        self.relations = {}
        self.map_id_semantic = {}
        self.detected_positions = {}
        self.bboxes_centers = []
        self.count = count
        self.count_save = 0
        
        self.instances_per_label = instances_per_label or {}
        self.tracked_objects = {}
        self.next_object_id = {}
        
        self.max_position_jump = max_position_jump
        self.max_bbox_jump = max_bbox_jump
        self.detection_outlier_count = {}
        self.max_outlier_frames = 50

    def debug_message(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    def update_yolo_to_pddl_mapping(self):
        if not self.relations:
            self.debug_message("No relations to update mapping from.")
            return
        self.map_id_semantic = {yolo_id: pddl_id for pddl_id, yolo_id in self.relations.items() if yolo_id is not None}
        self.debug_message(f"Updated map_id_semantic: {self.map_id_semantic}")

    def load_policy(self, detector=None, yolo_model=None, regressor_model=None, image_size=256):
        path = self.policy
        payload = torch.load(open(path, 'rb'), pickle_module=dill)
        cfg = payload['cfg']
        cls = TrainDiffusionTransformerLowdimWorkspace
        cfg.policy.num_inference_steps = 8
        workspace = cls(cfg)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        policy.to(self.device)
        policy.eval()
        policy.reset()
        self.model = policy

        if detector is not None:
            self.detector = detector
        if yolo_model is not None:
            self.yolo_model = yolo_model
            self.image_size = image_size
        if regressor_model is not None:
            self.regressor_model = regressor_model


    def _load_residual_regressor(self):
        import os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates = [
            os.path.join(root, "kinova", "models", "regressors", "hanoi_residual_regressor.pkl"),
            "kinova/models/regressors/hanoi_residual_regressor.pkl",
            os.path.join(root, "models", "regressors", "hanoi_residual_regressor.pkl"),
            "models/regressors/hanoi_residual_regressor.pkl",
        ]
        for path in candidates:
            if os.path.exists(path):
                try:
                    payload = joblib.load(path)
                    self.debug_message(f"Loaded residual regressor from {path}")
                    return payload
                except Exception as e:
                    self.debug_message(f"Failed to load residual regressor from {path}: {e}")
        self.debug_message("No residual regressor found; using raw triangulation.")
        return None

    def _load_mono_regressor(self):
        import os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates = [
            os.path.join(root, "kinova", "models", "regressors", "hanoi_mono_regressor.pkl"),
            "kinova/models/regressors/hanoi_mono_regressor.pkl",
            os.path.join(root, "models", "regressors", "hanoi_mono_regressor.pkl"),
            "models/regressors/hanoi_mono_regressor.pkl",
        ]
        for path in candidates:
            if os.path.exists(path):
                try:
                    payload = joblib.load(path)
                    self.debug_message(f"Loaded mono regressor from {path}")
                    return payload
                except Exception as e:
                    self.debug_message(f"Failed to load mono regressor from {path}: {e}")
        return None

    def _predict_mono_regressor(self, px1, py1, w1, h1, conf1, ee_x, ee_y, ee_z):
        if not hasattr(self, "mono_regressor"):
            self.mono_regressor = self._load_mono_regressor()
        mono = self.mono_regressor
        if mono is None:
            return None
        models = mono["models"] if isinstance(mono, dict) and "models" in mono and "reg_x" not in mono else mono
        feats = np.array([[
            float(px1), float(py1), float(w1), float(h1), float(conf1),
            float(ee_x), float(ee_y), float(ee_z),
        ]], dtype=np.float64)
        return (
            float(models["reg_x"].predict(feats)[0]),
            float(models["reg_y"].predict(feats)[0]),
            float(models["reg_z"].predict(feats)[0]),
        )

    def _camera_ray(self, sim, cam_name, row_disp, col_disp, image_h, image_w):
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
        dir_world = cam_rot @ dir_cam
        return cam_pos, dir_world

    @staticmethod
    def _triangulate_rays(o1, d1, o2, d2, max_gap=0.08):
        A = np.stack([d1, -d2], axis=1)
        ATA = A.T @ A
        if np.linalg.det(ATA) < 1e-12:
            return None
        ts = np.linalg.solve(ATA, A.T @ (o2 - o1))
        if ts[0] <= 0.05 or ts[1] <= 0.05:
            return None
        p1 = o1 + ts[0] * d1
        p2 = o2 + ts[1] * d2
        if np.linalg.norm(p1 - p2) > max_gap:
            return None
        return (p1 + p2) / 2.0

    @staticmethod
    def _in_hanoi_workspace(xyz):
        x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
        return abs(x) < 0.18 and -0.28 <= y <= 0.30 and 0.80 <= z <= 0.96

    CUBE_HALF_SIZE = {
        "blue cube": 0.02,
        "red cube": 0.0225,
        "green cube": 0.025,
    }

    def _single_view_size_depth(self, sim, cam_name, px, py, w, h, half_size, image_h, image_w):
        if w <= 0 or h <= 0 or half_size <= 0:
            return None
        cam_id = sim.model.camera_name2id(cam_name)
        cam_pos = sim.data.cam_xpos[cam_id].copy()
        cam_rot = sim.data.cam_xmat[cam_id].reshape(3, 3).copy()
        fovy = sim.model.cam_fovy[cam_id]
        f = 0.5 * image_h / np.tan(fovy * np.pi / 360.0)
        side = 2.0 * float(half_size)
        pix = max(0.5 * (float(w) + float(h)), 1.0)
        depth = f * side / pix
        row_disp = image_h - 1 - py
        v = image_h - 1 - row_disp
        u = px
        x_c = (u - image_w / 2.0) * (depth / f)
        y_c = -((v - image_h / 2.0) * (depth / f))
        z_c = -depth
        p_cam = np.array([x_c, y_c, z_c], dtype=np.float64)
        return cam_pos + cam_rot @ p_cam

    def pixel_to_world_dual(self, cls_id, px1, py1, w1, h1, conf1, px2, py2, w2, h2, conf2, ee_x, ee_y, ee_z,
                             sim=None, image_h=256, image_w=256, cls_name=None):
        self._last_pose_source = "none"
        cam1_valid = w1 > 0 and h1 > 0
        cam2_valid = w2 > 0 and h2 > 0

        if sim is not None and cam1_valid and cam2_valid:
            row1 = image_h - 1 - py1
            row2 = image_h - 1 - py2
            o1, d1 = self._camera_ray(sim, "agentview", row1, px1, image_h, image_w)
            o2, d2 = self._camera_ray(sim, "robot0_eye_in_hand", row2, px2, image_h, image_w)
            est = self._triangulate_rays(o1, d1, o2, d2)
            if est is not None and self._in_hanoi_workspace(est):
                raw = np.asarray(est, dtype=np.float64)
                if not hasattr(self, "residual_regressor"):
                    self.residual_regressor = self._load_residual_regressor()
                resid = self.residual_regressor
                if resid is not None:
                    feats = np.array([[
                        float(raw[0]), float(raw[1]), float(raw[2]),
                        float(px1), float(py1), float(w1), float(h1), float(conf1),
                        float(px2), float(py2), float(w2), float(h2), float(conf2),
                        float(ee_x), float(ee_y), float(ee_z)]], dtype=np.float64)
                    m = resid["models"] if "models" in resid else resid
                    est = np.array([
                        raw[0] + m["res_x"].predict(feats)[0],
                        raw[1] + m["res_y"].predict(feats)[0],
                        raw[2] + m["res_z"].predict(feats)[0],
                    ], dtype=np.float64)
                    if self._in_hanoi_workspace(est):
                        self._last_pose_source = "stereo"
                        return float(est[0]), float(est[1]), float(est[2])
                self._last_pose_source = "stereo"
                return float(raw[0]), float(raw[1]), float(raw[2])

        if not cam2_valid and cam1_valid:
            mono_est = self._predict_mono_regressor(
                px1, py1, w1, h1, conf1, ee_x, ee_y, ee_z)
            if mono_est is not None and self._in_hanoi_workspace(mono_est):
                self._last_pose_source = "mono"
                return mono_est

        reg_est = None
        if self.regressor_model is not None:
            models_dual = self.regressor_model
            if isinstance(models_dual, dict) and "models" in models_dual and "reg_x" not in models_dual:
                models_dual = models_dual["models"]
            reg_x_dual, reg_y_dual, reg_z_dual = models_dual["reg_x"], models_dual["reg_y"], models_dual["reg_z"]
            n_features = getattr(reg_x_dual, 'n_features_in_', 13)
            if n_features == 14:
                features = np.array([[float(cls_id),
                                    float(px1), float(py1), float(w1), float(h1), float(conf1),
                                    float(px2), float(py2), float(w2), float(h2), float(conf2),
                                    float(ee_x), float(ee_y), float(ee_z)]], dtype=np.float64)
            else:
                features = np.array([[
                          float(px1), float(py1), float(w1), float(h1), float(conf1),
                          float(px2), float(py2), float(w2), float(h2), float(conf2),
                          float(ee_x), float(ee_y), float(ee_z)]], dtype=np.float64)
            reg_est = (
                float(reg_x_dual.predict(features)[0]),
                float(reg_y_dual.predict(features)[0]),
                float(reg_z_dual.predict(features)[0]),
            )
            if self._in_hanoi_workspace(reg_est):
                self._last_pose_source = "dual"
                return reg_est

        if sim is not None and cls_name is not None and cam1_valid:
            half = self.CUBE_HALF_SIZE.get(cls_name)
            if half is not None:
                aspect = max(float(w1), float(h1)) / max(min(float(w1), float(h1)), 1.0)
                area = float(w1) * float(h1)
                # Tight gates: size-depth blows up on truncated/occluded boxes.
                if aspect <= 1.20 and area >= 450.0:
                    p = self._single_view_size_depth(
                        sim, "agentview", px1, py1, w1, h1, half, image_h, image_w)
                    if p is not None:
                        ok = (abs(float(p[0])) < 0.06 and -0.26 <= float(p[1]) <= 0.28
                              and 0.80 <= float(p[2]) <= 0.96)
                        self.debug_message(
                            f"  [SIZE-DEPTH] {cls_name} aspect={aspect:.2f} area={area:.0f} "
                            f"xyz={np.round(p, 4)} ok={ok} cam2={cam2_valid}"
                        )
                        if ok:
                            self._last_pose_source = "size"
                            return float(p[0]), float(p[1]), float(p[2])

        if reg_est is not None:
            self._last_pose_source = "dual"
            return reg_est

        self._last_pose_source = "ee"
        return float(ee_x), float(ee_y), float(ee_z)

    def detect_cubes_simple(self, image1, image2, ee_pos, conf_threshold=0.8, sim=None, render=False):
        image1 = cv2.cvtColor(cv2.flip(cv2.resize(image1, (256, 256)), 0), cv2.COLOR_RGB2BGR)
        image2 = cv2.cvtColor(cv2.flip(cv2.resize(image2, (256, 256)), 0), cv2.COLOR_RGB2BGR)
        pred1 = self.yolo_model.predict(image1, verbose=False, device=self.device)[0]
        pred2 = self.yolo_model.predict(image2, verbose=False, device=self.device)[0]

        best_cam1 = {}
        for box in pred1.boxes:
            conf = float(box.conf)
            if conf < conf_threshold:
                continue
            cls_id = int(box.cls)
            cls = self.yolo_model.names[cls_id]
            x, y, w, h = box.xywhn.tolist()[0]
            x, y = int(x * image1.shape[1]), int(y * image1.shape[0])
            w, h = int(w * image1.shape[1]), int(h * image1.shape[0])
            if cls not in best_cam1 or conf > best_cam1[cls][0]:
                best_cam1[cls] = (conf, cls_id, x, y, w, h)

        wrist_conf_threshold = min(conf_threshold, 0.25)
        best_cam2 = {}
        for box in pred2.boxes:
            conf = float(box.conf)
            if conf < wrist_conf_threshold:
                continue
            cls_id = int(box.cls)
            cls = self.yolo_model.names[cls_id]
            if cls not in best_cam1:
                continue
            x, y, w, h = box.xywhn.tolist()[0]
            x, y = int(x * image2.shape[1]), int(y * image2.shape[0])
            w, h = int(w * image2.shape[1]), int(h * image2.shape[0])
            if cls not in best_cam2 or conf > best_cam2[cls][0]:
                best_cam2[cls] = (conf, cls_id, x, y, w, h)

        predicted_pos = {}
        relations = {}
        viz = image1.copy() if render else None
        for cls, (conf1, cls_id, x1, y1, w1, h1) in best_cam1.items():
            pddl_id = self.HANOI_COLOR_TO_PDDL.get(cls)
            if pddl_id is None:
                continue
            if cls in best_cam2:
                conf2, _, x2, y2, w2, h2 = best_cam2[cls]
            else:
                conf2, x2, y2, w2, h2 = 0.0, 0, 0, 0, 0
            xyz = self.pixel_to_world_dual(
                cls_id, x1, y1, w1, h1, conf1,
                x2, y2, w2, h2, conf2,
                ee_pos[0], ee_pos[1], ee_pos[2],
                sim=sim, image_h=image1.shape[0], image_w=image1.shape[1],
                cls_name=cls,
            )
            yolo_id = f"{cls}_0"
            predicted_pos[yolo_id] = xyz
            relations[pddl_id] = yolo_id
            if not hasattr(self, "_last_det_conf2"):
                self._last_det_conf2 = {}
            self._last_det_conf2[yolo_id] = float(conf2)
            if not hasattr(self, "_last_det_source"):
                self._last_det_source = {}
            self._last_det_source[yolo_id] = getattr(self, "_last_pose_source", "none")
            self.debug_message(
                f"  [SIMPLE DET] {pddl_id} <- {yolo_id} "
                f"xyz={np.round(xyz, 4)} conf1={conf1:.2f} conf2={conf2:.2f}"
            )
            if viz is not None:
                x1i, y1i = int(x1 - w1 / 2), int(y1 - h1 / 2)
                x2i, y2i = int(x1 + w1 / 2), int(y1 + h1 / 2)
                cv2.rectangle(viz, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
                label = f"{pddl_id}:{conf1:.2f}"
                cv2.putText(viz, label, (x1i, max(12, y1i - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        if viz is not None:
            self._last_yolo_viz = viz
            cv2.imshow("YOLO Detections", viz)
            cv2.waitKey(1)

        return predicted_pos, relations

    def compute_iou(self, box1, box2):
        x1_min = box1[0] - box1[2] / 2
        y1_min = box1[1] - box1[3] / 2
        x1_max = box1[0] + box1[2] / 2
        y1_max = box1[1] + box1[3] / 2
        
        x2_min = box2[0] - box2[2] / 2
        y2_min = box2[1] - box2[3] / 2
        x2_max = box2[0] + box2[2] / 2
        y2_max = box2[1] + box2[3] / 2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0

    def is_detection_outlier(self, track_id, new_bbox, new_position):
        if track_id not in self.tracked_objects:
            return False
        
        if 'bbox' in self.tracked_objects[track_id]:
            old_bbox = self.tracked_objects[track_id]['bbox']
            bbox_center_old = np.array([old_bbox[0], old_bbox[1]])
            bbox_center_new = np.array([new_bbox[0], new_bbox[1]])
            bbox_jump = np.linalg.norm(bbox_center_new - bbox_center_old)
            
            if bbox_jump > self.max_bbox_jump:
                self.debug_message(f"  [OUTLIER] {track_id}: bbox jump {bbox_jump:.1f}px > {self.max_bbox_jump}px")
                return True
        
        if 'position' in self.tracked_objects[track_id]:
            old_position = np.array(self.tracked_objects[track_id]['position'])
            position_jump = np.linalg.norm(np.array(new_position) - old_position)
            
            if position_jump > self.max_position_jump:
                self.debug_message(f"  [OUTLIER] {track_id}: position jump {position_jump:.3f}m > {self.max_position_jump}m")
                return True
        
        return False

    def is_detection_set_valid(self, detections_by_class, current_tracked_count):
        detected_classes = set(detections_by_class.keys())
        tracked_classes = set(current_tracked_count.keys())
        
        if len(tracked_classes) > 0 and len(detected_classes) == 0:
            self.debug_message("  [INVALID SET] All objects lost in detection")
            return False
        
        for cls in detected_classes:
            detected_count = len(detections_by_class[cls])
            tracked_count = current_tracked_count.get(cls, 0)
            
            if tracked_count > 0 and detected_count > tracked_count * 2:
                self.debug_message(f"  [INVALID SET] Detected {detected_count} {cls} vs tracking {tracked_count}")
                return False
        
        return True

    def update_particle_filter_2d(self, track_id, bbox_center, velocity=None):
        pass

    def get_particle_filter_estimate_2d(self, track_id):
        if track_id in self.particle_filters_2d:
            return self.particle_filters_2d[track_id].get_estimate()
        return None

    def update_particle_filter_3d(self, track_id, position_3d, velocity=None):
        pass

    def get_particle_filter_estimate_3d(self, track_id):
        if track_id in self.particle_filters_3d:
            return self.particle_filters_3d[track_id].get_estimate()
        return None

    def project_3d_to_2d_approximate(self, position_3d, image_shape):
        
        world_x, world_y, world_z = position_3d
        
        img_x = int((world_x + 0.5) * image_shape[1])
        img_y = int((world_y + 0.5) * image_shape[0])
        
        img_x = np.clip(img_x, 0, image_shape[1] - 1)
        img_y = np.clip(img_y, 0, image_shape[0] - 1)
        
        return np.array([img_x, img_y])

    def get_grasped_objects(self):
        if not hasattr(self, 'detector'):
            return set()
        
        try:
            groundings = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            grasped_objects = set()
            
            for predicate, value in groundings.items():
                if 'grasped' in predicate and value:
                    obj_name = predicate.split('(')[1].split(')')[0]
                    grasped_objects.add(obj_name)
            
            return grasped_objects
        except Exception as e:
            self.debug_message(f"Error getting grasped objects: {e}")
            return set()

    def get_ground_truth_position(self, object_semantic_id):
        if not hasattr(self, 'detector'):
            return None
        
        try:
            all_positions = self.detector.get_all_objects_pos()
            if object_semantic_id in all_positions:
                return np.array(all_positions[object_semantic_id])
        except Exception as e:
            self.debug_message(f"Error getting ground truth position: {e}")
        
        return None

    def clean_noisy_tracks(self, min_detection_frames=5, max_unmatched_ratio=0.7):
        removed_tracks = []
        current_frame = sum(meta.get('missing_frames', 0) == 0 for meta in self.tracking_metadata.values())
        
        for track_id in list(self.tracked_objects.keys()):
            if track_id not in self.tracking_metadata:
                continue
            
            metadata = self.tracking_metadata[track_id]
            should_remove = False
            removal_reason = ""
            
            position_history_len = len(metadata.get('position_history', []))
            if position_history_len < min_detection_frames:
                should_remove = True
                removal_reason = f"too few detections ({position_history_len})"
            
            missing_frames = metadata.get('missing_frames', 0)
            if missing_frames > 0:
                total_frames = position_history_len + missing_frames
                unmatched_ratio = missing_frames / total_frames if total_frames > 0 else 1.0
                
                if unmatched_ratio > max_unmatched_ratio and total_frames >= min_detection_frames:
                    should_remove = True
                    removal_reason = f"high unmatched ratio ({unmatched_ratio:.2f})"
            
            if hasattr(self, 'relations') and self.relations:
                is_mapped = track_id in self.relations.values()
                
                if not is_mapped and position_history_len >= min_detection_frames * 2:
                    should_remove = True
                    removal_reason = "not mapped to any semantic object"
            
            outlier_count = self.detection_outlier_count.get(track_id, 0)
            if outlier_count >= self.max_outlier_frames:
                should_remove = True
                removal_reason = f"excessive outliers ({outlier_count})"
            
            obj_class = self.tracked_objects[track_id]['class']
            expected_count = self.instances_per_label.get(obj_class, 1)
            same_class_tracks = [tid for tid, obj in self.tracked_objects.items() 
                                if obj['class'] == obj_class]
            
            if len(same_class_tracks) > expected_count * 1.5:
                tracks_with_scores = []
                for tid in same_class_tracks:
                    conf = self.tracked_objects[tid].get('conf', 0)
                    missing = self.tracking_metadata[tid].get('missing_frames', 0)
                    score = conf - (missing * 0.1)
                    tracks_with_scores.append((tid, score))
                
                tracks_with_scores.sort(key=lambda x: x[1])
                tracks_to_remove = [t[0] for t in tracks_with_scores[:len(same_class_tracks) - expected_count]]
                
                if track_id in tracks_to_remove:
                    should_remove = True
                    removal_reason = f"excess {obj_class} detected (keeping top {expected_count})"
            
            if should_remove:
                self.debug_message(f"[CLEANUP] Removing noisy track {track_id}: {removal_reason}")
                removed_tracks.append(track_id)
                
                del self.tracked_objects[track_id]
                del self.tracking_metadata[track_id]
                
                if track_id in self.detection_outlier_count:
                    del self.detection_outlier_count[track_id]
                
                if track_id in self.detected_positions:
                    del self.detected_positions[track_id]
        
        if removed_tracks:
            self.debug_message(f"[CLEANUP] Removed {len(removed_tracks)} noisy tracks")
        
        return removed_tracks

    def assign_detections_to_tracks(self, detections, cls_name, iou_threshold=0.3):
        tracked_ids = [tid for tid, obj in self.tracked_objects.items() 
                      if obj['class'] == cls_name]
        
        if len(tracked_ids) == 0:
            return {}, list(range(len(detections)))
        
        cost_matrix = np.zeros((len(detections), len(tracked_ids)))
        for i, det in enumerate(detections):
            for j, tid in enumerate(tracked_ids):
                iou = self.compute_iou(det['bbox'], self.tracked_objects[tid]['bbox'])
                
                combined_score = iou
                
                cost_matrix[i, j] = -combined_score
        
        det_indices, track_indices = linear_sum_assignment(cost_matrix)
        
        assignments = {}
        unmatched_detections = list(range(len(detections)))
        
        for det_idx, track_idx in zip(det_indices, track_indices):
            score = -cost_matrix[det_idx, track_idx]
            matched = score >= iou_threshold
            if not matched:
                tracked_id = tracked_ids[track_idx]
                det_pos = det.get('position')
                track_pos = self.tracked_objects[tracked_id].get('position')
                if det_pos is not None and track_pos is not None:
                    dist_3d = np.linalg.norm(np.array(det_pos) - np.array(track_pos))
                    if dist_3d < 0.08:
                        matched = True
            if matched:
                tracked_id = tracked_ids[track_idx]
                
                det = detections[det_idx]
                if not self.is_detection_outlier(tracked_id, det['bbox'], det['position']):
                    assignments[det_idx] = tracked_id
                    unmatched_detections.remove(det_idx)
                    self.detection_outlier_count[tracked_id] = 0
                else:
                    self.detection_outlier_count[tracked_id] = \
                        self.detection_outlier_count.get(tracked_id, 0) + 1
        
        return assignments, unmatched_detections
    
    def is_object_grasped(self, track_id):
        semantic_id = self.map_id_semantic.get(track_id)
        
        if semantic_id is None:
            return False
        
        grasped_objects = self.get_grasped_objects()
        return semantic_id in grasped_objects

    def estimate_undetected_object_position(self, track_id, ee_pos, image_shape):
        metadata = self.tracking_metadata[track_id]
        last_pos = np.array(metadata['last_position'])
        last_velocity = np.array(metadata['last_velocity'])
        missing_frames = metadata['missing_frames']
        
        if self.is_object_grasped(track_id):
            metadata['grasped'] = True
            self.debug_message(f"  -> [GRASP] {track_id} using ee pos (grasped)")
            
            bbox_2d = self.project_3d_to_2d_approximate(ee_pos, image_shape)
            return {
                'position_3d': ee_pos,
                'bbox_center_2d': bbox_2d
            }
        
        if metadata['grasped'] and not self.is_object_grasped(track_id):
            metadata['grasped'] = False
            self.debug_message(f"  -> [RELEASE] {track_id} was released")

        bbox_2d = self.project_3d_to_2d_approximate(last_pos, image_shape)
        self.debug_message(f"  -> [STATIC] {track_id} keeping last position")
        
        return {
            'position_3d': last_pos,
            'bbox_center_2d': bbox_2d
        }

    def yolo_estimate(self, image1, image2, save_video=False, cubes_obs=None, ee_pos=None, conf_threshold=0.7, max_missing_frames=10, render=False, sim=None):
        cubes_predicted_xyz = {}

        try:
            image1 = cv2.resize(image1, (256, 256))
        except Exception as e:
            self.debug_message("Error resizing image: ", e, image1.shape, image1.dtype)
        try:
            image2 = cv2.resize(image2, (256, 256))
        except Exception as e:
            self.debug_message("Error resizing image2: ", e, image2.shape, image2.dtype)
        
        image1 = cv2.flip(image1, 0)
        image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
        ogi_image = image1.copy()
        predictions1 = self.yolo_model.predict(image1, verbose=False, device=self.device)[0]
        
        image2 = cv2.flip(image2, 0)
        image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)
        predictions2 = self.yolo_model.predict(image2, verbose=False, device=self.device)[0]

        if not isinstance(image1, np.ndarray):
            image1 = np.array(image1)
        if image2 is not None and not isinstance(image2, np.ndarray):
            image2 = np.array(image2)

        if not hasattr(self, 'tracking_metadata'):
            self.tracking_metadata = {}

        # STEP 1: Collect all detections
        detections_by_class = {}
        high_conf_count = {}
        
        for pred in predictions1.boxes:
            cls_id = int(pred.cls)
            cls = self.yolo_model.names[cls_id]
            x, y, w, h = pred.xywhn.tolist()[0]
            conf = float(pred.conf)
            
            x = int(x * image1.shape[1])
            y = int(y * image1.shape[0])
            w = int(w * image1.shape[1])
            h = int(h * image1.shape[0])
            
            if cls not in detections_by_class:
                detections_by_class[cls] = []
                high_conf_count[cls] = 0
            
            if conf >= conf_threshold:
                high_conf_count[cls] += 1
            
            x_cam2, y_cam2, w_cam2, h_cam2, conf_cam2 = 0, 0, 0, 0, 0
            for pred2 in predictions2.boxes:
                cls_id2 = int(pred2.cls)
                if cls_id2 == cls_id:
                    x2, y2, w2, h2 = pred2.xywhn.tolist()[0]
                    conf2 = float(pred2.conf)
                    
                    x_cam2 = int(x2 * image2.shape[1])
                    y_cam2 = int(y2 * image2.shape[0])
                    w_cam2 = int(w2 * image2.shape[1])
                    h_cam2 = int(h2 * image2.shape[0])
                    conf_cam2 = conf2
                    break
            
            ground_truth_xyz = None
            if cubes_obs and cls in self.map_id_semantic:
                semantic_id = self.map_id_semantic[cls]
                if semantic_id in cubes_obs:
                    ground_truth_xyz = cubes_obs[semantic_id]

            predicted_xyz = self.pixel_to_world_dual(
                cls_id, x, y, w, h, conf,
                x_cam2, y_cam2, w_cam2, h_cam2, conf_cam2,
                ee_pos[0], ee_pos[1], ee_pos[2],
                sim=sim, image_h=image1.shape[0], image_w=image1.shape[1],
                cls_name=cls,
            )
            
            detections_by_class[cls].append({
                'bbox': [x, y, w, h],
                'conf': conf,
                'position': predicted_xyz,
                'cls_id': cls_id,
                'cam2_bbox': [x_cam2, y_cam2, w_cam2, h_cam2],
                'cam2_conf': conf_cam2,
                'ground_truth': ground_truth_xyz
            })
        
        # STEP 2: Validate detection set
        current_tracked_count = {}
        for tid, obj in self.tracked_objects.items():
            cls = obj['class']
            current_tracked_count[cls] = current_tracked_count.get(cls, 0) + 1
        
        if not self.is_detection_set_valid(detections_by_class, current_tracked_count):
            detections_by_class = {}
        
        # STEP 3: Update instances_per_label
        for cls, count in high_conf_count.items():
            if count > 0:
                current_count = self.instances_per_label.get(cls, 0)
                self.instances_per_label[cls] = max(count, current_count)
        
        # STEP 4: Track which objects were matched
        matched_objects = set()
        
        # STEP 5: Process detections for each class
        for cls, detections in detections_by_class.items():
            n_instances = self.instances_per_label.get(cls, 1)
            
            detections.sort(key=lambda x: x['conf'], reverse=True)
            top_detections = detections[:n_instances]
            
            filtered_detections = [d for d in top_detections if d['conf'] >= conf_threshold]
            if len(filtered_detections) == 0 and len(top_detections) > 0:
                filtered_detections = [top_detections[0]]
            
            top_detections = filtered_detections
            
            assignments, unmatched = self.assign_detections_to_tracks(top_detections, cls)
            
            for det_idx, track_id in assignments.items():
                det = top_detections[det_idx]
                    
                velocity_3d = np.array([0.0, 0.0, 0.0])
                velocity_2d = np.array([0.0, 0.0])
                
                if track_id in self.tracked_objects and 'position' in self.tracked_objects[track_id]:
                    old_pos = np.array(self.tracked_objects[track_id]['position'])
                    new_pos = np.array(det['position'])
                    velocity_3d = new_pos - old_pos
                
                if track_id in self.tracked_objects and 'bbox' in self.tracked_objects[track_id]:
                    old_bbox_center = np.array([self.tracked_objects[track_id]['bbox'][0], 
                                               self.tracked_objects[track_id]['bbox'][1]])
                    new_bbox_center = np.array([det['bbox'][0], det['bbox'][1]])
                    velocity_2d = new_bbox_center - old_bbox_center
                
                self.tracked_objects[track_id]['bbox'] = det['bbox']
                self.tracked_objects[track_id]['position'] = det['position']
                self.tracked_objects[track_id]['conf'] = det['conf']
            
                
                if track_id not in self.tracking_metadata:
                    self.tracking_metadata[track_id] = {
                        'missing_frames': 0,
                        'last_position': det['position'],
                        'last_velocity': velocity_3d,
                        'bbox_velocity': velocity_2d,
                        'grasped': False,
                        'position_history': []
                    }
                else:
                    self.tracking_metadata[track_id]['missing_frames'] = 0
                    self.tracking_metadata[track_id]['last_position'] = det['position']
                    self.tracking_metadata[track_id]['last_velocity'] = velocity_3d
                    self.tracking_metadata[track_id]['bbox_velocity'] = velocity_2d
                    self.tracking_metadata[track_id]['position_history'].append(det['position'])
                    if len(self.tracking_metadata[track_id]['position_history']) > 5:
                        self.tracking_metadata[track_id]['position_history'].pop(0)
                
                matched_objects.add(track_id)
                
                cubes_predicted_xyz[track_id] = det['position']
                
                if save_video and det['ground_truth'] is not None:
                    self.bboxes_centers.append({
                        "object_id": track_id,
                        "px_cam1": det['bbox'][0],
                        "py_cam1": det['bbox'][1],
                        "w_cam1": det['bbox'][2],
                        "h_cam1": det['bbox'][3],
                        "conf_cam1": det['conf'],
                        "cls": cls,
                        "px_cam2": det['cam2_bbox'][0],
                        "py_cam2": det['cam2_bbox'][1],
                        "w_cam2": det['cam2_bbox'][2],
                        "h_cam2": det['cam2_bbox'][3],
                        "conf_cam2": det['cam2_conf'],
                        "ee_x": ee_pos[0] if ee_pos is not None else None,
                        "ee_y": ee_pos[1] if ee_pos is not None else None,
                        "ee_z": ee_pos[2] if ee_pos is not None else None,
                        "world_x": det['ground_truth'][0],
                        "world_y": det['ground_truth'][1],
                        "world_z": det['ground_truth'][2],
                    })
                
                if save_video or render:
                    x, y, w, h = det['bbox']
                    x1, y1 = int(x - w / 2), int(y - h / 2)
                    x2, y2 = int(x + w / 2), int(y + h / 2)
                    cv2.rectangle(image1, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image1, f"{track_id}:{det['conf']:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    if render:
                        cv2.imshow("Tracking", image1)
                        cv2.waitKey(1)
            
            for det_idx in unmatched:
                det = top_detections[det_idx]
                
                if cls not in self.next_object_id:
                    self.next_object_id[cls] = 0
                
                object_id = f"{cls}_{self.next_object_id[cls]}"
                self.next_object_id[cls] += 1
                
                self.tracked_objects[object_id] = {
                    'bbox': det['bbox'],
                    'position': det['position'],
                    'class': cls,
                    'conf': det['conf']
                }
                
                
                self.tracking_metadata[object_id] = {
                    'missing_frames': 0,
                    'last_position': det['position'],
                    'last_velocity': np.array([0.0, 0.0, 0.0]),
                    'bbox_velocity': np.array([0.0, 0.0]),
                    'grasped': False,
                    'position_history': [det['position']]
                }
                
                matched_objects.add(object_id)
                
                cubes_predicted_xyz[object_id] = det['position']
                
                self.debug_message(f"Created new track: {object_id} with conf {det['conf']:.2f}")
                
                if save_video and det['ground_truth'] is not None:
                    self.bboxes_centers.append({
                        "object_id": object_id,
                        "px_cam1": det['bbox'][0],
                        "py_cam1": det['bbox'][1],
                        "w_cam1": det['bbox'][2],
                        "h_cam1": det['bbox'][3],
                        "conf_cam1": det['conf'],
                        "cls": cls,
                        "px_cam2": det['cam2_bbox'][0],
                        "py_cam2": det['cam2_bbox'][1],
                        "w_cam2": det['cam2_bbox'][2],
                        "h_cam2": det['cam2_bbox'][3],
                        "conf_cam2": det['cam2_conf'],
                        "ee_x": ee_pos[0] if ee_pos is not None else None,
                        "ee_y": ee_pos[1] if ee_pos is not None else None,
                        "ee_z": ee_pos[2] if ee_pos is not None else None,
                        "world_x": det['ground_truth'][0],
                        "world_y": det['ground_truth'][1],
                        "world_z": det['ground_truth'][2],
                    })
                
                if save_video or render:
                    x, y, w, h = det['bbox']
                    x1, y1 = int(x - w / 2), int(y - h / 2)
                    x2, y2 = int(x + w / 2), int(y + h / 2)
                    cv2.rectangle(image1, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(image1, f"{object_id}:{det['conf']:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    if render:
                        cv2.imshow("Tracking", image1)
                        cv2.waitKey(1)

        # STEP 6: Unmatched tracked objects
        for track_id in list(self.tracked_objects.keys()):
            if track_id not in matched_objects:
                if track_id not in self.tracking_metadata:
                    self.tracking_metadata[track_id] = {
                        'missing_frames': 1,
                        'last_position': self.tracked_objects[track_id]['position'],
                        'last_velocity': np.array([0.0, 0.0, 0.0]),
                        'bbox_velocity': np.array([0.0, 0.0]),
                        'grasped': False,
                        'position_history': [self.tracked_objects[track_id]['position']]
                    }
                else:
                    self.tracking_metadata[track_id]['missing_frames'] += 1
                
                missing_frames = self.tracking_metadata[track_id]['missing_frames']
                
                outlier_count = self.detection_outlier_count.get(track_id, 0)
                if outlier_count >= self.max_outlier_frames:
                    self.debug_message(f"Object {track_id} marked as lost (outlier for {outlier_count} frames)")
                    continue
                
                if missing_frames <= max_missing_frames:
                    estimation = self.estimate_undetected_object_position(
                        track_id, 
                        ee_pos, 
                        image1.shape
                    )
                    
                    estimated_pos_3d = estimation['position_3d']
                    estimated_bbox_2d = estimation['bbox_center_2d']
                    
                    self.tracked_objects[track_id]['position'] = estimated_pos_3d
                    cubes_predicted_xyz[track_id] = estimated_pos_3d
                    
                    self.debug_message(f"Estimating position for {track_id} (missing {missing_frames} frames)")
                    
                    
                    if save_video or render:
                        if estimated_bbox_2d is not None:
                            est_x, est_y = int(estimated_bbox_2d[0]), int(estimated_bbox_2d[1])
                            cv2.circle(image1, (est_x, est_y), 15, (0, 165, 255), 2)
                            cv2.circle(image1, (est_x, est_y), 3, (0, 165, 255), -1)
                            cv2.putText(image1, f"{track_id}:EST", (est_x + 20, est_y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                        
                            if render:
                                cv2.imshow("Tracking", image1)
                                cv2.waitKey(1)
                else:
                    self.debug_message(f"Object {track_id} lost after {missing_frames} frames")
                    pass

        # STEP 7: Periodically clean noisy tracks
        self.count += 1
        if self.count % 10 == 0:
            self.clean_noisy_tracks(min_detection_frames=5, max_unmatched_ratio=0.7)


        if save_video:
            if not hasattr(self, "image_buffer"):
                self.image_buffer = []
            self.image_buffer.append(ogi_image)

        self.detected_positions.update(cubes_predicted_xyz)
        return cubes_predicted_xyz

    def save_video(self, name, output_path="video/output", fps=10, format="jpeg"):
        path_name = f"{output_path}/{name}/"
        os.makedirs(path_name, exist_ok=True)
        if format == "jpeg":
            for idx, frame in enumerate(self.image_buffer):
                cv2.imwrite(f"{path_name}{idx:04d}.jpg", frame)
            self.debug_message(f"Frames saved at {path_name}/XXXX.jpg")
            return
        if not self.image_buffer:
            self.debug_message("No frames to save.")
            return
        
        height, width, _ = self.image_buffer[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in self.image_buffer:
            out.write(frame)

        out.release()
        self.debug_message(f"Video saved at {output_path}")

    def save_csv_yolo(self, output_path="yolo_data.csv"):
        import pandas as pd
        if not self.bboxes_centers:
            self.debug_message("No bounding boxes data to save.")
            return
        
        pd.DataFrame(self.bboxes_centers).to_csv(output_path, index=False)
        self.debug_message(f"YOLO data saved at {output_path}")

    def get_last_known_position(self, semantic_id):
        yolo_id = self.relations.get(semantic_id)
        if yolo_id is not None:
            metadata = getattr(self, 'tracking_metadata', {}).get(yolo_id, {})
            if metadata.get('last_position') is not None:
                return np.asarray(metadata['last_position'])
            tracked = self.tracked_objects.get(yolo_id, {})
            if tracked.get('position') is not None:
                return np.asarray(tracked['position'])

        if hasattr(self, 'last_known_semantic_positions'):
            if semantic_id in self.last_known_semantic_positions:
                return self.last_known_semantic_positions[semantic_id]

        return None

    def resolve_object_position(self, semantic_id, predicted_pos, yolo_id=None):
        if yolo_id is None:
            yolo_id = self.relations.get(semantic_id)

        pos = None
        if yolo_id is not None and yolo_id in predicted_pos:
            pos = np.asarray(predicted_pos[yolo_id], dtype=np.float64)
            # A resting cube cannot move, so a large jump is an occlusion
            # artefact — unless it is large enough that the stored pose was the
            # wrong one, in which case the new detection is the correction.
            if (semantic_id and str(semantic_id).startswith("cube")
                    and hasattr(self, "last_known_semantic_positions")
                    and semantic_id in self.last_known_semantic_positions):
                last = np.asarray(self.last_known_semantic_positions[semantic_id], dtype=np.float64)
                jump = float(np.linalg.norm(pos - last))
                conf2 = float(getattr(self, "_last_det_conf2", {}).get(yolo_id, 0.0))
                if jump > 0.12:
                    self.debug_message(
                        f"  [POS GATE RESET] {semantic_id} jump={jump*1000:.0f}mm "
                        f"accept new={np.round(pos, 4)} (discard last={np.round(last, 4)})"
                    )
                elif jump > 0.04 and conf2 < 0.4:
                    self.debug_message(
                        f"  [POS GATE] {semantic_id} jump={jump*1000:.0f}mm "
                        f"keep={np.round(last, 4)} reject={np.round(pos, 4)}"
                    )
                    pos = last
                elif jump > 0.04:
                    self.debug_message(
                        f"  [POS GATE OVERRIDE] {semantic_id} jump={jump*1000:.0f}mm "
                        f"conf2={conf2:.2f} accept={np.round(pos, 4)}"
                    )
        else:
            pos = self.get_last_known_position(semantic_id)
            if pos is not None and yolo_id is not None and yolo_id not in predicted_pos:
                self.debug_message(
                    f"Warning: Mapped YOLO ID {yolo_id} for {semantic_id} not in predicted positions. "
                    "Using last known position."
                )

        if pos is not None:
            if not hasattr(self, 'last_known_semantic_positions'):
                self.last_known_semantic_positions = {}
            self.last_known_semantic_positions[semantic_id] = pos

        return pos

    def reset_tracking(self, preserve_last_known=False):
        kept = {}
        if preserve_last_known and hasattr(self, "last_known_semantic_positions"):
            # Keep resting-cube memory across gripper home; objects that did not
            # move should not be re-localized from a weak single-view guess.
            for k, v in (self.last_known_semantic_positions or {}).items():
                if str(k).startswith("cube") and v is not None:
                    kept[k] = np.asarray(v, dtype=np.float64).copy()
        self.tracked_objects = {}
        self.tracking_metadata = {}
        self.next_object_id = {}
        self.instances_per_label = {}
        self.detection_outlier_count = {}
        self.last_known_semantic_positions = kept
        self.relations = {}
        self.map_id_semantic = {}
        self._skill_snapshot_pos = None
        self.count = 0
        self.debug_message(
            f"Tracking data reset (preserved_last_known={list(kept.keys())})"
        )
    
    def set_tracking_data(self, tracking_data_dict):
        self.tracked_objects = tracking_data_dict.get('tracked_objects', {})
        self.tracking_metadata = tracking_data_dict.get('tracking_metadata', {})
        self.instances_per_label = tracking_data_dict.get('instances_per_label', {})
        self.detection_outlier_count = tracking_data_dict.get('detection_outlier_count', {})
        self.next_object_id = tracking_data_dict.get('next_object_id', {})
        self.relations = tracking_data_dict.get('relations', {})
        self.map_id_semantic = tracking_data_dict.get('map_id_semantic', {})
        self.last_known_semantic_positions = tracking_data_dict.get(
            'last_known_semantic_positions', {})
        self._skill_snapshot_pos = tracking_data_dict.get('_skill_snapshot_pos', None)

    def get_tracking_data(self):
        if not hasattr(self, 'tracking_metadata'):
            return {}
        return {
            'tracked_objects': self.tracked_objects,
            'tracking_metadata': self.tracking_metadata,
            'instances_per_label': self.instances_per_label,
            'detection_outlier_count': self.detection_outlier_count,
            'next_object_id': self.next_object_id,
            'relations': getattr(self, 'relations', {}),
            'map_id_semantic': getattr(self, 'map_id_semantic', {}),
            'last_known_semantic_positions': getattr(
                self, 'last_known_semantic_positions', {}),
            '_skill_snapshot_pos': getattr(self, '_skill_snapshot_pos', None),
        }

    def action_obs_mapping(self, obs, action_step="PickPlace", relative=False):
        index_obs = {"gripper_pos": (0,3), "aperture": (3,4), "obj_to_pick_pos": (4,7), "place_to_drop_pos": (7,10), "gripper_z": (2,3), "obj_to_pick_z": (6,7), "place_to_drop_z": (9,10)}

        oracle = np.array([])
        if action_step == "PickPlace":
            if relative:
                oracle = np.concatenate([obs[index_obs["obj_to_pick_pos"][0]:index_obs["obj_to_pick_pos"][1]] - obs[index_obs["gripper_pos"][0]:index_obs["gripper_pos"][1]], obs[index_obs["aperture"][0]:index_obs["aperture"][1]], obs[index_obs["place_to_drop_pos"][0]:index_obs["place_to_drop_pos"][1]] - obs[index_obs["gripper_pos"][0]:index_obs["gripper_pos"][1]]])
            else:
                oracle = np.concatenate([obs[index_obs["obj_to_pick_pos"][0]:index_obs["obj_to_pick_pos"][1]], obs[index_obs["aperture"][0]:index_obs["aperture"][1]], obs[index_obs["place_to_drop_pos"][0]:index_obs["place_to_drop_pos"][1]]])
        elif action_step == "ReachPick":
            if relative:
                oracle = np.concatenate([obs[index_obs["obj_to_pick_pos"][0]:index_obs["obj_to_pick_pos"][1]] - obs[index_obs["gripper_pos"][0]:index_obs["gripper_pos"][1]]])
            else:
                oracle = obs[index_obs["obj_to_pick_pos"][0]:index_obs["obj_to_pick_pos"][1]]
        elif action_step == "Grasp" or action_step == "Pick":
            if relative:
                oracle = np.concatenate([obs[index_obs["obj_to_pick_z"][0]:index_obs["obj_to_pick_z"][1]] - obs[index_obs["gripper_z"][0]:index_obs["gripper_z"][1]], obs[index_obs["aperture"][0]:index_obs["aperture"][1]]])
            else:
                oracle = np.concatenate([obs[index_obs["obj_to_pick_z"][0]:index_obs["obj_to_pick_z"][1]], obs[index_obs["aperture"][0]:index_obs["aperture"][1]]])
        elif action_step == "ReachDrop":
            if relative:
                oracle = np.concatenate([obs[index_obs["place_to_drop_pos"][0]:index_obs["place_to_drop_pos"][1]] - obs[index_obs["gripper_pos"][0]:index_obs["gripper_pos"][1]]])
            else:
                oracle = obs[index_obs["place_to_drop_pos"][0]:index_obs["place_to_drop_pos"][1]]
        elif action_step == "Drop":
            if relative:
                oracle = np.concatenate([obs[index_obs["place_to_drop_z"][0]:index_obs["place_to_drop_z"][1]] - obs[index_obs["gripper_z"][0]:index_obs["gripper_z"][1]], obs[index_obs["aperture"][0]:index_obs["aperture"][1]]])
            else:
                oracle = np.concatenate([obs[index_obs["place_to_drop_z"][0]:index_obs["place_to_drop_z"][1]], obs[index_obs["aperture"][0]:index_obs["aperture"][1]]])
        else:
            oracle = obs
        return oracle

    def prepare_obs(self, obs, action_step="PickPlace"):
        obs_dim = {"PickPlace": 7, "ReachPick": 3, "Grasp": 2, "ReachDrop": 3, "Drop": 2, "Pick": 2}
        if action_step not in obs_dim.keys():
            return obs
        returned_obs = np.zeros((len(obs), obs_dim[action_step]))
        for j, env_j_obs in enumerate(obs):
            obs_policy = self.action_obs_mapping(env_j_obs, action_step=action_step, relative=False)
            returned_obs[j] = obs_policy
        return returned_obs

    def get_object_obs(self, env, objects_pos, predicted_pos, obj_to_pick, place_to_drop, relative_obs=False):
        gripper_pos = objects_pos["gripper"]
        try:
            from robot_utils import gripper_aperture
            aperture = gripper_aperture(env.sim)
        except Exception:
            left_finger_pos = np.asarray(env.sim.data.body_xpos[env.sim.model.body_name2id("gripper0_leftfinger")])
            right_finger_pos = np.asarray(env.sim.data.body_xpos[env.sim.model.body_name2id("gripper0_rightfinger")])
            aperture = np.linalg.norm(left_finger_pos - right_finger_pos)

        obj_to_pick_yolo_id = self.relations.get(obj_to_pick, None)
        place_to_drop_yolo_id = self.relations.get(place_to_drop, None)

        if obj_to_pick_yolo_id is None and self.warnings["obj_to_pick"]:
            self.debug_message(f"Warning: No YOLO prediction matched for object to pick: {obj_to_pick}")
            self.warnings["obj_to_pick"] = False
        if place_to_drop_yolo_id is None and self.warnings["place_to_drop"]:
            self.debug_message(f"Warning: No YOLO prediction matched for place to drop: {place_to_drop}")
            self.warnings["place_to_drop"] = False

        obj_to_pick_pos = self.resolve_object_position(obj_to_pick, predicted_pos, obj_to_pick_yolo_id)
        place_to_drop_pos = self.resolve_object_position(place_to_drop, predicted_pos, place_to_drop_yolo_id)

        def _static_fixture(name):
            return name is not None and str(name).startswith("peg")

        if obj_to_pick_pos is None:
            if not self.use_yolo and obj_to_pick in objects_pos:
                obj_to_pick_pos = np.asarray(objects_pos[obj_to_pick])
            elif _static_fixture(obj_to_pick) and obj_to_pick in objects_pos:
                obj_to_pick_pos = np.asarray(objects_pos[obj_to_pick])
            else:
                obj_to_pick_pos = np.asarray(gripper_pos, dtype=np.float64)
                self.debug_message(
                    f"  [POS] missing detection for pick target {obj_to_pick}; using gripper pos"
                )
        if place_to_drop_pos is None:
            if not self.use_yolo and place_to_drop in objects_pos:
                place_to_drop_pos = np.asarray(objects_pos[place_to_drop])
            elif _static_fixture(place_to_drop) and place_to_drop in objects_pos:
                place_to_drop_pos = np.asarray(objects_pos[place_to_drop])
            elif place_to_drop in objects_pos and not str(place_to_drop).startswith("cube"):
                place_to_drop_pos = np.asarray(objects_pos[place_to_drop])
            else:
                place_to_drop_pos = np.asarray(gripper_pos, dtype=np.float64)
                self.debug_message(
                    f"  [POS] missing detection for place target {place_to_drop}; using gripper pos"
                )

        if self.use_yolo:
            gt_pick = np.asarray(objects_pos.get(obj_to_pick))
            gt_drop = np.asarray(objects_pos.get(place_to_drop))
            self.debug_message(
                f"  [POS DEBUG] {obj_to_pick}: yolo_id={obj_to_pick_yolo_id} "
                f"pred={np.round(obj_to_pick_pos, 4)} gt={np.round(gt_pick, 4)} "
                f"err={np.round(np.asarray(obj_to_pick_pos) - gt_pick, 4)}"
            )
            self.debug_message(
                f"  [POS DEBUG] {place_to_drop}: yolo_id={place_to_drop_yolo_id} "
                f"pred={np.round(place_to_drop_pos, 4)} gt={np.round(gt_drop, 4)} "
                f"err={np.round(np.asarray(place_to_drop_pos) - gt_drop, 4)}"
            )
            self.debug_message(f"  [POS DEBUG] gripper={np.round(np.asarray(gripper_pos), 4)}")

        if relative_obs:
            rel_obj_to_pick_pos = gripper_pos - obj_to_pick_pos
            rel_place_to_drop_pos = gripper_pos - place_to_drop_pos
            obs = np.concatenate([gripper_pos, [aperture], rel_obj_to_pick_pos, rel_place_to_drop_pos])
        else:
            obs = np.concatenate([gripper_pos, [aperture], obj_to_pick_pos, place_to_drop_pos])
        return obs

    def map_gripper(self, action):
        action_gripper = action[-1]
        if -0.5 < action_gripper < 0.5:
            action_gripper = np.array([0])
        elif action_gripper <= -0.5:
            action_gripper = np.array([-10.0])
        elif action_gripper >= 0.5:
            action_gripper = np.array([10.0])
        action = np.concatenate([action[:3], action_gripper])
        return action
    
    HANOI_COLOR_TO_PDDL = {
        "blue cube": "cube1",
        "red cube": "cube2",
        "green cube": "cube3",
    }

    def build_object_relations(self, predicted_pos, objects_pos):
        relations = {}
        for yolo_id in predicted_pos.keys():
            cls = yolo_id.rsplit("_", 1)[0]
            pddl_id = self.HANOI_COLOR_TO_PDDL.get(cls)
            if pddl_id is None:
                continue
            if pddl_id not in relations:
                relations[pddl_id] = yolo_id

        cubes_only = {obj_id: pos for obj_id, pos in objects_pos.items()
                      if obj_id != "gripper" and "cube" in obj_id}
        unmapped_pddl = [cid for cid in cubes_only if cid not in relations]
        if unmapped_pddl:
            predicted_objs = [SceneObject(id=obj_id, position=predicted_pos[obj_id])
                              for obj_id in predicted_pos.keys()]
            update_object_metadata(predicted_objs, eps=1e-3)
            sim_objs = [SceneObject(id=obj_id, position=cubes_only[obj_id])
                        for obj_id in cubes_only.keys()]
            update_object_metadata(sim_objs, eps=1e-3)
            rel_match = match_objects_by_relationships(sim_objs, predicted_objs)
            used_yolo = set(relations.values())
            for pddl_id in unmapped_pddl:
                yolo_id = rel_match.get(pddl_id)
                if yolo_id and yolo_id not in used_yolo:
                    relations[pddl_id] = yolo_id
                    used_yolo.add(yolo_id)

        self.debug_message("\n=== Detected-to-PDDL Mapping (color-first) ===")
        for pddl_id in sorted(cubes_only.keys()):
            yolo_id = relations.get(pddl_id)
            if yolo_id:
                self.debug_message(f"{pddl_id}  -->  {yolo_id}")
            else:
                self.debug_message(f"{pddl_id}  -->  (no confident match found)")

        return relations
    
    def update_relations_with_new_detections(self, new_predicted_pos, objects_pos):
        needs_update = False
        
        if not self.relations:
            needs_update = True
        
        current_yolo_ids = set(new_predicted_pos.keys())
        mapped_yolo_ids = set(self.relations.values())
        if current_yolo_ids - mapped_yolo_ids:
            self.debug_message(f"New YOLO tracks detected: {current_yolo_ids - mapped_yolo_ids}")
            needs_update = True
        
        missing_yolo_ids = mapped_yolo_ids - current_yolo_ids
        if missing_yolo_ids:
            self.debug_message(f"Mapped YOLO tracks disappeared: {missing_yolo_ids}")
            needs_update = True
        
        if needs_update:
            new_relations = self.build_object_relations(new_predicted_pos, objects_pos)
            
            if self.relations:
                for pddl_id, yolo_id in self.relations.items():
                    if yolo_id in current_yolo_ids:
                        missing_frames = self.tracking_metadata.get(yolo_id, {}).get('missing_frames', 0)
                        if missing_frames >= 5:
                            continue
                        freshly_mapped = new_relations.get(pddl_id)
                        if freshly_mapped is not None and freshly_mapped != yolo_id:
                            continue
                        if yolo_id in new_relations.values() and new_relations.get(pddl_id) != yolo_id:
                            continue
                        old_pos = self.tracked_objects.get(yolo_id, {}).get('position')
                        new_pos = new_predicted_pos.get(yolo_id)
                        if old_pos is not None and new_pos is not None:
                            dist = np.linalg.norm(np.array(new_pos) - np.array(old_pos))
                            if dist < 0.05:
                                new_relations[pddl_id] = yolo_id
            
            self.relations = new_relations
            self.update_yolo_to_pddl_mapping()

    def correct_grasped_object_positions(self, predicted_pos, ee_pos, image_shape, only_pddl_id=None):
        if not self.map_id_semantic:
            return predicted_pos
        
        grasped_objects = self.get_grasped_objects()
        self.debug_message(f"  -> Currently grasped objects (PDDL): {grasped_objects}")
        
        for yolo_id, pddl_id in self.map_id_semantic.items():
            # Only latch EE onto the skill's pick target. Stacked supports often
            # briefly report grasped() when the gripper closes on the top cube;
            # writing EE into their last_known poisons later picks.
            if only_pddl_id is not None and pddl_id != only_pddl_id:
                continue
            if pddl_id in grasped_objects:
                self.debug_message(f"  -> [GRASP CORRECTION] {yolo_id} (PDDL: {pddl_id}) is grasped")
                
                predicted_pos[yolo_id] = ee_pos
                
                if yolo_id in self.tracked_objects:
                    self.tracked_objects[yolo_id]['position'] = ee_pos
                    self.tracked_objects[yolo_id]['conf'] = 1.0
                    
                    bbox_2d = self.project_3d_to_2d_approximate(ee_pos, image_shape)
                    self.tracked_objects[yolo_id]['bbox'] = [bbox_2d[0], bbox_2d[1], 
                                                            self.tracked_objects[yolo_id]['bbox'][2],
                                                            self.tracked_objects[yolo_id]['bbox'][3]]
                else:
                    self.debug_message(f"  -> [GRASP ADD] Adding undetected grasped object {yolo_id}")
                    bbox_2d = self.project_3d_to_2d_approximate(ee_pos, image_shape)
                    
                    if yolo_id in self.tracking_metadata:
                        last_bbox = self.tracking_metadata[yolo_id].get('last_bbox', [0, 0, 50, 50])
                        bbox_w, bbox_h = last_bbox[2], last_bbox[3]
                    else:
                        bbox_w, bbox_h = 50, 50
                    
                    self.tracked_objects[yolo_id] = {
                        'bbox': [bbox_2d[0], bbox_2d[1], bbox_w, bbox_h],
                        'position': ee_pos,
                        'class': yolo_id.rsplit('_', 1)[0],
                        'conf': 1.0,
                        'grasped': True
                    }
                
                if yolo_id in self.tracking_metadata:
                    self.tracking_metadata[yolo_id]['last_position'] = ee_pos
                    self.tracking_metadata[yolo_id]['grasped'] = True
                    self.tracking_metadata[yolo_id]['missing_frames'] = 0
                    
                    if yolo_id in self.tracked_objects:
                        self.tracking_metadata[yolo_id]['last_bbox'] = self.tracked_objects[yolo_id]['bbox']
                else:
                    self.tracking_metadata[yolo_id] = {
                        'missing_frames': 0,
                        'last_position': ee_pos,
                        'last_velocity': np.array([0.0, 0.0, 0.0]),
                        'bbox_velocity': np.array([0.0, 0.0]),
                        'grasped': True,
                        'position_history': [ee_pos],
                        'last_bbox': [bbox_2d[0], bbox_2d[1], 50, 50]
                    }

                if not hasattr(self, 'last_known_semantic_positions'):
                    self.last_known_semantic_positions = {}
                self.last_known_semantic_positions[pddl_id] = np.asarray(ee_pos)
        
        return predicted_pos

    def execute(self, env, observations, n_act, symgoal, task_goals=None, render=False):
        self.warnings = {"obj_to_pick": True, "place_to_drop": True}
        self.image_buffer = []
        self.detected_positions = {}
        horizon = self.horizon if self.horizon is not None else 50
        self.debug_message("\tTask goal: ", symgoal)

        # Snapshot policy:
        # - ReachPick / ReachDrop: always take a fresh detection (target may have moved).
        # - Pick / Drop: reuse the reach snapshot — re-detecting under the gripper is
        #   unreliable (occlusion / extreme close-up), and Z must stay stable for grasp.
        if self.id in ("ReachPick", "ReachDrop"):
            self._skill_snapshot_pos = None
        elif not hasattr(self, "_skill_snapshot_pos"):
            self._skill_snapshot_pos = None

        step_executor = 0
        done = False
        success = False
        self.debug_message("\tStarting executor for step: ", self.id)
        self._beta_hits = 0

        while not done:
            anomaly_safe = False
            max_shift_threshold = 0.1
            while not anomaly_safe:
                processed_obs = []
                
                for obs_num, observation in enumerate(observations):
                    if self.use_yolo or self.save_data:
                        objects_pos = observation["objects_pos"]
                        agentview_image = np.array(observation["agentview_image"].reshape((self.image_size, self.image_size, 3)), dtype=np.uint8)
                        wrist_image = np.array(observation["robot0_eye_in_hand_image"].reshape((self.image_size, self.image_size, 3)), dtype=np.uint8)
                        ee_pos = observation["robot0_eef_pos"]
                        
                        if self._skill_snapshot_pos is None and obs_num == 0:
                            predicted_cubes_xyz, relations = self.detect_cubes_simple(
                                agentview_image, wrist_image, ee_pos,
                                conf_threshold=0.8,
                                sim=getattr(env, "sim", None),
                                render=render,
                            )
                            if relations:
                                self.relations = relations
                                self.map_id_semantic = {y: p for p, y in relations.items()}
                            self._skill_snapshot_pos = copy.deepcopy(predicted_cubes_xyz)
                            # Cache every detected cube so later place-on-cube can
                            # fall back if a re-detect is occluded/noisy.
                            if not hasattr(self, "last_known_semantic_positions"):
                                self.last_known_semantic_positions = {}
                            for pddl_id, yolo_id in (relations or self.relations or {}).items():
                                if yolo_id not in predicted_cubes_xyz:
                                    continue
                                new = np.asarray(predicted_cubes_xyz[yolo_id], dtype=np.float64)
                                self.last_known_semantic_positions[pddl_id] = new
                            # If the skill target is still mono-only, refresh on the
                            # next policy step (keep this batch's snapshot intact —
                            # n_obs>>1 would otherwise see empty poses).
                            skill_tid = None
                            if self.id == "ReachPick" and symgoal and symgoal[0]:
                                skill_tid = symgoal[0]
                            elif (self.id == "ReachDrop" and symgoal and symgoal[1]
                                  and str(symgoal[1]).startswith("cube")):
                                skill_tid = symgoal[1]
                            skill_yolo = (relations or self.relations or {}).get(skill_tid) if skill_tid else None
                            skill_c2 = getattr(self, "_last_det_conf2", {}).get(skill_yolo, 0.0) if skill_yolo else 0.0
                            self.debug_message(
                                f"  [OPERATOR SNAPSHOT] target={symgoal} "
                                f"relations={self.relations} "
                                f"pos_keys={list(predicted_cubes_xyz.keys())} c2={skill_c2:.2f}"
                            )
                            if self.use_yolo and symgoal:
                                tid = symgoal[0]
                                yid = (relations or self.relations or {}).get(tid)
                                if yid and yid in predicted_cubes_xyz:
                                    print(f"\t  [YOLO] snapshot {tid} {np.round(np.asarray(predicted_cubes_xyz[yid]), 4)} "
                                          f"c2={getattr(self, '_last_det_conf2', {}).get(yid, 0):.2f} "
                                          f"src={getattr(self, '_last_det_source', {}).get(yid, '?')}")

                        elif self.save_data and obs_num == 0:
                            predicted_cubes_xyz = self.yolo_estimate(
                                image1=agentview_image,
                                image2=wrist_image,
                                save_video=True,
                                cubes_obs={},
                                ee_pos=ee_pos,
                                conf_threshold=0.8,
                                max_missing_frames=200,
                                render=render,
                                sim=getattr(env, "sim", None)
                            )
                            if predicted_cubes_xyz:
                                self.update_relations_with_new_detections(predicted_cubes_xyz, objects_pos)
                        else:
                            predicted_cubes_xyz = copy.deepcopy(self._skill_snapshot_pos or {})
                            if render and getattr(self, "_last_yolo_viz", None) is not None:
                                cv2.imshow("YOLO Detections", self._last_yolo_viz)
                                cv2.waitKey(1)

                        predicted_cubes_xyz = self.correct_grasped_object_positions(
                            predicted_cubes_xyz,
                            ee_pos,
                            image_shape=agentview_image.shape,
                            only_pddl_id=symgoal[0] if symgoal else None,
                        )

                        obs = self.get_object_obs(env, objects_pos, predicted_cubes_xyz,
                                                symgoal[0], symgoal[1], relative_obs=self.oracle)
                    else:
                        objects_pos = observation["objects_pos"]
                        obs = self.get_object_obs(env, objects_pos, {}, 
                                                symgoal[0], symgoal[1], relative_obs=self.oracle)
                        
                    processed_obs.append(obs)
                
                processed_obs = np.array(processed_obs)
                if self.oracle:
                    processed_obs = self.prepare_obs(processed_obs, action_step=self.id)

                anomaly_detected = False
                anomaly_indices = []
        
                for i in range(len(processed_obs) - 1):
                    obs_current = processed_obs[i]
                    obs_next = processed_obs[i + 1]
                    
                    diff = np.abs(obs_next - obs_current)
                    max_diff = np.max(diff)
                    
                    if max_diff > max_shift_threshold:
                        anomaly_detected = True
                        anomaly_indices.append(i)
                        self.debug_message(f"  Max shift: {max_diff:.2f} mm (threshold: {max_shift_threshold:.2f})")
                        self.debug_message(f"  Obs {i}: {obs_current}")
                        self.debug_message(f"  Obs {i+1}: {obs_next}")
                        self.debug_message(f"  Diff: {diff}")
                        processed_obs[i] = obs_next.copy()
                
                if len(anomaly_indices) > 2:
                    self.debug_message(
                        f"[RECOVERY] Getting {len(observations)} fresh observations"
                    )
                    for _ in range(len(observations)):
                        obs = env._get_observations()
                        objects_pos = self.detector.get_all_objects_pos()
                        obs['objects_pos'] = objects_pos
                        processed_obs.append(obs)
                        processed_obs.pop(0)
                else:
                    anomaly_safe = True
            processed_obs = np.array([processed_obs])
            # Policies are trained with n_obs_steps (typically 4). The eval
            # buffer may be longer; condition on the most recent To frames so
            # the policy sees current state rather than stale history.
            to = int(getattr(self.model, "n_obs_steps", processed_obs.shape[1]))
            if processed_obs.shape[1] > to:
                processed_obs = processed_obs[:, -to:]
            np_obs_dict = {'obs': processed_obs.astype(np.float32)}
            obs_dict = dict_apply(np_obs_dict, 
                lambda x: torch.from_numpy(x).to(device=self.device))
            
            self.debug_message(processed_obs.shape)
            with torch.no_grad():
                action_dict = self.model.predict_action(obs_dict)
            
            np_action_dict = dict_apply(action_dict,
                lambda x: x.detach().to('cpu').numpy())
            actions = np_action_dict['action']
            
            if len(actions[0][0]) < 4:
                for index in self.nulified_action_indexes:
                    fill = self.nulified_action_values.get(index, 0.0)
                    actions = np.insert(actions, index, fill, axis=2)

            i_act = 0
            success = False
            for action in actions[0]:
                action = self.map_gripper(action)
                _, _, done, info = env.step(action)
                if render:
                    env.render()
                obs = env._get_observations()
                objects_pos = self.detector.get_all_objects_pos()
                obs['objects_pos'] = objects_pos
                observations.append(obs)
                observations.pop(0)
                state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
                beta_ok = bool(self.Beta(state, symgoal))
                if beta_ok:
                    self._beta_hits = getattr(self, "_beta_hits", 0) + 1
                    settle = 3 if self.id in ("ReachPick", "ReachDrop") else 1
                    if self._beta_hits >= settle:
                        success = True
                        break
                else:
                    self._beta_hits = 0
                if i_act == n_act - 1:
                    break
                i_act += 1
            if done:
                self.debug_message("Environment terminated")
            
            step_executor += 1
            if not success:
                state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
                success = bool(self.Beta(state, symgoal))

            self.debug_message()
            self.debug_message("Checking goal predicates: ")
            self.debug_message(state)
            goal_reached = False
            for predicate in task_goals:
                self.debug_message("Checking predicate: ", predicate)
                predicate_parts = predicate.split(' ')
                predicate_name = predicate_parts[0]
                predicate_args = ','.join(predicate_parts[1:]).replace(' ', '')
                predicate_str = f"{predicate_name}({predicate_args})"
                if not state[predicate_str]:
                    goal_reached = False
                    break
            success = success or goal_reached
            self.debug_message(
                f"Step: {step_executor}, Success: {success}, Goal Reached: {goal_reached}"
            )
            if success:
                done = True
            if step_executor > horizon:
                self.debug_message("Reached executor horizon")
                done = True 
        
        if self.save_data:
            self.save_csv_yolo(output_path=f"{self.id}_dualcam_{self.count_save}.csv")
            self.save_video(name=f"{self.id}_{self.count_save}")
            self.count_save += 1

        return observations, success, goal_reached
