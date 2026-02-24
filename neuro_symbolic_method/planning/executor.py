'''
# authors: Pierrick Lorang
# email: pierrick.lorang@tufts.edu

# This files implements the structure of the executor object used in this paper.

'''
import os
import dill
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
                 oracle=False, 
                 horizon=None, 
                 use_yolo=True, 
                 save_data=False,
                 instances_per_label=None,
                 particle_filter_particles_2d=100,
                 particle_filter_particles_3d=100,
                 max_position_jump=0.10,  # 10cm max jump in 3D world coords
                 max_bbox_jump=20,  # 20 pixels max jump in image space
                 debug=False
                 ):
        super().__init__(id, "RL", Beta)
        self.debug = debug
        self.policy = policy
        self.model = None
        self.nulified_action_indexes = nulified_action_indexes
        self.horizon = horizon
        self.device = torch.device("cpu")
        self.oracle = oracle
        self.use_yolo = use_yolo
        self.save_data = save_data
        self.image_buffer = []
        self.relations = {}  # Maps PDDL semantic IDs -> YOLO track IDs
        self.map_id_semantic = {}  # Maps YOLO track IDs -> PDDL semantic IDs
        self.detected_positions = {}
        self.bboxes_centers = []
        self.count = count
        self.count_save = 0
        
        # Multi-instance tracking
        self.instances_per_label = instances_per_label or {}
        self.tracked_objects = {}
        self.next_object_id = {}
        
        # Noise removal parameters
        self.max_position_jump = max_position_jump  # meters
        self.max_bbox_jump = max_bbox_jump  # pixels
        self.detection_outlier_count = {}  # Track consecutive outlier detections
        self.max_outlier_frames = 50  # Ignore object after 3 consecutive outlier frames 

    # Creates a debug self.debug_message that only self.debug_messages if self.debug is True
    def debug_message(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    def update_yolo_to_pddl_mapping(self):
        """
        Update map_id_semantic to be the inverse of self.relations.
        Maps YOLO track IDs (e.g., 'blue cube_0') to PDDL semantic IDs (e.g., 'cube1')
        """
        if not self.relations:
            self.debug_message("No relations to update mapping from.")
            return
        #print(self.relations.items())
        self.map_id_semantic = {yolo_id: pddl_id for pddl_id, yolo_id in self.relations.items() if yolo_id is not None}
        self.debug_message(f"Updated map_id_semantic: {self.map_id_semantic}")

    def load_policy(self, detector=None, yolo_model=None, regressor_model=None, image_size=256):
        path = self.policy
        payload = torch.load(open(path, 'rb'), pickle_module=dill)
        cfg = payload['cfg']
        cls = TrainDiffusionTransformerLowdimWorkspace
        cfg.policy.num_inference_steps = 7
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



    def pixel_to_world_dual(self, cls_id, px1, py1, w1, h1, conf1, px2, py2, w2, h2, conf2, ee_x, ee_y, ee_z):
        models_dual = self.regressor_model
        reg_x_dual, reg_y_dual, reg_z_dual = models_dual["reg_x"], models_dual["reg_y"], models_dual["reg_z"]

        try:
            features = np.array([[float(cls_id),
                                float(px1), float(py1), float(w1), float(h1), float(conf1),
                                float(px2), float(py2), float(w2), float(h2), float(conf2),
                                float(ee_x), float(ee_y), float(ee_z)]], dtype=np.float64)
            x = reg_x_dual.predict(features)[0]
            y = reg_y_dual.predict(features)[0]
            z = reg_z_dual.predict(features)[0]
        except:
            features = np.array([[
                      float(px1), float(py1), float(w1), float(h1), float(conf1),
                      float(px2), float(py2), float(w2), float(h2), float(conf2),
                      float(ee_x), float(ee_y), float(ee_z)]], dtype=np.float64)
            x = reg_x_dual.predict(features)[0] + 0.03 # small bias correction
            y = reg_y_dual.predict(features)[0]
            z = reg_z_dual.predict(features)[0]
        return x, y, z

    def compute_iou(self, box1, box2):
        """Compute IoU between two bounding boxes [x_center, y_center, w, h]"""
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
        """
        Check if a detection is an outlier based on bbox jump and position jump.
        
        Args:
            track_id: ID of tracked object
            new_bbox: New bounding box [x, y, w, h]
            new_position: New 3D position [x, y, z]
            
        Returns:
            bool: True if detection is an outlier
        """
        if track_id not in self.tracked_objects:
            return False
        
        # Check bbox jump in image space
        if 'bbox' in self.tracked_objects[track_id]:
            old_bbox = self.tracked_objects[track_id]['bbox']
            bbox_center_old = np.array([old_bbox[0], old_bbox[1]])
            bbox_center_new = np.array([new_bbox[0], new_bbox[1]])
            bbox_jump = np.linalg.norm(bbox_center_new - bbox_center_old)
            
            if bbox_jump > self.max_bbox_jump:
                self.debug_message(f"  [OUTLIER] {track_id}: bbox jump {bbox_jump:.1f}px > {self.max_bbox_jump}px")
                return True
        
        # Check position jump in 3D world space
        if 'position' in self.tracked_objects[track_id]:
            old_position = np.array(self.tracked_objects[track_id]['position'])
            position_jump = np.linalg.norm(np.array(new_position) - old_position)
            
            if position_jump > self.max_position_jump:
                self.debug_message(f"  [OUTLIER] {track_id}: position jump {position_jump:.3f}m > {self.max_position_jump}m")
                return True
        
        return False

    def is_detection_set_valid(self, detections_by_class, current_tracked_count):
        """
        Check if the entire detection set is valid (not a sudden drastic change).
        
        Args:
            detections_by_class: Dict of detections by class
            current_tracked_count: Dict of currently tracked object counts by class
            
        Returns:
            bool: True if detection set is valid
        """
        # Check if number of detected classes changed drastically
        detected_classes = set(detections_by_class.keys())
        tracked_classes = set(current_tracked_count.keys())
        
        # If we suddenly lose all objects, it's likely a bad frame
        if len(tracked_classes) > 0 and len(detected_classes) == 0:
            self.debug_message("  [INVALID SET] All objects lost in detection")
            return False
        
        # If we suddenly detect way more objects than we're tracking, be suspicious
        for cls in detected_classes:
            detected_count = len(detections_by_class[cls])
            tracked_count = current_tracked_count.get(cls, 0)
            
            if tracked_count > 0 and detected_count > tracked_count * 2:
                self.debug_message(f"  [INVALID SET] Detected {detected_count} {cls} vs tracking {tracked_count}")
                return False
        
        return True

    def update_particle_filter_2d(self, track_id, bbox_center, velocity=None):
        """
        Update or initialize particle filter for a tracked object.
        
        Args:
            track_id: ID of tracked object
            bbox_center: Current bbox center [x, y]
            velocity: Optional velocity [vx, vy]
        """

    def get_particle_filter_estimate_2d(self, track_id):
        """
        Get particle filter estimate for bbox center.
        
        Args:
            track_id: ID of tracked object
            
        Returns:
            Estimated bbox center [x, y] or None
        """
        if track_id in self.particle_filters_2d:
            return self.particle_filters_2d[track_id].get_estimate()
        return None

    def update_particle_filter_3d(self, track_id, position_3d, velocity=None):
        """
        Update or initialize particle filter for 3D position of a tracked object.
        
        Args:
            track_id: ID of tracked object
            position_3d: Current 3D position [x, y, z]
            velocity: Optional velocity [vx, vy, vz]
        """

    def get_particle_filter_estimate_3d(self, track_id):
        """
        Get particle filter estimate for 3D position.
        
        Args:
            track_id: ID of tracked object
            
        Returns:
            Estimated 3D position [x, y, z] or None
        """
        if track_id in self.particle_filters_3d:
            return self.particle_filters_3d[track_id].get_estimate()
        return None

    def project_3d_to_2d_approximate(self, position_3d, image_shape):
        """
        Approximate projection from 3D world coordinates to 2D image coordinates.
        This is a rough approximation - ideally you'd use camera intrinsics.
        
        Args:
            position_3d: 3D position [x, y, z]
            image_shape: Image shape (height, width)
            
        Returns:
            Approximate 2D position [x, y] in image space
        """
        # Simple linear approximation (you may need to adjust based on your camera setup)
        # Assuming camera is looking down at workspace
        # Map x,y world coords to image coords
        
        # This is a placeholder - adjust based on your actual camera calibration
        world_x, world_y, world_z = position_3d
        
        # Rough mapping (adjust scale and offset based on your setup)
        img_x = int((world_x + 0.5) * image_shape[1])  # Assuming world center at x=0
        img_y = int((world_y + 0.5) * image_shape[0])  # Assuming world center at y=0
        
        # Clamp to image bounds
        img_x = np.clip(img_x, 0, image_shape[1] - 1)
        img_y = np.clip(img_y, 0, image_shape[0] - 1)
        
        return np.array([img_x, img_y])

    def get_grasped_objects(self):
        """
        Get list of currently grasped objects from detector.
        
        Returns:
            Set of object IDs that are grasped (e.g., {'cube1', 'cube2'})
        """
        if not hasattr(self, 'detector'):
            return set()
        
        try:
            groundings = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            grasped_objects = set()
            
            for predicate, value in groundings.items():
                if 'grasped' in predicate and value:
                    # Extract object name from predicate like "grasped(cube1)"
                    obj_name = predicate.split('(')[1].split(')')[0]
                    grasped_objects.add(obj_name)
            
            return grasped_objects
        except Exception as e:
            self.debug_message(f"Error getting grasped objects: {e}")
            return set()

    def get_ground_truth_position(self, object_semantic_id):
        """
        Get ground truth position from detector for a semantic object ID.
        
        Args:
            object_semantic_id: Semantic ID like 'cube1', 'cube2'
            
        Returns:
            3D position [x, y, z] or None
        """
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
        """
        Remove tracked objects that are likely noise based on several heuristics.
        
        Args:
            min_detection_frames: Minimum number of frames an object must be detected to be kept
            max_unmatched_ratio: Maximum ratio of missing frames to total frames before removal
        
        Returns:
            List of removed track IDs
        """
        removed_tracks = []
        current_frame = sum(meta.get('missing_frames', 0) == 0 for meta in self.tracking_metadata.values())
        
        for track_id in list(self.tracked_objects.keys()):
            if track_id not in self.tracking_metadata:
                continue
            
            metadata = self.tracking_metadata[track_id]
            should_remove = False
            removal_reason = ""
            
            # Heuristic 1: Too few detection frames (likely spurious detection)
            position_history_len = len(metadata.get('position_history', []))
            if position_history_len < min_detection_frames:
                should_remove = True
                removal_reason = f"too few detections ({position_history_len})"
            
            # Heuristic 2: High ratio of missing frames (object disappeared)
            missing_frames = metadata.get('missing_frames', 0)
            if missing_frames > 0:
                # Estimate total frames this object has been tracked
                total_frames = position_history_len + missing_frames
                unmatched_ratio = missing_frames / total_frames if total_frames > 0 else 1.0
                
                if unmatched_ratio > max_unmatched_ratio and total_frames >= min_detection_frames:
                    should_remove = True
                    removal_reason = f"high unmatched ratio ({unmatched_ratio:.2f})"
            
            # Heuristic 3: Object not in relations mapping (no semantic match found)
            if hasattr(self, 'relations') and self.relations:
                # Check if this track_id appears in the relations mapping
                is_mapped = track_id in self.relations.values()
                
                # If object has been around long enough but still not mapped, likely noise
                if not is_mapped and position_history_len >= min_detection_frames * 2:
                    should_remove = True
                    removal_reason = "not mapped to any semantic object"
            
            # Heuristic 4: Excessive outlier count
            outlier_count = self.detection_outlier_count.get(track_id, 0)
            if outlier_count >= self.max_outlier_frames:
                should_remove = True
                removal_reason = f"excessive outliers ({outlier_count})"
            
            # Heuristic 5: Too many objects of the same class detected
            obj_class = self.tracked_objects[track_id]['class']
            expected_count = self.instances_per_label.get(obj_class, 1)
            same_class_tracks = [tid for tid, obj in self.tracked_objects.items() 
                                if obj['class'] == obj_class]
            
            if len(same_class_tracks) > expected_count * 1.5:  # 50% tolerance
                # Remove tracks with lowest confidence or most missing frames
                tracks_with_scores = []
                for tid in same_class_tracks:
                    conf = self.tracked_objects[tid].get('conf', 0)
                    missing = self.tracking_metadata[tid].get('missing_frames', 0)
                    score = conf - (missing * 0.1)  # Penalize missing frames
                    tracks_with_scores.append((tid, score))
                
                tracks_with_scores.sort(key=lambda x: x[1])
                tracks_to_remove = [t[0] for t in tracks_with_scores[:len(same_class_tracks) - expected_count]]
                
                if track_id in tracks_to_remove:
                    should_remove = True
                    removal_reason = f"excess {obj_class} detected (keeping top {expected_count})"
            
            if should_remove:
                self.debug_message(f"[CLEANUP] Removing noisy track {track_id}: {removal_reason}")
                removed_tracks.append(track_id)
                
                # Clean up all related data structures
                del self.tracked_objects[track_id]
                del self.tracking_metadata[track_id]
                
                if track_id in self.detection_outlier_count:
                    del self.detection_outlier_count[track_id]
                
                # Remove from detected_positions if present
                if track_id in self.detected_positions:
                    del self.detected_positions[track_id]
        
        if removed_tracks:
            self.debug_message(f"[CLEANUP] Removed {len(removed_tracks)} noisy tracks")
        
        return removed_tracks

    def assign_detections_to_tracks(self, detections, cls_name, iou_threshold=0.3):
        """
        Assign new detections to existing tracked objects using Hungarian algorithm.
        Incorporates particle filter estimates for better matching.
        
        Args:
            detections: list of dicts with keys 'bbox', 'conf', 'position'
            cls_name: class name (e.g., "blue cube")
            iou_threshold: minimum IoU to consider a match
            
        Returns:
            assignments: dict mapping detection_idx -> tracked_object_id
            unmatched_detections: list of detection indices that weren't matched
        """
        # Get all tracked objects of this class
        tracked_ids = [tid for tid, obj in self.tracked_objects.items() 
                      if obj['class'] == cls_name]
        
        if len(tracked_ids) == 0:
            return {}, list(range(len(detections)))
        
        # Build cost matrix using both IoU and particle filter estimates
        cost_matrix = np.zeros((len(detections), len(tracked_ids)))
        for i, det in enumerate(detections):
            for j, tid in enumerate(tracked_ids):
                # Get IoU score
                iou = self.compute_iou(det['bbox'], self.tracked_objects[tid]['bbox'])
                
                combined_score = iou
                
                cost_matrix[i, j] = -combined_score  # Negative because we minimize
        
        # Use Hungarian algorithm
        det_indices, track_indices = linear_sum_assignment(cost_matrix)
        
        assignments = {}
        unmatched_detections = list(range(len(detections)))
        
        for det_idx, track_idx in zip(det_indices, track_indices):
            score = -cost_matrix[det_idx, track_idx]
            if score >= iou_threshold:
                tracked_id = tracked_ids[track_idx]
                
                # Additional outlier check
                det = detections[det_idx]
                if not self.is_detection_outlier(tracked_id, det['bbox'], det['position']):
                    assignments[det_idx] = tracked_id
                    unmatched_detections.remove(det_idx)
                    # Reset outlier count
                    self.detection_outlier_count[tracked_id] = 0
                else:
                    # Increment outlier count
                    self.detection_outlier_count[tracked_id] = \
                        self.detection_outlier_count.get(tracked_id, 0) + 1
        
        return assignments, unmatched_detections
    
    def is_object_grasped(self, track_id):
        """
        Check if a YOLO-tracked object is currently grasped.
        
        Args:
            track_id: YOLO track ID (e.g., 'blue cube_0')
        
        Returns:
            bool: True if object is grasped
        """
        # Get the PDDL semantic ID for this YOLO track
        semantic_id = self.map_id_semantic.get(track_id)
        
        if semantic_id is None:
            return False
        
        # Check if object is grasped using detector
        grasped_objects = self.get_grasped_objects()
        return semantic_id in grasped_objects

    def estimate_undetected_object_position(self, track_id, ee_pos, image_shape):
        """
        Estimate the position of an object that is not currently detected.
        Uses particle filter, grasp state, and other heuristics.
        
        Args:
            track_id: ID of the tracked object
            ee_pos: Current end effector position
            image_shape: Shape of the image (for projection)
        
        Returns:
            Dictionary with 'position_3d' and 'bbox_center_2d'
        """
        metadata = self.tracking_metadata[track_id]
        last_pos = np.array(metadata['last_position'])
        last_velocity = np.array(metadata['last_velocity'])
        missing_frames = metadata['missing_frames']
        
        # Heuristic 1: Object is grasped - use ee_pos
        if self.is_object_grasped(track_id):
            #print(f"  -> {track_id} is currently grasped")
            metadata['grasped'] = True
            self.debug_message(f"  -> [GRASP] {track_id} using ee pos (grasped)")
            
            # Update particle filter with projected position
            bbox_2d = self.project_3d_to_2d_approximate(ee_pos, image_shape)
            return {
                'position_3d': ee_pos,
                'bbox_center_2d': bbox_2d
            }
        
        # Heuristic 2: Was grasped but no longer - check if near gripper
        if metadata['grasped'] and not self.is_object_grasped(track_id):
            metadata['grasped'] = False
            self.debug_message(f"  -> [RELEASE] {track_id} was released")

        # Heuristic 5: Keep last known position
        bbox_2d = self.project_3d_to_2d_approximate(last_pos, image_shape)
        self.debug_message(f"  -> [STATIC] {track_id} keeping last position")
        
        return {
            'position_3d': last_pos,
            'bbox_center_2d': bbox_2d
        }

    def yolo_estimate(self, image1, image2, save_video=False, cubes_obs=None, ee_pos=None, conf_threshold=0.7, max_missing_frames=10, render=False):
        """
        Enhanced YOLO estimation with particle filter tracking and noise removal.
        """
        cubes_predicted_xyz = {}

        try:
            image1 = cv2.resize(image1, (256, 256))
        except Exception as e:
            self.debug_message("Error resizing image: ", e, image1.shape, image1.dtype)
        try:
            image2 = cv2.resize(image2, (256, 256))
        except Exception as e:
            self.debug_message("Error resizing image2: ", e, image2.shape, image2.dtype)
        
        # Mirror and convert images
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

        # Initialize tracking metadata if not exists
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
            
            # Find matching detection in camera 2
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
                ee_pos[0], ee_pos[1], ee_pos[2]
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
            #self.debug_message("[WARNING] Invalid detection set - using estimation only")
            detections_by_class = {}  # Ignore all detections this frame
        
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
            
            # Sort by confidence and keep top-N
            detections.sort(key=lambda x: x['conf'], reverse=True)
            top_detections = detections[:n_instances]
            
            # Filter by confidence threshold
            filtered_detections = [d for d in top_detections if d['conf'] >= conf_threshold]
            if len(filtered_detections) == 0 and len(top_detections) > 0:
                filtered_detections = [top_detections[0]]
                #self.debug_message(f"Warning: No {cls} detections above {conf_threshold}, using highest conf: {top_detections[0]['conf']:.2f}")
            
            top_detections = filtered_detections
            
            # Assign detections to existing tracks
            assignments, unmatched = self.assign_detections_to_tracks(top_detections, cls)
            
            # Update existing tracks
            for det_idx, track_id in assignments.items():
                det = top_detections[det_idx]
                    
                # Calculate velocities
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
                
                # Update tracked object
                self.tracked_objects[track_id]['bbox'] = det['bbox']
                self.tracked_objects[track_id]['position'] = det['position']
                self.tracked_objects[track_id]['conf'] = det['conf']
            
                
                # Update tracking metadata
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
                
                # Mark as matched
                matched_objects.add(track_id)
                
                # Store in output dict
                cubes_predicted_xyz[track_id] = det['position']
                
                # Save data for analysis
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
                    # Draw detected object (green)
                    x, y, w, h = det['bbox']
                    x1, y1 = int(x - w / 2), int(y - h / 2)
                    x2, y2 = int(x + w / 2), int(y + h / 2)
                    cv2.rectangle(image1, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image1, f"{track_id}:{det['conf']:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    if render:
                        cv2.imshow("Tracking", image1)
                        cv2.waitKey(1)
            
            # Create new tracks for unmatched detections
            for det_idx in unmatched:
                det = top_detections[det_idx]
                
                if cls not in self.next_object_id:
                    self.next_object_id[cls] = 0
                
                object_id = f"{cls}_{self.next_object_id[cls]}"
                self.next_object_id[cls] += 1
                
                # Add to tracked objects
                self.tracked_objects[object_id] = {
                    'bbox': det['bbox'],
                    'position': det['position'],
                    'class': cls,
                    'conf': det['conf']
                }
                
                
                # Initialize tracking metadata
                self.tracking_metadata[object_id] = {
                    'missing_frames': 0,
                    'last_position': det['position'],
                    'last_velocity': np.array([0.0, 0.0, 0.0]),
                    'bbox_velocity': np.array([0.0, 0.0]),
                    'grasped': False,
                    'position_history': [det['position']]
                }
                
                # Mark as matched
                matched_objects.add(object_id)
                
                # Store in output dict
                cubes_predicted_xyz[object_id] = det['position']
                
                self.debug_message(f"Created new track: {object_id} with conf {det['conf']:.2f}")
                
                # Save data for analysis
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
                    # Draw new object (blue)
                    x, y, w, h = det['bbox']
                    x1, y1 = int(x - w / 2), int(y - h / 2)
                    x2, y2 = int(x + w / 2), int(y + h / 2)
                    cv2.rectangle(image1, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(image1, f"{object_id}:{det['conf']:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    if render:
                        cv2.imshow("Tracking", image1)
                        cv2.waitKey(1)

        # STEP 6: Handle unmatched tracked objects (not detected in this frame)
        for track_id in list(self.tracked_objects.keys()):
            if track_id not in matched_objects:
                # Object was not detected this frame
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
                
                # Check if object has been outlier for too long
                outlier_count = self.detection_outlier_count.get(track_id, 0)
                if outlier_count >= self.max_outlier_frames:
                    self.debug_message(f"Object {track_id} marked as lost (outlier for {outlier_count} frames)")
                    continue
                
                if missing_frames <= max_missing_frames:
                    # Estimate position using heuristics and particle filter
                    estimation = self.estimate_undetected_object_position(
                        track_id, 
                        ee_pos, 
                        image1.shape
                    )
                    
                    estimated_pos_3d = estimation['position_3d']
                    estimated_bbox_2d = estimation['bbox_center_2d']
                    
                    # Update position with estimation
                    self.tracked_objects[track_id]['position'] = estimated_pos_3d
                    cubes_predicted_xyz[track_id] = estimated_pos_3d
                    
                    self.debug_message(f"Estimating position for {track_id} (missing {missing_frames} frames)")
                    
                    
                    if save_video or render:
                        # Draw estimated position (orange circle at particle filter estimate)
                        if estimated_bbox_2d is not None:
                            est_x, est_y = int(estimated_bbox_2d[0]), int(estimated_bbox_2d[1])
                            # Draw larger circle for estimated position
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
        if self.count % 10 == 0:  # Clean every 10 frames
            self.clean_noisy_tracks(min_detection_frames=5, max_unmatched_ratio=0.7)


        if save_video:
            if not hasattr(self, "image_buffer"):
                self.image_buffer = []
            #self.image_buffer.append(image1.copy())
            self.image_buffer.append(ogi_image)

        self.detected_positions.update(cubes_predicted_xyz)
        return cubes_predicted_xyz

    def save_video(self, name, output_path="video/output", fps=10, format="jpeg"):
        # path name is output_path/name
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

    def reset_tracking(self):
        """Reset all tracking data. Call this at the start of a new episode."""
        self.tracked_objects = {}
        self.tracking_metadata = {}
        self.next_object_id = {}
        self.instances_per_label = {}
        self.detection_outlier_count = {}
        self.debug_message("Tracking data reset")
    
    def set_tracking_data(self, tracking_data_dict):
        """Sets all the variables related to tracking from an external source."""
        self.tracked_objects = tracking_data_dict.get('tracked_objects', {})
        self.tracking_metadata = tracking_data_dict.get('tracking_metadata', {})
        self.instances_per_label = tracking_data_dict.get('instances_per_label', {})
        self.detection_outlier_count = tracking_data_dict.get('detection_outlier_count', {})
        self.next_object_id = tracking_data_dict.get('next_object_id', {})

    def get_tracking_data(self):
        """Returns all the variables related to tracking for external use."""
        if not hasattr(self, 'tracking_metadata'):
            return {}
        return {
            'tracked_objects': self.tracked_objects,
            'tracking_metadata': self.tracking_metadata,
            'instances_per_label': self.instances_per_label,
            'detection_outlier_count': self.detection_outlier_count,
            'next_object_id': self.next_object_id,
            'instances_per_label': self.instances_per_label
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
                oracle = np.concatenate([obs[index_obs["obj_to_pick_z"][0]:index_obs["obj_to_pick_z"][1]] + 0.01, obs[index_obs["aperture"][0]:index_obs["aperture"][1]]])
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
        left_finger_pos = np.asarray(env.sim.data.body_xpos[env.sim.model.body_name2id("gripper0_left_inner_finger")])
        right_finger_pos = np.asarray(env.sim.data.body_xpos[env.sim.model.body_name2id("gripper0_right_inner_finger")])
        aperture = np.linalg.norm(left_finger_pos - right_finger_pos)#*1000.

        if len(predicted_pos) > 1:

            # Get relationships between predicted objects
            predicted_objs = [SceneObject(id=obj_id, position=predicted_pos[obj_id]) for obj_id in predicted_pos.keys()]
            update_object_metadata(predicted_objs, eps=1e-3)
            # Query example:
            # a = objs[0]
            # b = objs[1]

            # Get relationships between sim objects
            cubes_only = {obj_id: pos for obj_id, pos in objects_pos.items() if obj_id != "gripper" and 'cube' in obj_id}
            sim_objs = [SceneObject(id=obj_id, position=cubes_only[obj_id]) for obj_id in cubes_only.keys()]
            update_object_metadata(sim_objs, eps=1e-3)

            obj_to_pick_yolo_id = self.relations.get(obj_to_pick, None)
            place_to_drop_yolo_id = self.relations.get(place_to_drop, None)

            # if None, self.debug_message warning
            if obj_to_pick_yolo_id is None and self.warnings["obj_to_pick"]:
                self.debug_message(f"Warning: No YOLO prediction matched for object to pick: {obj_to_pick}")
                self.warnings["obj_to_pick"] = False  # Only warn once per episode
            if place_to_drop_yolo_id is None and self.warnings["place_to_drop"]:
                self.debug_message(f"Warning: No YOLO prediction matched for place to drop: {place_to_drop}")
                self.warnings["place_to_drop"] = False  # Only warn once per episode

            if obj_to_pick_yolo_id is not None and obj_to_pick_yolo_id not in predicted_pos:
                self.debug_message(f"Warning: Mapped YOLO ID {obj_to_pick_yolo_id} for object to pick not in predicted positions. Using tracked positions if available.")
                obj_to_pick_pos = self.tracking_metadata.get(obj_to_pick_yolo_id, {}).get('last_position', objects_pos[obj_to_pick])
            else:
                obj_to_pick_pos = predicted_pos[obj_to_pick_yolo_id] if obj_to_pick_yolo_id is not None else objects_pos[obj_to_pick]
            
            if place_to_drop_yolo_id is not None and place_to_drop_yolo_id not in predicted_pos:
                self.debug_message(f"Warning: Mapped YOLO ID {place_to_drop_yolo_id} for place to drop not in predicted positions. Using tracked positions if available.")
                place_to_drop_pos = self.tracking_metadata.get(place_to_drop_yolo_id, {}).get('last_position', objects_pos[place_to_drop])
            else:
                place_to_drop_pos = predicted_pos[place_to_drop_yolo_id] if place_to_drop_yolo_id is not None else objects_pos[place_to_drop]
        else:
            obj_to_pick_pos = objects_pos[obj_to_pick]
            place_to_drop_pos = objects_pos[place_to_drop]

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
        if action_gripper <= -0.5:
            action_gripper = np.array([-0.1])
        elif action_gripper >= 0.5:
            action_gripper = np.array([0.1])
        action = np.concatenate([action[:3], action_gripper])
        return action
    
    def build_object_relations(self, predicted_pos, objects_pos):
        """
        Build relations mapping between YOLO detections and PDDL semantic IDs.
        
        Args:
            predicted_pos: Dict of YOLO track IDs -> 3D positions
            objects_pos: Dict of PDDL semantic IDs -> 3D positions from sim
        
        Returns:
            relations dict: PDDL semantic ID -> YOLO track ID
        """
        # Get relationships between predicted objects
        predicted_objs = [SceneObject(id=obj_id, position=predicted_pos[obj_id]) 
                        for obj_id in predicted_pos.keys()]
        update_object_metadata(predicted_objs, eps=1e-3)
        
        # Get relationships between sim objects
        cubes_only = {obj_id: pos for obj_id, pos in objects_pos.items() 
                    if obj_id != "gripper" and 'cube' in obj_id}
        sim_objs = [SceneObject(id=obj_id, position=cubes_only[obj_id]) 
                    for obj_id in cubes_only.keys()]
        update_object_metadata(sim_objs, eps=1e-3)
        
        # Map predicted positions to object positions based on relationships
        relations = match_objects_by_relationships(sim_objs, predicted_objs)
        
        self.debug_message("\n=== Detected-to-PDDL Mapping (based on relational similarity) ===")
        for pddl_id, yolo_id in relations.items():
            if yolo_id:
                self.debug_message(f"{pddl_id}  -->  {yolo_id}")
            else:
                self.debug_message(f"{pddl_id}  -->  (no confident match found)")
        
        return relations
    
    def update_relations_with_new_detections(self, new_predicted_pos, objects_pos):
        """
        Update relations when new objects are detected or tracking IDs change.
        Only updates if there are significant changes.
        
        Args:
            new_predicted_pos: Dict of current YOLO track IDs -> 3D positions
            objects_pos: Dict of PDDL semantic IDs -> 3D positions from sim
        """
        # Check if we need to rebuild relations
        needs_update = False
        
        # Case 1: No relations exist yet
        if not self.relations:
            needs_update = True
        
        # Case 2: New YOLO tracks appeared
        current_yolo_ids = set(new_predicted_pos.keys())
        mapped_yolo_ids = set(self.relations.values())
        if current_yolo_ids - mapped_yolo_ids:
            self.debug_message(f"New YOLO tracks detected: {current_yolo_ids - mapped_yolo_ids}")
            needs_update = True
        
        # Case 3: Mapped YOLO tracks disappeared (need remapping)
        missing_yolo_ids = mapped_yolo_ids - current_yolo_ids
        if missing_yolo_ids:
            self.debug_message(f"Mapped YOLO tracks disappeared: {missing_yolo_ids}")
            needs_update = True
        
        if needs_update:
            new_relations = self.build_object_relations(new_predicted_pos, objects_pos)
            
            # Merge with existing relations, preferring stable mappings
            if self.relations:
                # Keep stable mappings where YOLO ID still exists
                for pddl_id, yolo_id in self.relations.items():
                    if yolo_id in current_yolo_ids:
                        # Check if position is consistent
                        old_pos = self.tracked_objects.get(yolo_id, {}).get('position')
                        new_pos = new_predicted_pos.get(yolo_id)
                        if old_pos is not None and new_pos is not None:
                            dist = np.linalg.norm(np.array(new_pos) - np.array(old_pos))
                            if dist < 0.05:  # 5cm threshold for stability
                                new_relations[pddl_id] = yolo_id  # Keep stable mapping
            
            self.relations = new_relations
            self.update_yolo_to_pddl_mapping()

    def correct_grasped_object_positions(self, predicted_pos, ee_pos, image_shape):
        """
        After YOLO detection and relations mapping, correct positions of grasped objects.
        Also adds back grasped objects that weren't detected.
        
        Args:
            predicted_pos: Dict of YOLO track IDs -> 3D positions (will be modified in-place)
            ee_pos: Current end effector position
            image_shape: Shape of image for 2D projection
        
        Returns:
            Updated predicted_pos dict
        """
        if not self.map_id_semantic:
            # No mapping available yet
            return predicted_pos
        
        grasped_objects = self.get_grasped_objects()
        self.debug_message(f"  -> Currently grasped objects (PDDL): {grasped_objects}")
        
        # Check all known YOLO tracks
        for yolo_id, pddl_id in self.map_id_semantic.items():
            if pddl_id in grasped_objects:
                self.debug_message(f"  -> [GRASP CORRECTION] {yolo_id} (PDDL: {pddl_id}) is grasped")
                
                # Override position with EE position
                predicted_pos[yolo_id] = ee_pos
                
                # Update tracked object data
                if yolo_id in self.tracked_objects:
                    self.tracked_objects[yolo_id]['position'] = ee_pos
                    self.tracked_objects[yolo_id]['conf'] = 1.0  # High confidence
                    
                    # Update bbox to projected EE position
                    bbox_2d = self.project_3d_to_2d_approximate(ee_pos, image_shape)
                    self.tracked_objects[yolo_id]['bbox'] = [bbox_2d[0], bbox_2d[1], 
                                                            self.tracked_objects[yolo_id]['bbox'][2],
                                                            self.tracked_objects[yolo_id]['bbox'][3]]
                else:
                    # Object was not detected but is grasped - add it back!
                    self.debug_message(f"  -> [GRASP ADD] Adding undetected grasped object {yolo_id}")
                    bbox_2d = self.project_3d_to_2d_approximate(ee_pos, image_shape)
                    
                    # Get last known bbox size or use default
                    if yolo_id in self.tracking_metadata:
                        last_bbox = self.tracking_metadata[yolo_id].get('last_bbox', [0, 0, 50, 50])
                        bbox_w, bbox_h = last_bbox[2], last_bbox[3]
                    else:
                        bbox_w, bbox_h = 50, 50  # Default size
                    
                    self.tracked_objects[yolo_id] = {
                        'bbox': [bbox_2d[0], bbox_2d[1], bbox_w, bbox_h],
                        'position': ee_pos,
                        'class': yolo_id.rsplit('_', 1)[0],  # Extract class from 'blue cube_0'
                        'conf': 1.0,
                        'grasped': True
                    }
                
                # Update tracking metadata
                if yolo_id in self.tracking_metadata:
                    self.tracking_metadata[yolo_id]['last_position'] = ee_pos
                    self.tracking_metadata[yolo_id]['grasped'] = True
                    self.tracking_metadata[yolo_id]['missing_frames'] = 0  # Reset missing frames
                    
                    # Save bbox size for future reference
                    if yolo_id in self.tracked_objects:
                        self.tracking_metadata[yolo_id]['last_bbox'] = self.tracked_objects[yolo_id]['bbox']
                else:
                    # Initialize metadata for this grasped object
                    self.tracking_metadata[yolo_id] = {
                        'missing_frames': 0,
                        'last_position': ee_pos,
                        'last_velocity': np.array([0.0, 0.0, 0.0]),
                        'bbox_velocity': np.array([0.0, 0.0]),
                        'grasped': True,
                        'position_history': [ee_pos],
                        'last_bbox': [bbox_2d[0], bbox_2d[1], 50, 50]
                    }
        
        return predicted_pos


    def execute(self, env, observations, n_act, symgoal, task_goals=None, render=False):
        self.warnings = {"obj_to_pick": True, "place_to_drop": True}
        self.image_buffer = []
        self.detected_positions = {}
        self.yolo_frequency = 15
        horizon = self.horizon if self.horizon is not None else 50
        self.debug_message("\tTask goal: ", symgoal)

        step_executor = 0
        done = False
        success = False 
        self.debug_message("\tStarting executor for step: ", self.id)

        while not done:
            anomaly_safe = False
            max_shift_threshold = 0.1#100.0  # in mm
            while not anomaly_safe:
                processed_obs = []
                
                for obs_num, observation in enumerate(observations):
                    if self.use_yolo or self.save_data:
                        objects_pos = observation["objects_pos"]
                        agentview_image = np.array(observation["agentview_image"].reshape((self.image_size, self.image_size, 3)), dtype=np.uint8)
                        wrist_image = np.array(observation["robot0_eye_in_hand_image"].reshape((self.image_size, self.image_size, 3)), dtype=np.uint8)
                        ee_pos = observation["robot0_eef_pos"]
                        
                        # STEP 1: Run YOLO detection (without grasp heuristics)
                        if (step_executor % self.yolo_frequency == 0 and obs_num == 0) or self.save_data:
                            predicted_cubes_xyz = self.yolo_estimate(
                                image1=agentview_image, 
                                image2=wrist_image, 
                                save_video=self.save_data, 
                                cubes_obs={},
                                ee_pos=ee_pos,
                                conf_threshold=0.8,
                                max_missing_frames=200,
                                render=render
                            )
                        else:
                            # Use tracked positions
                            tracked_positions = {}
                            for obj_id in self.tracked_objects.keys():
                                tracked_positions[obj_id] = self.tracking_metadata[obj_id]['last_position']
                            predicted_cubes_xyz = copy.deepcopy(tracked_positions)
                        
                        # STEP 2: Build/update relations mapping
                        if predicted_cubes_xyz:
                            self.update_relations_with_new_detections(predicted_cubes_xyz, objects_pos)
                        
                        # STEP 3: Correct positions of grasped objects (for both detected and undetected)
                        predicted_cubes_xyz = self.correct_grasped_object_positions(
                            predicted_cubes_xyz, 
                            ee_pos, 
                            image_shape=agentview_image.shape  # Pass image shape
                        )
                        
                        # STEP 4: Build observation for policy
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

                # Check for anomalies between consecutive observations
                anomaly_detected = False
                anomaly_indices = []
        
                for i in range(len(processed_obs) - 1):
                    obs_current = processed_obs[i]
                    obs_next = processed_obs[i + 1]
                    
                    # Compute absolute difference
                    diff = np.abs(obs_next - obs_current)
                    max_diff = np.max(diff)
                    
                    if max_diff > max_shift_threshold:
                        anomaly_detected = True
                        anomaly_indices.append(i)
                        #print(f"[ANOMALY] Large shift detected between observations {i} and {i+1}, reating as YOLO failure. Patching observations.")
                        self.debug_message(f"  Max shift: {max_diff:.2f} mm (threshold: {max_shift_threshold:.2f})")
                        self.debug_message(f"  Obs {i}: {obs_current}")
                        self.debug_message(f"  Obs {i+1}: {obs_next}")
                        self.debug_message(f"  Diff: {diff}")
                        # Patch the current observation by copying the next one
                        processed_obs[i] = obs_next.copy()
                
                # If anomaly detected, get fresh observations with forced YOLO detection
                if len(anomaly_indices) > 2:
                    print(f"[RECOVERY] Getting {len(observations)} fresh observations with YOLO detection")
                    for _ in range(len(observations)):
                        # Get current observation
                        obs = env._get_observations()
                        objects_pos = self.detector.get_all_objects_pos()
                        obs['objects_pos'] = objects_pos
                        observations.append(obs)
                        observations.pop(0)
                else:
                    anomaly_safe = True
            #print(processed_obs)
            processed_obs = np.array([processed_obs])
            np_obs_dict = {'obs': processed_obs.astype(np.float32)}
            obs_dict = dict_apply(np_obs_dict, 
                lambda x: torch.from_numpy(x).to(device=self.device))
            
            with torch.no_grad():
                action_dict = self.model.predict_action(obs_dict)
            
            np_action_dict = dict_apply(action_dict,
                lambda x: x.detach().to('cpu').numpy())
            actions = np_action_dict['action']#/1000.0
            #print("Actions: ", actions)
            
            if len(actions[0][0]) < 4:
                for index in self.nulified_action_indexes:
                    actions = np.insert(actions, index, 0, axis=2)
            
            #observations = []
            i_act = 0
            for action in actions[0]:
                action = self.map_gripper(action)
                _, _, done, info = env.step(action)
                if render:
                    env.render()
                obs = env._get_observations()
                objects_pos = self.detector.get_all_objects_pos()
                obs['objects_pos'] = objects_pos
                # Append last obs and pop first obs to keep the buffer size
                observations.append(obs)
                observations.pop(0)
                if i_act == n_act - 1:
                    break
                i_act += 1

            
            if done:
                self.debug_message("Environment terminated")
            
            step_executor += 1
            state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            success = self.Beta(state, symgoal)

            self.debug_message()
            self.debug_message("Checking goal predicates: ")
            self.debug_message(state)
            goal_reached = True
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
