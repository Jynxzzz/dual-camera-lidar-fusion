"""
Triple-View Late Fusion Pipeline
=================================
Fuses detections from:
- Drone Camera (80m top-down, 2D boxes)
- Ego Camera (front-view, 2D boxes)
- LiDAR (BEV, 3D boxes)

Author: Xingnan Zhou
Date: 2026-02-09
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple, Optional


# ============================================================================
# Coordinate Transformations
# ============================================================================

def get_carla_to_opencv_rotation():
    """
    Get rotation matrix from CARLA to OpenCV camera coordinates.

    CARLA: X forward, Y right, Z up (left-handed)
    OpenCV: X right, Y down, Z forward (right-handed)

    Returns:
        R: 3x3 rotation matrix
    """
    return np.array([
        [0,  1,  0],  # X_cv = Y_carla
        [0,  0, -1],  # Y_cv = -Z_carla
        [1,  0,  0]   # Z_cv = X_carla
    ], dtype=np.float32)


def get_3d_box_corners(location, dimensions, rotation_y):
    """
    Get 8 corners of a 3D bounding box.

    Args:
        location: [x, y, z] center in LiDAR frame
        dimensions: [l, w, h] length, width, height
        rotation_y: yaw angle in radians

    Returns:
        corners: 3x8 array of corner coordinates
    """
    x, y, z = location
    l, w, h = dimensions

    # 8 corners in object coordinate (center at origin)
    corners = np.array([
        [-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2],
        [ w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],
        [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]
    ])

    # Rotation around Z-axis (yaw)
    R_z = np.array([
        [np.cos(rotation_y), -np.sin(rotation_y), 0],
        [np.sin(rotation_y),  np.cos(rotation_y), 0],
        [0,                   0,                  1]
    ])

    # Rotate and translate
    corners = R_z @ corners + np.array([[x], [y], [z]])

    return corners


def project_3d_box_to_2d(box_3d, K, T_lidar_to_cam_cv, image_width=1920, image_height=1280):
    """
    Project a 3D bounding box to 2D image plane.

    Args:
        box_3d: dict with keys ['location', 'dimensions', 'rotation_y']
        K: 3x3 camera intrinsic matrix
        T_lidar_to_cam_cv: 4x4 transform from LiDAR to camera OpenCV frame
                           (Z=forward/depth, X=right, Y=down).
                           This must already include the camera orientation.
        image_width: image width in pixels
        image_height: image height in pixels

    Returns:
        box_2d: (x_min, y_min, x_max, y_max) or None if not visible
    """
    # Get 3D corners
    corners_3d = get_3d_box_corners(
        box_3d['location'],
        box_3d['dimensions'],
        box_3d['rotation_y']
    )

    # Convert to homogeneous coordinates
    corners_3d_homo = np.vstack([corners_3d, np.ones((1, 8))])  # 4x8

    # Transform to camera OpenCV frame (already includes rotation)
    corners_cam_cv = (T_lidar_to_cam_cv @ corners_3d_homo)[:3, :]  # 3x8

    # Filter points behind camera (z > 0 in OpenCV = in front)
    valid_mask = corners_cam_cv[2, :] > 0.1

    if not valid_mask.any():
        return None

    corners_valid = corners_cam_cv[:, valid_mask]

    # Project to image plane
    corners_2d = K @ corners_valid  # 3xN
    corners_2d = corners_2d[:2, :] / corners_2d[2, :]

    # Get 2D bounding box
    x_min = float(corners_2d[0, :].min())
    y_min = float(corners_2d[1, :].min())
    x_max = float(corners_2d[0, :].max())
    y_max = float(corners_2d[1, :].max())

    # Clamp to image bounds
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(image_width - 1, x_max)
    y_max = min(image_height - 1, y_max)

    # Check if box is valid (minimum size 5x5 pixels)
    if (x_max - x_min) < 5 or (y_max - y_min) < 5:
        return None

    return (x_min, y_min, x_max, y_max)


# ============================================================================
# Matching Algorithms
# ============================================================================

def compute_iou_2d(box1, box2):
    """
    Compute IoU between two 2D bounding boxes.

    Args:
        box1: (x_min, y_min, x_max, y_max)
        box2: (x_min, y_min, x_max, y_max)

    Returns:
        iou: intersection over union [0, 1]
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Intersection
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0, inter_xmax - inter_xmin)
    inter_h = max(0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    # Union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def match_boxes_iou(det_2d_list, proj_2d_list, iou_threshold=0.5):
    """
    Match detected 2D boxes with projected 3D boxes using IoU and Hungarian algorithm.

    Args:
        det_2d_list: List of tuples (x1, y1, x2, y2, conf, class_id)
        proj_2d_list: List of tuples (x1, y1, x2, y2, lidar_idx)
        iou_threshold: Minimum IoU for a valid match

    Returns:
        matches: List of tuples (det_idx, proj_idx, iou)
        unmatched_dets: List of unmatched detection indices
        unmatched_projs: List of unmatched projection indices
    """
    if len(det_2d_list) == 0 or len(proj_2d_list) == 0:
        return [], list(range(len(det_2d_list))), list(range(len(proj_2d_list)))

    # Compute IoU matrix
    iou_matrix = np.zeros((len(det_2d_list), len(proj_2d_list)))

    for i, det in enumerate(det_2d_list):
        det_box = det[:4]  # (x1, y1, x2, y2)
        for j, proj in enumerate(proj_2d_list):
            proj_box = proj[:4]
            iou_matrix[i, j] = compute_iou_2d(det_box, proj_box)

    # Hungarian matching (maximize IoU)
    row_indices, col_indices = linear_sum_assignment(-iou_matrix)

    # Filter matches by threshold
    matches = []
    for i, j in zip(row_indices, col_indices):
        if iou_matrix[i, j] >= iou_threshold:
            matches.append((i, j, iou_matrix[i, j]))

    # Get unmatched indices
    matched_det_idx = {m[0] for m in matches}
    matched_proj_idx = {m[1] for m in matches}

    unmatched_dets = [i for i in range(len(det_2d_list)) if i not in matched_det_idx]
    unmatched_projs = [j for j in range(len(proj_2d_list)) if j not in matched_proj_idx]

    return matches, unmatched_dets, unmatched_projs


# ============================================================================
# Fusion Pipeline
# ============================================================================

def triple_view_late_fusion(
    drone_dets_2d: List[Tuple],
    ego_dets_2d: List[Tuple],
    lidar_dets_3d: List[Dict],
    K_drone: np.ndarray,
    K_ego: np.ndarray,
    T_lidar_to_drone: np.ndarray,
    T_lidar_to_ego: np.ndarray,
    weights: Tuple[float, float, float] = (0.25, 0.25, 0.50),
    iou_threshold: float = 0.5,
    confidence_threshold: float = 0.3
) -> List[Dict]:
    """
    Triple-view late fusion of Drone + Ego + LiDAR detections.

    Args:
        drone_dets_2d: List of (x1, y1, x2, y2, conf, class_id) from Drone camera
        ego_dets_2d: List of (x1, y1, x2, y2, conf, class_id) from Ego camera
        lidar_dets_3d: List of dicts with keys:
            - location: [x, y, z]
            - dimensions: [l, w, h]
            - rotation_y: yaw angle
            - confidence: detection confidence
            - class_name: object class
        K_drone: 3x3 Drone camera intrinsic matrix
        K_ego: 3x3 Ego camera intrinsic matrix
        T_lidar_to_drone: 4x4 transform from LiDAR to Drone camera
        T_lidar_to_ego: 4x4 transform from LiDAR to Ego camera
        weights: (w_drone, w_ego, w_lidar) fusion weights
        iou_threshold: IoU threshold for matching
        confidence_threshold: Minimum fused confidence to keep detection

    Returns:
        fused_detections: List of dicts with fused 3D detections
    """
    w_drone, w_ego, w_lidar = weights

    # Step 1: Project LiDAR 3D boxes to both camera views
    drone_proj_2d = []
    ego_proj_2d = []

    for idx, box_3d in enumerate(lidar_dets_3d):
        # Project to Drone camera
        drone_box = project_3d_box_to_2d(box_3d, K_drone, T_lidar_to_drone)
        if drone_box is not None:
            drone_proj_2d.append((*drone_box, idx))  # (x1, y1, x2, y2, lidar_idx)

        # Project to Ego camera
        ego_box = project_3d_box_to_2d(box_3d, K_ego, T_lidar_to_ego)
        if ego_box is not None:
            ego_proj_2d.append((*ego_box, idx))

    # Step 2: Match Drone detections with projections
    matches_drone, unmatched_drone, _ = match_boxes_iou(
        drone_dets_2d, drone_proj_2d, iou_threshold=iou_threshold
    )

    # Step 3: Match Ego detections with projections
    matches_ego, unmatched_ego, _ = match_boxes_iou(
        ego_dets_2d, ego_proj_2d, iou_threshold=iou_threshold
    )

    # Step 4: Build fusion map (lidar_idx -> {drone_conf, ego_conf, lidar_conf})
    fusion_map = {}

    # Add Drone matches
    for det_idx, proj_idx, iou in matches_drone:
        lidar_idx = drone_proj_2d[proj_idx][-1]
        drone_conf = drone_dets_2d[det_idx][4]  # confidence

        if lidar_idx not in fusion_map:
            fusion_map[lidar_idx] = {'drone': 0.0, 'ego': 0.0, 'lidar': 0.0}
        fusion_map[lidar_idx]['drone'] = float(drone_conf)

    # Add Ego matches
    for det_idx, proj_idx, iou in matches_ego:
        lidar_idx = ego_proj_2d[proj_idx][-1]
        ego_conf = ego_dets_2d[det_idx][4]

        if lidar_idx not in fusion_map:
            fusion_map[lidar_idx] = {'drone': 0.0, 'ego': 0.0, 'lidar': 0.0}
        fusion_map[lidar_idx]['ego'] = float(ego_conf)

    # Add all LiDAR detections
    for idx, box_3d in enumerate(lidar_dets_3d):
        if idx not in fusion_map:
            fusion_map[idx] = {'drone': 0.0, 'ego': 0.0, 'lidar': 0.0}
        fusion_map[idx]['lidar'] = box_3d.get('confidence', 0.5)

    # Step 5: Compute fused confidence and create output
    fused_detections = []

    for lidar_idx, conf_dict in fusion_map.items():
        # Weighted average (normalize by sum of active weights)
        active_weights = 0.0
        fused_conf = 0.0

        if conf_dict['drone'] > 0:
            fused_conf += w_drone * conf_dict['drone']
            active_weights += w_drone

        if conf_dict['ego'] > 0:
            fused_conf += w_ego * conf_dict['ego']
            active_weights += w_ego

        if conf_dict['lidar'] > 0:
            fused_conf += w_lidar * conf_dict['lidar']
            active_weights += w_lidar

        # Normalize
        if active_weights > 0:
            fused_conf /= active_weights

        # Filter by confidence threshold
        if fused_conf < confidence_threshold:
            continue

        # Use LiDAR 3D box as reference (most accurate for 3D position)
        box_3d = lidar_dets_3d[lidar_idx]

        fused_detections.append({
            'location': box_3d['location'].copy() if isinstance(box_3d['location'], np.ndarray) else list(box_3d['location']),
            'dimensions': box_3d['dimensions'].copy() if isinstance(box_3d['dimensions'], np.ndarray) else list(box_3d['dimensions']),
            'rotation_y': float(box_3d['rotation_y']),
            'confidence': float(fused_conf),
            'class_name': box_3d['class_name'],
            'sources': {
                'drone': conf_dict['drone'] > 0,
                'ego': conf_dict['ego'] > 0,
                'lidar': conf_dict['lidar'] > 0
            },
            'source_confidences': {
                'drone': float(conf_dict['drone']),
                'ego': float(conf_dict['ego']),
                'lidar': float(conf_dict['lidar'])
            }
        })

    return fused_detections


# ============================================================================
# Adaptive Weighting
# ============================================================================

def get_adaptive_weights(box_3d, drone_matched, ego_matched):
    """
    Get adaptive fusion weights based on object distance and sensor visibility.

    Args:
        box_3d: 3D bounding box dict
        drone_matched: bool, whether matched with drone detection
        ego_matched: bool, whether matched with ego detection

    Returns:
        weights: (w_drone, w_ego, w_lidar) tuple
    """
    # Compute distance from ego vehicle
    x, y = box_3d['location'][:2]
    distance = np.sqrt(x**2 + y**2)

    # Close objects (<20m): ego camera more reliable
    if distance < 20:
        if ego_matched:
            return (0.15, 0.35, 0.50)  # boost ego

    # Far objects (>50m): drone camera more reliable
    elif distance > 50:
        if drone_matched:
            return (0.35, 0.15, 0.50)  # boost drone

    # Medium distance or no matches: balanced weights
    return (0.25, 0.25, 0.50)


# ============================================================================
# Visualization Helpers
# ============================================================================

def draw_3d_box_on_image(image, box_3d, K, T_lidar_to_cam, color=(0, 255, 0), thickness=2):
    """
    Draw a 3D bounding box on a 2D image.

    Args:
        image: numpy array (H, W, 3)
        box_3d: dict with 3D box parameters
        K: camera intrinsic matrix
        T_lidar_to_cam: transformation matrix
        color: RGB color tuple
        thickness: line thickness

    Returns:
        image: image with box drawn
    """
    import cv2

    # Get 3D corners
    corners_3d = get_3d_box_corners(
        box_3d['location'],
        box_3d['dimensions'],
        box_3d['rotation_y']
    )

    # Transform and project (T_lidar_to_cam already in OpenCV frame)
    corners_3d_homo = np.vstack([corners_3d, np.ones((1, 8))])
    corners_cam_cv = (T_lidar_to_cam @ corners_3d_homo)[:3, :]

    # Check if in front of camera
    if not (corners_cam_cv[2, :] > 0).any():
        return image

    # Project
    corners_2d = K @ corners_cam_cv
    corners_2d = corners_2d[:2, :] / corners_2d[2, :]
    corners_2d = corners_2d.astype(np.int32)

    # Draw 12 edges of the 3D box
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # vertical edges
    ]

    for i, j in edges:
        if i < corners_2d.shape[1] and j < corners_2d.shape[1]:
            pt1 = tuple(corners_2d[:, i])
            pt2 = tuple(corners_2d[:, j])
            cv2.line(image, pt1, pt2, color, thickness)

    return image


# ============================================================================
# Main Interface
# ============================================================================

class TripleViewFusion:
    """
    Triple-view sensor fusion system.
    """

    def __init__(
        self,
        K_drone: np.ndarray,
        K_ego: np.ndarray,
        T_lidar_to_drone: np.ndarray,
        T_lidar_to_ego: np.ndarray,
        weights: Tuple[float, float, float] = (0.25, 0.25, 0.50),
        iou_threshold: float = 0.5,
        confidence_threshold: float = 0.3
    ):
        """
        Initialize fusion system.

        Args:
            K_drone: 3x3 Drone camera intrinsic matrix
            K_ego: 3x3 Ego/SDC camera intrinsic matrix
            T_lidar_to_drone: 4x4 transform from LiDAR to Drone camera
            T_lidar_to_ego: 4x4 transform from LiDAR to Ego/SDC camera
            weights: Fusion weights (drone, ego, lidar)
            iou_threshold: IoU threshold for matching
            confidence_threshold: Minimum fused confidence
        """
        self.K_drone = K_drone
        self.K_ego = K_ego
        self.T_lidar_to_drone = T_lidar_to_drone
        self.T_lidar_to_ego = T_lidar_to_ego

        self.weights = weights
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold

    def fuse(
        self,
        drone_dets_2d: List[Tuple],
        ego_dets_2d: List[Tuple],
        lidar_dets_3d: List[Dict]
    ) -> List[Dict]:
        """
        Perform fusion on a single frame.

        Returns:
            fused_detections: List of fused 3D detections
        """
        return triple_view_late_fusion(
            drone_dets_2d,
            ego_dets_2d,
            lidar_dets_3d,
            self.K_drone,
            self.K_ego,
            self.T_lidar_to_drone,
            self.T_lidar_to_ego,
            weights=self.weights,
            iou_threshold=self.iou_threshold,
            confidence_threshold=self.confidence_threshold
        )

    @classmethod
    def from_dataset_dir(cls, dataset_dir, frame_id="000000", **kwargs):
        """
        Load calibration from per-frame calib text files and known sensor poses.

        The calib files store translation-only transforms (rotation was omitted
        during collection). We reconstruct the full transforms using known
        CARLA sensor orientations.

        Sensor poses (attached to vehicle, CARLA coords X=fwd, Y=right, Z=up):
            SDC:   (1.5, 0, 2.4),  pitch=0,   yaw=0  (forward-facing)
            Drone: (0, 0, 40),     pitch=-90, yaw=0  (looking down)
            LiDAR: (0, 0, 2.5),    pitch=0,   yaw=0  (forward)
        """
        import os

        calib_path = os.path.join(dataset_dir, "calib", f"{frame_id}.txt")
        calib = {}
        with open(calib_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                key, values = line.split(':', 1)
                calib[key.strip()] = np.array([float(v) for v in values.strip().split()])

        # Parse intrinsics (9 values -> 3x3)
        K_sdc = calib['K_sdc'].reshape(3, 3)
        K_drone = calib['K_drone'].reshape(3, 3)

        # Build LiDAR → Camera OpenCV transforms directly.
        #
        # CARLA: X=forward, Y=right, Z=up (vehicle frame)
        # OpenCV camera: X=right, Y=down, Z=depth (forward for camera)
        #
        # Sensor positions in vehicle frame:
        #   SDC:   (1.5, 0, 2.4), pitch=0  (forward-facing)
        #   Drone: (0, 0, 40),    pitch=-90 (looking down)
        #   LiDAR: (0, 0, 2.5),   pitch=0  (forward)
        #
        # Since labels are in LiDAR frame ≈ vehicle frame (just z offset),
        # we compute T_lidar_to_cam_opencv directly.

        # --- SDC Camera (forward-facing) ---
        # OpenCV axes in world: X_cv=Y_world, Y_cv=-Z_world, Z_cv=X_world
        R_sdc = np.array([
            [0,  1,  0],   # X_cv = Y (right)
            [0,  0, -1],   # Y_cv = -Z (down)
            [1,  0,  0]    # Z_cv = X (forward = depth)
        ], dtype=np.float64)
        # Translation: LiDAR pos relative to SDC, rotated to OpenCV
        t_lidar_in_sdc = np.array([0.0 - 0.0, 0.0 - 0.0, 2.5 - 2.4])  # lidar - sdc
        t_sdc_cv = R_sdc @ t_lidar_in_sdc  # = [0, -0.1, -1.5]

        T_lidar_to_sdc_cv = np.eye(4, dtype=np.float64)
        T_lidar_to_sdc_cv[:3, :3] = R_sdc
        T_lidar_to_sdc_cv[:3, 3] = t_sdc_cv

        # --- Drone Camera (looking straight down, pitch=-90) ---
        # Drone looks down (-Z_world). With yaw=0, roll=0:
        #   X_cv = Y_world (right)
        #   Y_cv = -X_world (backward = down in image for downward cam)
        #   Z_cv = -Z_world (depth = distance below drone)
        R_drone = np.array([
            [ 0,  1,  0],   # X_cv = Y (right)
            [-1,  0,  0],   # Y_cv = -X (objects in front appear at top)
            [ 0,  0, -1]    # Z_cv = -Z (depth = below drone)
        ], dtype=np.float64)
        # Translation: LiDAR pos relative to Drone, rotated to OpenCV
        t_lidar_in_drone = np.array([0.0 - 0.0, 0.0 - 0.0, 2.5 - 40.0])  # lidar - drone
        t_drone_cv = R_drone @ t_lidar_in_drone  # = [0, 0, 37.5]

        T_lidar_to_drone_cv = np.eye(4, dtype=np.float64)
        T_lidar_to_drone_cv[:3, :3] = R_drone
        T_lidar_to_drone_cv[:3, 3] = t_drone_cv

        return cls(
            K_drone, K_sdc,
            T_lidar_to_drone_cv, T_lidar_to_sdc_cv,
            **kwargs
        )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the fusion pipeline.
    """

    # Example: Load calibration
    dataset_dir = "/mnt/hdd12t/outputs/carla_out/triple_view_dataset"

    # Simulated detections for testing
    drone_dets_2d = [
        (100, 100, 200, 200, 0.9, 0),  # (x1, y1, x2, y2, conf, class)
        (300, 150, 400, 250, 0.85, 0),
    ]

    ego_dets_2d = [
        (500, 400, 600, 550, 0.88, 0),
        (700, 300, 800, 450, 0.92, 0),
    ]

    lidar_dets_3d = [
        {
            'location': [10.0, 2.0, 0.5],
            'dimensions': [4.5, 2.0, 1.8],
            'rotation_y': 0.1,
            'confidence': 0.8,
            'class_name': 'Car'
        },
        {
            'location': [15.0, -3.0, 0.5],
            'dimensions': [4.2, 1.9, 1.7],
            'rotation_y': -0.2,
            'confidence': 0.75,
            'class_name': 'Car'
        }
    ]

    # Create dummy calibration for testing
    K_drone = np.array([
        [960, 0, 960],
        [0, 960, 540],
        [0, 0, 1]
    ], dtype=np.float32)

    K_ego = np.array([
        [960, 0, 960],
        [0, 960, 540],
        [0, 0, 1]
    ], dtype=np.float32)

    T_lidar = np.eye(4)
    T_lidar[2, 3] = 2.5  # z = 2.5m

    T_drone = np.eye(4)
    T_drone[2, 3] = 80.0  # z = 80m

    T_ego_cam = np.eye(4)
    T_ego_cam[0, 3] = 0.5
    T_ego_cam[2, 3] = 2.4

    # Compute LiDAR-to-camera transforms
    T_lidar_to_drone = np.linalg.inv(T_drone) @ T_lidar
    T_lidar_to_ego = np.linalg.inv(T_ego_cam) @ T_lidar

    # Initialize fusion system
    fusion = TripleViewFusion(
        K_drone, K_ego,
        T_lidar_to_drone, T_lidar_to_ego,
        weights=(0.25, 0.25, 0.50)
    )

    # Perform fusion
    fused = fusion.fuse(drone_dets_2d, ego_dets_2d, lidar_dets_3d)

    print(f"Fused {len(fused)} detections:")
    for det in fused:
        print(f"  {det['class_name']} at {det['location']} - conf={det['confidence']:.3f}")
        print(f"    Sources: {det['sources']}")
