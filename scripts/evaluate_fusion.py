"""
Evaluate fusion configurations for ablation study.

Computes 3D detection metrics (AP, Recall, Precision) at various IoU thresholds
for different sensor configurations:
  1. LiDAR only
  2. LiDAR + SDC (ego camera, boost-only)
  3. LiDAR + Drone (boost + suppress)
  4. LiDAR + SDC + Drone (asymmetric full fusion)
  5. Symmetric fusion (both cameras boost + suppress) — ablation baseline
  6. Naive average fusion (simple confidence averaging) — ablation baseline

Usage:
    python evaluate_fusion.py \
        --dataset /mnt/hdd12t/outputs/carla_out/dual_camera_lidar_town10hd_v2 \
        --lidar-results /path/to/result.pkl \
        --drone-model /home/xingnan/DRONE_DETECTION/model/traffic_analysis.pt \
        --ego-model yolov8n.pt \
        --output /mnt/hdd12t/outputs/carla_out/fusion_eval_results \
        --split val

Author: Xingnan Zhou
Date: 2026-02-10
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from triple_view_fusion import (
    TripleViewFusion,
    project_3d_box_to_2d,
    compute_iou_2d,
    triple_view_late_fusion,
)


# ============================================================================
# 3D IoU Computation
# ============================================================================

def _box_to_corners(cx, cy, dx, dy, ry):
    """Convert a rotated BEV box (center, size, yaw) to four corner points."""
    cos_r = np.cos(ry)
    sin_r = np.sin(ry)
    # Half-extents along each local axis
    hdx, hdy = dx / 2.0, dy / 2.0
    # Four corners in local frame, then rotate
    corners = np.array([
        [-hdx, -hdy],
        [ hdx, -hdy],
        [ hdx,  hdy],
        [-hdx,  hdy],
    ])
    rot = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
    corners = corners @ rot.T  # (4, 2)
    corners[:, 0] += cx
    corners[:, 1] += cy
    return corners


def compute_iou_3d_bev(box1, box2):
    """
    Compute oriented BEV IoU between two 3D boxes using Shapely.

    Respects the heading angle (rotation_y) for accurate overlap computation,
    matching KITTI-style evaluation conventions.

    Args:
        box1: dict with 'location' [x,y,z], 'dimensions' [dx,dy,dz], 'rotation_y' float
        box2: same format

    Returns:
        iou: Oriented BEV IoU [0, 1]
    """
    from shapely.geometry import Polygon

    x1, y1 = box1['location'][0], box1['location'][1]
    dx1, dy1 = box1['dimensions'][0], box1['dimensions'][1]
    ry1 = box1.get('rotation_y', 0.0)

    x2, y2 = box2['location'][0], box2['location'][1]
    dx2, dy2 = box2['dimensions'][0], box2['dimensions'][1]
    ry2 = box2.get('rotation_y', 0.0)

    corners1 = _box_to_corners(x1, y1, dx1, dy1, ry1)
    corners2 = _box_to_corners(x2, y2, dx2, dy2, ry2)

    poly1 = Polygon(corners1)
    poly2 = Polygon(corners2)

    if not poly1.is_valid or not poly2.is_valid:
        return 0.0

    inter_area = poly1.intersection(poly2).area
    union_area = poly1.area + poly2.area - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def compute_center_distance(box1, box2):
    """Euclidean distance between box centers in BEV."""
    loc1 = np.array(box1['location'][:2])
    loc2 = np.array(box2['location'][:2])
    return np.linalg.norm(loc1 - loc2)


# ============================================================================
# Matching GT to Predictions
# ============================================================================

def match_detections_to_gt(predictions, ground_truth, iou_threshold=0.5, class_name=None):
    """
    Match predictions to ground truth using BEV IoU.

    Args:
        predictions: list of detection dicts (sorted by confidence, descending)
        ground_truth: list of GT dicts
        iou_threshold: minimum IoU for a true positive
        class_name: if specified, only match this class

    Returns:
        tp: array of 0/1 for each prediction (true positive or not)
        fp: array of 0/1 for each prediction (false positive or not)
        num_gt: number of ground truth objects
    """
    if class_name:
        predictions = [p for p in predictions if p['class_name'] == class_name]
        ground_truth = [g for g in ground_truth if g['class_name'] == class_name]

    num_gt = len(ground_truth)
    if num_gt == 0:
        return np.ones(len(predictions)), np.ones(len(predictions)), 0

    # Sort predictions by confidence (descending)
    predictions = sorted(predictions, key=lambda x: -x.get('confidence', 0.5))

    tp = np.zeros(len(predictions))
    fp = np.zeros(len(predictions))
    gt_matched = np.zeros(num_gt, dtype=bool)

    for pred_idx, pred in enumerate(predictions):
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(ground_truth):
            if gt_matched[gt_idx]:
                continue
            iou = compute_iou_3d_bev(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp[pred_idx] = 1
            gt_matched[best_gt_idx] = True
        else:
            fp[pred_idx] = 1

    return tp, fp, num_gt


def compute_ap(tp_all, fp_all, num_gt_total):
    """
    Compute Average Precision from accumulated TP/FP across frames.

    Args:
        tp_all: concatenated TP flags
        fp_all: concatenated FP flags
        num_gt_total: total number of GT objects

    Returns:
        ap: Average Precision
        precision: precision array
        recall: recall array
    """
    if num_gt_total == 0:
        return 0.0, np.array([]), np.array([])

    # Cumulative sums
    tp_cum = np.cumsum(tp_all)
    fp_cum = np.cumsum(fp_all)

    recall = tp_cum / num_gt_total
    precision = tp_cum / (tp_cum + fp_cum)

    # 11-point interpolation (PASCAL VOC style)
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        mask = recall >= t
        if mask.any():
            ap += precision[mask].max() / 11.0

    return ap, precision, recall


# ============================================================================
# Data Loading
# ============================================================================

def load_gt_labels(label_path):
    """Load GT labels from text file."""
    if not os.path.exists(label_path):
        return []

    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            dx, dy, dz = float(parts[3]), float(parts[4]), float(parts[5])
            ry = float(parts[6])
            cls = parts[7]
            labels.append({
                'location': [x, y, z],
                'dimensions': [dx, dy, dz],
                'rotation_y': ry,
                'class_name': cls
            })
    return labels


def load_all_lidar_detections(result_pkl_path):
    """Load all detections from OpenPCDet result.pkl."""
    with open(result_pkl_path, 'rb') as f:
        results_list = pickle.load(f)

    results_by_frame = {}
    for item in results_list:
        fid = str(item['frame_id'])
        dets = []
        num_dets = len(item['name'])
        for i in range(num_dets):
            box = item['boxes_lidar'][i]
            dets.append({
                'location': [float(box[0]), float(box[1]), float(box[2])],
                'dimensions': [float(box[3]), float(box[4]), float(box[5])],
                'rotation_y': float(box[6]),
                'confidence': float(item['score'][i]),
                'class_name': str(item['name'][i])
            })
        results_by_frame[fid] = dets
    return results_by_frame


def load_gt_labels_from_label(label_path):
    """Load ground truth labels from text file (same as load_gt_labels)."""
    return load_gt_labels(label_path)


def run_yolo_detection(model, image_path, conf_threshold=0.25):
    """Run YOLO on an image and return detections."""
    results = model(image_path, conf=conf_threshold, verbose=False, device=0)
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                detections.append((x1, y1, x2, y2, conf, cls))
    return detections


# ============================================================================
# Fusion Configurations
# ============================================================================

def run_lidar_only(lidar_dets, conf_threshold=0.3):
    """Config 1: LiDAR only — just filter by confidence."""
    return [d for d in lidar_dets if d.get('confidence', 0.5) >= conf_threshold]


def run_lidar_plus_camera(
    lidar_dets, camera_dets_2d, K_cam, T_lidar_to_cam,
    conf_threshold=0.3, image_width=1920, image_height=1280,
    boost_factor=1.15, match_iou_thresh=0.3,
    allow_suppress=False, suppress_factor=0.75,
    suppress_conf_gate=0.45, suppress_classes=('Car',),
):
    """
    Config 2/3: LiDAR + single camera fusion.

    - Boost camera-confirmed LiDAR detections.
    - Optionally suppress in-FOV unconfirmed low-confidence detections
      (only for specified classes, only below conf gate).
      SDC uses boost-only; Drone uses boost+suppress.
    """
    from triple_view_fusion import match_boxes_iou

    # Project all LiDAR boxes to camera
    proj_2d = []
    for idx, box_3d in enumerate(lidar_dets):
        box_2d = project_3d_box_to_2d(
            box_3d, K_cam, T_lidar_to_cam,
            image_width=image_width, image_height=image_height
        )
        if box_2d is not None:
            proj_2d.append((*box_2d, idx))

    # Match camera detections with projected LiDAR boxes
    matches, _, _ = match_boxes_iou(camera_dets_2d, proj_2d, iou_threshold=match_iou_thresh)

    # Build matched set and in-FOV set
    matched_lidar_idx = set()
    for det_idx, proj_idx, iou in matches:
        lidar_idx = proj_2d[proj_idx][-1]
        matched_lidar_idx.add(lidar_idx)

    in_fov_set = {p[-1] for p in proj_2d}

    # Apply boost + optional suppress
    fused = []
    for idx, det in enumerate(lidar_dets):
        conf = det.get('confidence', 0.5)
        cls = det.get('class_name', '')

        if idx in matched_lidar_idx:
            # Camera confirmed → boost
            conf = min(conf * boost_factor, 1.0)
        elif allow_suppress and idx in in_fov_set:
            # In FOV but not confirmed → suppress (class-aware, confidence-gated)
            if cls in suppress_classes and conf < suppress_conf_gate:
                conf = conf * suppress_factor

        if conf >= conf_threshold:
            d = det.copy()
            d['confidence'] = conf
            fused.append(d)

    return fused


def drone_2d_to_3d(det_2d, K_drone, T_lidar_to_drone_cv, default_dims=None):
    """
    Lift a drone 2D detection to approximate 3D box in LiDAR frame.

    The drone looks straight down from 40m, so we can estimate (x, y)
    by inverse-projecting the 2D box center through the ground plane.

    Args:
        det_2d: (x1, y1, x2, y2, conf, cls_id)
        K_drone: 3x3 intrinsics
        T_lidar_to_drone_cv: 4x4 LiDAR-to-drone-OpenCV transform
        default_dims: dict mapping YOLO cls_id -> (dx, dy, dz) default sizes

    Returns:
        det_3d: dict with location, dimensions, etc. or None
    """
    if default_dims is None:
        default_dims = {
            0: [3.5, 1.5, 1.4],  # bus -> treat as large car
            1: [4.0, 1.8, 1.6],  # car
            2: [6.0, 2.5, 2.5],  # truck
            3: [4.5, 2.0, 1.8],  # van
        }

    x1, y1, x2, y2, conf, cls_id = det_2d
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    # Inverse projection: pixel -> drone camera ray
    # In OpenCV camera: [cx, cy, 1] = K @ [X/Z, Y/Z, 1]
    fx, fy = K_drone[0, 0], K_drone[1, 1]
    cx_k, cy_k = K_drone[0, 2], K_drone[1, 2]

    # Ray direction in drone OpenCV frame
    ray_cv = np.array([(cx - cx_k) / fx, (cy - cy_k) / fy, 1.0])

    # Assume ground plane at z_lidar ≈ -1.5 (typical ground level relative to LiDAR)
    # We need to find where this ray hits z_lidar = z_ground
    # Transform ray from drone OpenCV back to LiDAR frame
    R_inv = T_lidar_to_drone_cv[:3, :3].T  # inv rotation
    t_inv = -R_inv @ T_lidar_to_drone_cv[:3, 3]

    # Ray origin in LiDAR frame = camera position in LiDAR frame
    cam_pos_lidar = t_inv
    ray_dir_lidar = R_inv @ ray_cv

    # Ground plane: z_lidar = z_ground
    z_ground = -1.5  # approximate ground level relative to LiDAR
    if abs(ray_dir_lidar[2]) < 1e-6:
        return None  # ray parallel to ground

    t_hit = (z_ground - cam_pos_lidar[2]) / ray_dir_lidar[2]
    if t_hit <= 0:
        return None  # behind camera

    hit_point = cam_pos_lidar + t_hit * ray_dir_lidar
    x_lidar, y_lidar = float(hit_point[0]), float(hit_point[1])

    # Distance check
    dist = np.sqrt(x_lidar**2 + y_lidar**2)
    if dist > 70:
        return None

    # Get dimensions based on YOLO class
    dims = default_dims.get(int(cls_id), [4.0, 1.8, 1.6])

    # Map YOLO class to our class names
    # traffic_analysis.pt: 0=bus, 1=car, 2=truck, 3=van -> all "Car"
    class_name = 'Car'

    return {
        'location': [x_lidar, y_lidar, z_ground],
        'dimensions': dims,
        'rotation_y': 0.0,  # unknown from BEV
        'confidence': float(conf) * 0.7,  # discount camera-only detections
        'class_name': class_name,
    }


def run_enhanced_fusion(
    lidar_dets, drone_dets_2d, ego_dets_2d,
    K_drone, K_ego, T_lidar_to_drone, T_lidar_to_ego,
    conf_threshold=0.3,
    boost_single=1.15, boost_dual=1.30,
    match_iou_thresh=0.3,
    suppress_factor=0.75, suppress_conf_gate=0.45,
    suppress_classes=('Car',),
    add_camera_proposals=False,
):
    """
    Asymmetric dual-camera fusion.

    The drone (top-down, wide coverage) provides both boost and suppress signals.
    The SDC (forward-only, limited coverage) provides boost-only.
    Suppress is class-aware (Car only) and confidence-gated.
    """
    from triple_view_fusion import match_boxes_iou

    # Step 1: Project LiDAR to both cameras
    drone_proj = []
    ego_proj = []
    for idx, box in enumerate(lidar_dets):
        d_box = project_3d_box_to_2d(box, K_drone, T_lidar_to_drone, 1920, 1280)
        if d_box is not None:
            drone_proj.append((*d_box, idx))
        e_box = project_3d_box_to_2d(box, K_ego, T_lidar_to_ego, 1920, 1280)
        if e_box is not None:
            ego_proj.append((*e_box, idx))

    # Step 2: Match camera detections
    matches_d, unmatched_drone, _ = match_boxes_iou(drone_dets_2d, drone_proj, match_iou_thresh)
    matches_e, _, _ = match_boxes_iou(ego_dets_2d, ego_proj, match_iou_thresh)

    matched_by_drone = {drone_proj[proj_idx][-1] for _, proj_idx, _ in matches_d}
    matched_by_ego = {ego_proj[proj_idx][-1] for _, proj_idx, _ in matches_e}
    in_drone_fov = {p[-1] for p in drone_proj}

    # Step 3: Asymmetric fusion — boost from both, suppress from drone only
    fused = []
    for idx, det in enumerate(lidar_dets):
        conf = det.get('confidence', 0.5)
        cls = det.get('class_name', '')
        n_confirmed = int(idx in matched_by_drone) + int(idx in matched_by_ego)

        if n_confirmed >= 2:
            conf = min(conf * boost_dual, 1.0)
        elif n_confirmed == 1:
            conf = min(conf * boost_single, 1.0)
        elif idx in in_drone_fov:
            # In drone FOV but not confirmed by either camera → suppress
            # Only for specified classes and below confidence gate
            if cls in suppress_classes and conf < suppress_conf_gate:
                conf = conf * suppress_factor

        if conf >= conf_threshold:
            d = det.copy()
            d['confidence'] = conf
            fused.append(d)

    # Step 4: Camera-only proposals from drone (high confidence only)
    if add_camera_proposals:
        existing_locs = np.array([d['location'][:2] for d in fused]) if fused else np.empty((0, 2))
        for det_idx in unmatched_drone:
            det_2d = drone_dets_2d[det_idx]
            if det_2d[4] < 0.7:
                continue
            proposal = drone_2d_to_3d(det_2d, K_drone, T_lidar_to_drone)
            if proposal is None:
                continue
            if len(existing_locs) > 0:
                dists = np.linalg.norm(existing_locs - np.array(proposal['location'][:2]), axis=1)
                if dists.min() < 5.0:
                    continue
            if proposal['confidence'] >= conf_threshold:
                fused.append(proposal)
                existing_locs = np.vstack([existing_locs, proposal['location'][:2]]) if len(existing_locs) > 0 else np.array([proposal['location'][:2]])

    return fused


def run_symmetric_fusion(
    lidar_dets, drone_dets_2d, ego_dets_2d,
    K_drone, K_ego, T_lidar_to_drone, T_lidar_to_ego,
    conf_threshold=0.3,
    boost_single=1.15, boost_dual=1.30,
    match_iou_thresh=0.3,
    suppress_factor=0.75, suppress_conf_gate=0.45,
    suppress_classes=('Car',),
):
    """
    Symmetric dual-camera fusion baseline (ablation).

    Unlike the asymmetric version (run_enhanced_fusion), BOTH cameras
    apply boost AND suppress with identical parameters. This serves as
    a comparison to demonstrate the value of the asymmetric design where
    SDC is boost-only.

    Symmetric rule:
      - Confirmed by 2 cameras → boost ×boost_dual
      - Confirmed by 1 camera  → boost ×boost_single
      - In FOV of EITHER camera but not confirmed → suppress
        (class-aware, confidence-gated, same as drone in asymmetric)
    """
    from triple_view_fusion import match_boxes_iou

    # Step 1: Project LiDAR to both cameras
    drone_proj = []
    ego_proj = []
    for idx, box in enumerate(lidar_dets):
        d_box = project_3d_box_to_2d(box, K_drone, T_lidar_to_drone, 1920, 1280)
        if d_box is not None:
            drone_proj.append((*d_box, idx))
        e_box = project_3d_box_to_2d(box, K_ego, T_lidar_to_ego, 1920, 1280)
        if e_box is not None:
            ego_proj.append((*e_box, idx))

    # Step 2: Match camera detections
    matches_d, _, _ = match_boxes_iou(drone_dets_2d, drone_proj, match_iou_thresh)
    matches_e, _, _ = match_boxes_iou(ego_dets_2d, ego_proj, match_iou_thresh)

    matched_by_drone = {drone_proj[proj_idx][-1] for _, proj_idx, _ in matches_d}
    matched_by_ego = {ego_proj[proj_idx][-1] for _, proj_idx, _ in matches_e}
    in_drone_fov = {p[-1] for p in drone_proj}
    in_ego_fov = {p[-1] for p in ego_proj}

    # Step 3: Symmetric fusion — boost from both, suppress from BOTH
    fused = []
    for idx, det in enumerate(lidar_dets):
        conf = det.get('confidence', 0.5)
        cls = det.get('class_name', '')
        n_confirmed = int(idx in matched_by_drone) + int(idx in matched_by_ego)

        if n_confirmed >= 2:
            conf = min(conf * boost_dual, 1.0)
        elif n_confirmed == 1:
            conf = min(conf * boost_single, 1.0)
        elif idx in in_drone_fov or idx in in_ego_fov:
            # In FOV of EITHER camera but not confirmed → suppress
            # (This is the key difference: asymmetric only suppresses from drone FOV)
            if cls in suppress_classes and conf < suppress_conf_gate:
                conf = conf * suppress_factor

        if conf >= conf_threshold:
            d = det.copy()
            d['confidence'] = conf
            fused.append(d)

    return fused


def run_naive_average_fusion(
    lidar_dets, drone_dets_2d, ego_dets_2d,
    K_drone, K_ego, T_lidar_to_drone, T_lidar_to_ego,
    conf_threshold=0.3,
    match_iou_thresh=0.3,
):
    """
    Naive average fusion baseline (ablation).

    The simplest possible fusion: for LiDAR detections matched by a camera,
    replace confidence with the average of LiDAR confidence and camera
    confidence. No suppress, no FOV-aware logic, no boost factors.

    For detections matched by multiple cameras, average all available
    confidences (LiDAR + camera1 + camera2) / N.

    Unmatched detections pass through unchanged.
    """
    from triple_view_fusion import match_boxes_iou

    # Step 1: Project LiDAR to both cameras
    drone_proj = []
    ego_proj = []
    for idx, box in enumerate(lidar_dets):
        d_box = project_3d_box_to_2d(box, K_drone, T_lidar_to_drone, 1920, 1280)
        if d_box is not None:
            drone_proj.append((*d_box, idx))
        e_box = project_3d_box_to_2d(box, K_ego, T_lidar_to_ego, 1920, 1280)
        if e_box is not None:
            ego_proj.append((*e_box, idx))

    # Step 2: Match camera detections
    matches_d, _, _ = match_boxes_iou(drone_dets_2d, drone_proj, match_iou_thresh)
    matches_e, _, _ = match_boxes_iou(ego_dets_2d, ego_proj, match_iou_thresh)

    # Build maps: lidar_idx -> camera confidence
    drone_conf_map = {}  # lidar_idx -> camera confidence
    for det_idx, proj_idx, iou in matches_d:
        lidar_idx = drone_proj[proj_idx][-1]
        cam_conf = drone_dets_2d[det_idx][4]  # (x1,y1,x2,y2,conf,cls)
        drone_conf_map[lidar_idx] = cam_conf

    ego_conf_map = {}
    for det_idx, proj_idx, iou in matches_e:
        lidar_idx = ego_proj[proj_idx][-1]
        cam_conf = ego_dets_2d[det_idx][4]
        ego_conf_map[lidar_idx] = cam_conf

    # Step 3: Naive average — no suppress, no boost, just average matched confs
    fused = []
    for idx, det in enumerate(lidar_dets):
        lidar_conf = det.get('confidence', 0.5)

        # Collect all available confidences
        confs = [lidar_conf]
        if idx in drone_conf_map:
            confs.append(drone_conf_map[idx])
        if idx in ego_conf_map:
            confs.append(ego_conf_map[idx])

        # Simple average
        avg_conf = np.mean(confs)

        if avg_conf >= conf_threshold:
            d = det.copy()
            d['confidence'] = float(avg_conf)
            fused.append(d)

    return fused


# ============================================================================
# Main Evaluation
# ============================================================================

def evaluate_config(predictions_by_frame, gt_by_frame, class_names, iou_thresholds):
    """
    Evaluate a configuration across all frames.

    Returns:
        results: dict with AP, recall, precision per class per IoU threshold
    """
    results = {}

    for cls in class_names:
        results[cls] = {}
        for iou_thresh in iou_thresholds:
            all_tp = []
            all_fp = []
            total_gt = 0

            for frame_id in gt_by_frame:
                gt = gt_by_frame[frame_id]
                preds = predictions_by_frame.get(frame_id, [])

                tp, fp, num_gt = match_detections_to_gt(preds, gt, iou_thresh, class_name=cls)
                all_tp.append(tp)
                all_fp.append(fp)
                total_gt += num_gt

            if total_gt == 0:
                results[cls][f'AP@{iou_thresh}'] = 0.0
                results[cls][f'Recall@{iou_thresh}'] = 0.0
                results[cls][f'Precision@{iou_thresh}'] = 0.0
                continue

            all_tp = np.concatenate(all_tp) if all_tp else np.array([])
            all_fp = np.concatenate(all_fp) if all_fp else np.array([])

            ap, precision, recall = compute_ap(all_tp, all_fp, total_gt)

            final_recall = recall[-1] if len(recall) > 0 else 0.0
            final_precision = precision[-1] if len(precision) > 0 else 0.0

            results[cls][f'AP@{iou_thresh}'] = float(ap)
            results[cls][f'Recall@{iou_thresh}'] = float(final_recall)
            results[cls][f'Precision@{iou_thresh}'] = float(final_precision)
            results[cls][f'num_gt'] = total_gt

    # Compute mean AP (mAP) across classes
    for iou_thresh in iou_thresholds:
        aps = [results[cls][f'AP@{iou_thresh}'] for cls in class_names]
        results[f'mAP@{iou_thresh}'] = float(np.mean(aps))

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate fusion configurations")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--lidar-results", type=str, required=True,
                        help="Path to PointPillar result.pkl")
    parser.add_argument("--drone-model", type=str, default=None,
                        help="Path to Drone YOLO model (.pt)")
    parser.add_argument("--ego-model", type=str, default=None,
                        help="Path to Ego YOLO model (.pt)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for evaluation results")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--seed-tag", type=str, default="",
                        help="Tag for this seed run (e.g., seed0)")
    parser.add_argument("--max-range", type=float, default=None,
                        help="Max BEV range from ego (meters). Only evaluate objects within this range.")
    parser.add_argument("--yolo-cache", type=str, default=None,
                        help="Path to YOLO cache .pkl file. If exists, load cached detections; if not, run YOLO and save.")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load frame IDs
    split_file = dataset_dir / "ImageSets" / f"{args.split}.txt"
    with open(split_file, 'r') as f:
        frame_ids = [line.strip() for line in f]
    print(f"Evaluating on {len(frame_ids)} {args.split} frames")

    # Load ground truth
    print("Loading ground truth...")
    gt_by_frame = {}
    for fid in frame_ids:
        gt_by_frame[fid] = load_gt_labels(str(dataset_dir / "labels" / f"{fid}.txt"))
    total_gt = sum(len(v) for v in gt_by_frame.values())
    print(f"  Total GT objects: {total_gt}")

    # Load LiDAR detections
    print("Loading LiDAR detections...")
    lidar_by_frame = load_all_lidar_detections(args.lidar_results)
    total_lidar = sum(len(lidar_by_frame.get(fid, [])) for fid in frame_ids)
    print(f"  Total LiDAR detections: {total_lidar}")

    # Apply range filter if specified
    if args.max_range is not None:
        max_r = args.max_range
        print(f"  Applying range filter: 0-{max_r}m")
        for fid in frame_ids:
            gt_by_frame[fid] = [g for g in gt_by_frame[fid]
                                if np.sqrt(g['location'][0]**2 + g['location'][1]**2) <= max_r]
            if fid in lidar_by_frame:
                lidar_by_frame[fid] = [d for d in lidar_by_frame[fid]
                                       if np.sqrt(d['location'][0]**2 + d['location'][1]**2) <= max_r]
        filtered_gt = sum(len(v) for v in gt_by_frame.values())
        filtered_lidar = sum(len(lidar_by_frame.get(fid, [])) for fid in frame_ids)
        print(f"  After filter: {filtered_gt} GT objects, {filtered_lidar} LiDAR dets")

    # Load calibration
    fusion_sys = TripleViewFusion.from_dataset_dir(str(dataset_dir))

    # Load YOLO detections (from cache or by running inference)
    drone_dets_by_frame = {}
    ego_dets_by_frame = {}

    yolo_cache_path = args.yolo_cache
    if yolo_cache_path and os.path.exists(yolo_cache_path):
        print(f"Loading YOLO cache from {yolo_cache_path}")
        with open(yolo_cache_path, 'rb') as f:
            yolo_cache = pickle.load(f)
        drone_dets_by_frame = yolo_cache.get('drone', {})
        ego_dets_by_frame = yolo_cache.get('ego', {})
        print(f"  Loaded {len(drone_dets_by_frame)} drone, {len(ego_dets_by_frame)} ego cached detections")
    else:
        drone_model = None
        ego_model = None
        if args.drone_model:
            print("Loading Drone YOLO model...")
            from ultralytics import YOLO
            drone_model = YOLO(args.drone_model)
        if args.ego_model:
            print("Loading Ego YOLO model...")
            from ultralytics import YOLO
            ego_model = YOLO(args.ego_model)

        if drone_model or ego_model:
            print("Running YOLO inference...")
            for fid in tqdm(frame_ids, desc="YOLO"):
                if drone_model:
                    img_path = str(dataset_dir / "images_drone" / f"{fid}.jpg")
                    if os.path.exists(img_path):
                        drone_dets_by_frame[fid] = run_yolo_detection(drone_model, img_path)
                    else:
                        drone_dets_by_frame[fid] = []
                if ego_model:
                    img_path = str(dataset_dir / "images_sdc" / f"{fid}.jpg")
                    if os.path.exists(img_path):
                        ego_dets_by_frame[fid] = run_yolo_detection(ego_model, img_path)
                    else:
                        ego_dets_by_frame[fid] = []

        if yolo_cache_path:
            print(f"Saving YOLO cache to {yolo_cache_path}")
            with open(yolo_cache_path, 'wb') as f:
                pickle.dump({'drone': drone_dets_by_frame, 'ego': ego_dets_by_frame}, f)
            print(f"  Saved {len(drone_dets_by_frame)} drone, {len(ego_dets_by_frame)} ego detections")

    # ========================================================================
    # Run 4 configurations
    # ========================================================================
    class_names = ['Car', 'Pedestrian']
    iou_thresholds = [0.3, 0.5, 0.7]
    all_results = {}

    # Config 1: LiDAR Only
    print("\n--- Config 1: LiDAR Only ---")
    preds_lidar = {}
    for fid in frame_ids:
        preds_lidar[fid] = run_lidar_only(lidar_by_frame.get(fid, []))
    all_results['lidar_only'] = evaluate_config(preds_lidar, gt_by_frame, class_names, iou_thresholds)
    print(f"  mAP@0.5: {all_results['lidar_only']['mAP@0.5']:.4f}")

    # Config 2: LiDAR + SDC (boost-only — SDC has limited forward FOV)
    if ego_dets_by_frame:
        print("\n--- Config 2: LiDAR + SDC (boost-only) ---")
        preds_lidar_sdc = {}
        for fid in frame_ids:
            preds_lidar_sdc[fid] = run_lidar_plus_camera(
                lidar_by_frame.get(fid, []),
                ego_dets_by_frame.get(fid, []),
                fusion_sys.K_ego, fusion_sys.T_lidar_to_ego,
                allow_suppress=False,  # SDC: boost-only for robustness
            )
        all_results['lidar_sdc'] = evaluate_config(preds_lidar_sdc, gt_by_frame, class_names, iou_thresholds)
        print(f"  mAP@0.5: {all_results['lidar_sdc']['mAP@0.5']:.4f}")

    # Config 3: LiDAR + Drone (boost + suppress — drone has wide top-down FOV)
    if drone_dets_by_frame:
        print("\n--- Config 3: LiDAR + Drone (boost+suppress) ---")
        preds_lidar_drone = {}
        for fid in frame_ids:
            preds_lidar_drone[fid] = run_lidar_plus_camera(
                lidar_by_frame.get(fid, []),
                drone_dets_by_frame.get(fid, []),
                fusion_sys.K_drone, fusion_sys.T_lidar_to_drone,
                allow_suppress=True,  # Drone: suppress unconfirmed low-conf Cars
                suppress_factor=0.75,
                suppress_conf_gate=0.45,
                suppress_classes=('Car',),
            )
        all_results['lidar_drone'] = evaluate_config(preds_lidar_drone, gt_by_frame, class_names, iou_thresholds)
        print(f"  mAP@0.5: {all_results['lidar_drone']['mAP@0.5']:.4f}")

    # Config 4: LiDAR + SDC + Drone (enhanced fusion with camera proposals)
    if drone_dets_by_frame and ego_dets_by_frame:
        print("\n--- Config 4: LiDAR + SDC + Drone (Enhanced) ---")
        preds_full = {}
        for fid in frame_ids:
            preds_full[fid] = run_enhanced_fusion(
                lidar_by_frame.get(fid, []),
                drone_dets_by_frame.get(fid, []),
                ego_dets_by_frame.get(fid, []),
                fusion_sys.K_drone, fusion_sys.K_ego,
                fusion_sys.T_lidar_to_drone, fusion_sys.T_lidar_to_ego,
            )
        all_results['lidar_sdc_drone'] = evaluate_config(preds_full, gt_by_frame, class_names, iou_thresholds)
        print(f"  mAP@0.5: {all_results['lidar_sdc_drone']['mAP@0.5']:.4f}")

    # Config 5: Symmetric Fusion (ablation baseline — both cameras boost+suppress)
    if drone_dets_by_frame and ego_dets_by_frame:
        print("\n--- Config 5: Symmetric Fusion (both boost+suppress) ---")
        preds_symmetric = {}
        for fid in frame_ids:
            preds_symmetric[fid] = run_symmetric_fusion(
                lidar_by_frame.get(fid, []),
                drone_dets_by_frame.get(fid, []),
                ego_dets_by_frame.get(fid, []),
                fusion_sys.K_drone, fusion_sys.K_ego,
                fusion_sys.T_lidar_to_drone, fusion_sys.T_lidar_to_ego,
            )
        all_results['symmetric_fusion'] = evaluate_config(preds_symmetric, gt_by_frame, class_names, iou_thresholds)
        print(f"  mAP@0.5: {all_results['symmetric_fusion']['mAP@0.5']:.4f}")

    # Config 6: Naive Average Fusion (simplest possible baseline)
    if drone_dets_by_frame and ego_dets_by_frame:
        print("\n--- Config 6: Naive Average Fusion ---")
        preds_naive = {}
        for fid in frame_ids:
            preds_naive[fid] = run_naive_average_fusion(
                lidar_by_frame.get(fid, []),
                drone_dets_by_frame.get(fid, []),
                ego_dets_by_frame.get(fid, []),
                fusion_sys.K_drone, fusion_sys.K_ego,
                fusion_sys.T_lidar_to_drone, fusion_sys.T_lidar_to_ego,
            )
        all_results['naive_average'] = evaluate_config(preds_naive, gt_by_frame, class_names, iou_thresholds)
        print(f"  mAP@0.5: {all_results['naive_average']['mAP@0.5']:.4f}")

    # ========================================================================
    # Save results
    # ========================================================================
    tag = f"_{args.seed_tag}" if args.seed_tag else ""
    result_file = output_dir / f"eval_results{tag}.json"
    with open(result_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 80)
    print("ABLATION STUDY RESULTS")
    print("=" * 80)
    print(f"{'Config':<25} {'mAP@0.3':>10} {'mAP@0.5':>10} {'mAP@0.7':>10}")
    print("-" * 55)
    for config_name, config_results in all_results.items():
        print(f"{config_name:<25} "
              f"{config_results.get('mAP@0.3', 0):.4f}     "
              f"{config_results.get('mAP@0.5', 0):.4f}     "
              f"{config_results.get('mAP@0.7', 0):.4f}")

    print(f"\nPer-class breakdown (AP@0.5):")
    print(f"{'Config':<25} {'Car':>10} {'Pedestrian':>10}")
    print("-" * 45)
    for config_name, config_results in all_results.items():
        car_ap = config_results.get('Car', {}).get('AP@0.5', 0)
        ped_ap = config_results.get('Pedestrian', {}).get('AP@0.5', 0)
        print(f"{config_name:<25} {car_ap:.4f}     {ped_ap:.4f}")

    print(f"\nResults saved to: {result_file}")


if __name__ == "__main__":
    main()
