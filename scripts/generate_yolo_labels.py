"""
Generate YOLO-format 2D labels from 3D ground truth by projecting 3D boxes to camera views.

Projects 3D bounding boxes (in LiDAR frame) to both SDC and drone camera images
using known calibration transforms. Outputs YOLO format:
    class_id  x_center  y_center  width  height  (all normalized by image size)

This produces training data for fine-tuning YOLOv8 on CARLA synthetic images,
eliminating the domain gap between pretrained (real-world) YOLO and CARLA renders.

Usage:
    python generate_yolo_labels.py \
        --dataset /mnt/hdd12t/outputs/carla_out/dual_camera_lidar_town10hd_v2 \
        --output /mnt/hdd12t/outputs/carla_out/yolo_carla

Author: Xingnan Zhou
Date: 2026-02-10
"""

import argparse
import os
import shutil
from pathlib import Path

import numpy as np


# ===========================================================================
# 3D Box Projection (reuses logic from triple_view_fusion.py)
# ===========================================================================

# Camera intrinsic (both cameras identical: 1920x1280, FOV=90)
K = np.array([
    [960.0, 0.0, 960.0],
    [0.0, 960.0, 640.0],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

# LiDAR -> SDC camera OpenCV transform (forward-facing, pitch=0)
# SDC at (1.5, 0, 2.4), LiDAR at (0, 0, 2.5) in vehicle frame
R_sdc = np.array([
    [0,  1,  0],   # X_cv = Y (right)
    [0,  0, -1],   # Y_cv = -Z (down)
    [1,  0,  0]    # Z_cv = X (forward = depth)
], dtype=np.float64)
t_lidar_in_sdc = np.array([0.0, 0.0, 2.5 - 2.4])  # lidar - sdc positions
t_sdc_cv = R_sdc @ t_lidar_in_sdc  # [0, -0.1, -1.5]

T_lidar_to_sdc = np.eye(4, dtype=np.float64)
T_lidar_to_sdc[:3, :3] = R_sdc
T_lidar_to_sdc[:3, 3] = t_sdc_cv

# LiDAR -> Drone camera OpenCV transform (looking down, pitch=-90)
# Drone at (0, 0, 40), LiDAR at (0, 0, 2.5)
R_drone = np.array([
    [ 0,  1,  0],   # X_cv = Y (right)
    [-1,  0,  0],   # Y_cv = -X (backward = down in image)
    [ 0,  0, -1]    # Z_cv = -Z (depth = distance below drone)
], dtype=np.float64)
t_lidar_in_drone = np.array([0.0, 0.0, 2.5 - 40.0])  # lidar - drone
t_drone_cv = R_drone @ t_lidar_in_drone  # [0, 0, 37.5]

T_lidar_to_drone = np.eye(4, dtype=np.float64)
T_lidar_to_drone[:3, :3] = R_drone
T_lidar_to_drone[:3, 3] = t_drone_cv

IMAGE_W, IMAGE_H = 1920, 1280

# YOLO class mapping (same for both cameras)
CLASS_MAP = {'Car': 0, 'Pedestrian': 1}


def get_3d_box_corners(location, dimensions, rotation_y):
    """Get 8 corners of a 3D bounding box in LiDAR frame."""
    x, y, z = location
    l, w, h = dimensions

    corners = np.array([
        [-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2],
        [ w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],
        [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]
    ])

    # Rotation around Z-axis (yaw)
    c, s = np.cos(rotation_y), np.sin(rotation_y)
    R_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    corners = R_z @ corners + np.array([[x], [y], [z]])
    return corners  # 3x8


def project_box_to_2d(location, dimensions, rotation_y, T_lidar_to_cam, min_size=5):
    """
    Project a 3D box to 2D bounding box in camera image.

    Returns:
        (x_min, y_min, x_max, y_max) in pixel coordinates, or None if not visible.
    """
    corners = get_3d_box_corners(location, dimensions, rotation_y)  # 3x8
    corners_homo = np.vstack([corners, np.ones((1, 8))])  # 4x8

    corners_cam = (T_lidar_to_cam @ corners_homo)[:3, :]  # 3x8

    # Filter points behind camera
    valid = corners_cam[2, :] > 0.1
    if not valid.any():
        return None

    corners_valid = corners_cam[:, valid]

    # Project to 2D
    proj = K @ corners_valid  # 3xN
    u = proj[0, :] / proj[2, :]
    v = proj[1, :] / proj[2, :]

    x_min = max(0, float(u.min()))
    y_min = max(0, float(v.min()))
    x_max = min(IMAGE_W - 1, float(u.max()))
    y_max = min(IMAGE_H - 1, float(v.max()))

    # Minimum box size
    if (x_max - x_min) < min_size or (y_max - y_min) < min_size:
        return None

    return (x_min, y_min, x_max, y_max)


def parse_label_file(label_path):
    """Parse a 3D label file: x y z dx dy dz heading class_name."""
    labels = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            labels.append({
                'location': [float(parts[0]), float(parts[1]), float(parts[2])],
                'dimensions': [float(parts[3]), float(parts[4]), float(parts[5])],
                'rotation_y': float(parts[6]),
                'class_name': parts[7]
            })
    return labels


def generate_yolo_label(labels_3d, T_lidar_to_cam):
    """
    Convert 3D labels to YOLO format for one frame.

    Returns:
        List of YOLO label strings: "class_id cx cy w h" (normalized).
    """
    yolo_lines = []
    for label in labels_3d:
        cls_name = label['class_name']
        if cls_name not in CLASS_MAP:
            continue

        box_2d = project_box_to_2d(
            label['location'], label['dimensions'],
            label['rotation_y'], T_lidar_to_cam
        )
        if box_2d is None:
            continue

        x_min, y_min, x_max, y_max = box_2d
        cls_id = CLASS_MAP[cls_name]

        # YOLO format: normalized center + size
        cx = ((x_min + x_max) / 2.0) / IMAGE_W
        cy = ((y_min + y_max) / 2.0) / IMAGE_H
        w = (x_max - x_min) / IMAGE_W
        h = (y_max - y_min) / IMAGE_H

        # Sanity check
        if cx < 0 or cx > 1 or cy < 0 or cy > 1 or w <= 0 or h <= 0:
            continue

        yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return yolo_lines


def setup_yolo_dataset(output_dir, cam_name, dataset_dir, imagesets_dir):
    """
    Set up YOLO dataset directory structure with symlinked images.

    Structure:
        output_dir/cam_name/
            images/train/  (symlinks to source images)
            images/val/
            labels/train/
            labels/val/
            data.yaml
    """
    cam_dir = Path(output_dir) / cam_name
    for split in ['train', 'val']:
        (cam_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (cam_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Read imagesets
    splits = {}
    for split in ['train', 'val']:
        split_file = Path(imagesets_dir) / f'{split}.txt'
        with open(split_file) as f:
            splits[split] = [line.strip() for line in f if line.strip()]

    # Create symlinks for images
    img_dir_name = f'images_{cam_name}'  # images_sdc or images_drone
    src_img_dir = Path(dataset_dir) / img_dir_name

    for split, frame_ids in splits.items():
        for fid in frame_ids:
            src = src_img_dir / f'{fid}.jpg'
            dst = cam_dir / 'images' / split / f'{fid}.jpg'
            if src.exists() and not dst.exists():
                os.symlink(src, dst)

    # Create data.yaml
    yaml_content = f"""path: {cam_dir}
train: images/train
val: images/val

nc: {len(CLASS_MAP)}
names: {list(CLASS_MAP.keys())}
"""
    with open(cam_dir / 'data.yaml', 'w') as f:
        f.write(yaml_content)

    return splits


def main():
    parser = argparse.ArgumentParser(description="Generate YOLO 2D labels from 3D GT")
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to dual_camera_lidar dataset')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for YOLO datasets')
    parser.add_argument('--min-box-size', type=int, default=10,
                        help='Minimum 2D box size in pixels')
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    output_dir = Path(args.output)
    labels_dir = dataset_dir / 'labels'
    imagesets_dir = dataset_dir / 'ImageSets'

    cameras = {
        'sdc': T_lidar_to_sdc,
        'drone': T_lidar_to_drone,
    }

    total_stats = {}

    for cam_name, T_lidar_to_cam in cameras.items():
        print(f"\n{'='*60}")
        print(f"Processing {cam_name.upper()} camera")
        print(f"{'='*60}")

        # Setup directory structure
        splits = setup_yolo_dataset(output_dir, cam_name, dataset_dir, imagesets_dir)

        stats = {'total_3d': 0, 'total_2d': 0, 'per_class': {}, 'per_split': {}}
        for cls_name in CLASS_MAP:
            stats['per_class'][cls_name] = {'total_3d': 0, 'total_2d': 0}

        for split, frame_ids in splits.items():
            split_stats = {'frames': 0, 'labels': 0, 'empty_frames': 0}

            for fid in frame_ids:
                label_path = labels_dir / f'{fid}.txt'
                if not label_path.exists():
                    continue

                labels_3d = parse_label_file(label_path)
                yolo_lines = generate_yolo_label(labels_3d, T_lidar_to_cam)

                # Count per-class stats
                for label in labels_3d:
                    cls_name = label['class_name']
                    if cls_name in CLASS_MAP:
                        stats['total_3d'] += 1
                        stats['per_class'][cls_name]['total_3d'] += 1

                stats['total_2d'] += len(yolo_lines)
                for line in yolo_lines:
                    cls_id = int(line.split()[0])
                    cls_name = [k for k, v in CLASS_MAP.items() if v == cls_id][0]
                    stats['per_class'][cls_name]['total_2d'] += 1

                # Write YOLO label file
                out_label = output_dir / cam_name / 'labels' / split / f'{fid}.txt'
                with open(out_label, 'w') as f:
                    f.write('\n'.join(yolo_lines) + '\n' if yolo_lines else '')

                split_stats['frames'] += 1
                split_stats['labels'] += len(yolo_lines)
                if not yolo_lines:
                    split_stats['empty_frames'] += 1

            stats['per_split'][split] = split_stats
            print(f"  {split}: {split_stats['frames']} frames, "
                  f"{split_stats['labels']} labels, "
                  f"{split_stats['empty_frames']} empty frames")

        # Print summary
        visibility = stats['total_2d'] / max(stats['total_3d'], 1) * 100
        print(f"\n  Visibility rate: {stats['total_2d']}/{stats['total_3d']} "
              f"= {visibility:.1f}%")

        for cls_name, cls_stats in stats['per_class'].items():
            vis = cls_stats['total_2d'] / max(cls_stats['total_3d'], 1) * 100
            print(f"    {cls_name}: {cls_stats['total_2d']}/{cls_stats['total_3d']} "
                  f"= {vis:.1f}%")

        total_stats[cam_name] = stats

    # Final summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for cam_name, stats in total_stats.items():
        vis = stats['total_2d'] / max(stats['total_3d'], 1) * 100
        train = stats['per_split'].get('train', {})
        val = stats['per_split'].get('val', {})
        print(f"{cam_name.upper()}: {stats['total_2d']} 2D labels "
              f"({vis:.1f}% visibility), "
              f"train={train.get('labels', 0)}, val={val.get('labels', 0)}")

    print(f"\nYOLO datasets saved to: {output_dir}")
    print(f"  SDC data.yaml:   {output_dir}/sdc/data.yaml")
    print(f"  Drone data.yaml: {output_dir}/drone/data.yaml")


if __name__ == '__main__':
    main()
