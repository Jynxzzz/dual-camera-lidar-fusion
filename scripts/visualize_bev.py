#!/usr/bin/env python3
"""
BEV (Bird's Eye View) Visualization for LiDAR Point Clouds.

Generates publication-quality BEV images from CARLA LiDAR data with
ground truth bounding boxes and optional detection predictions.

Usage:
    python visualize_bev.py --frame_id 000100
    python visualize_bev.py --frame_id 000100,001000,002000
    python visualize_bev.py --frame_id 000100 --pred_dir /path/to/predictions

Label format (per line):
    x y z dx dy dz heading_angle class_name

Point cloud format:
    .npy file with shape (N, 4) -- columns: x, y, z, intensity

Author: Xingnan Zhou, Concordia University
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless rendering

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import numpy as np
from matplotlib.collections import LineCollection


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Point cloud range: [x_min, y_min, z_min, x_max, y_max, z_max]
PC_RANGE = [-70.4, -70.4, -3, 70.4, 70.4, 10]

# Colours for ground truth boxes
CLASS_COLORS_GT = {
    'Car':        '#2ca02c',   # green
    'Pedestrian': '#1f77b4',   # blue
}

# Colours for prediction boxes
CLASS_COLORS_PRED = {
    'Car':        '#d62728',   # red
    'Pedestrian': '#17becf',   # cyan
}

POINT_COLOR_CMAP = 'gray'       # colormap for intensity
RANGE_RING_COLOR = '#888888'
RANGE_RING_STYLE = '--'
RANGE_RING_ALPHA = 0.5
RANGE_RING_INTERVALS = [20, 40, 60]  # metres

# Figure styling
FIGURE_SIZE = (10, 10)           # inches
FIGURE_DPI = 300                 # publication quality
BACKGROUND_COLOR = '#0a0a0a'     # near-black background
AXIS_LABEL_SIZE = 12
TITLE_SIZE = 14
TICK_SIZE = 10
LEGEND_SIZE = 10

# Box drawing
GT_BOX_LINEWIDTH = 1.5
PRED_BOX_LINEWIDTH = 1.5
GT_BOX_LINESTYLE = '-'
PRED_BOX_LINESTYLE = '--'


# ---------------------------------------------------------------------------
# Data I/O
# ---------------------------------------------------------------------------

def load_point_cloud(filepath: str) -> np.ndarray:
    """Load a .npy point cloud file with columns [x, y, z, intensity].

    Args:
        filepath: Path to the .npy file.

    Returns:
        np.ndarray of shape (N, 4).
    """
    points = np.load(filepath)
    assert points.ndim == 2 and points.shape[1] >= 4, (
        f"Expected (N, >=4) array, got {points.shape}"
    )
    return points[:, :4]


def load_labels(filepath: str) -> list:
    """Load ground truth labels from a text file.

    Format per line: x y z dx dy dz heading_angle class_name

    Args:
        filepath: Path to the label .txt file.

    Returns:
        List of dicts with keys: x, y, z, dx, dy, dz, heading, class_name.
    """
    labels = []
    if not os.path.isfile(filepath):
        return labels
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            labels.append({
                'x':          float(parts[0]),
                'y':          float(parts[1]),
                'z':          float(parts[2]),
                'dx':         float(parts[3]),  # length
                'dy':         float(parts[4]),  # width
                'dz':         float(parts[5]),  # height
                'heading':    float(parts[6]),
                'class_name': parts[7],
            })
    return labels


def load_predictions(filepath: str) -> list:
    """Load detection predictions (same format as labels).

    Args:
        filepath: Path to the prediction .txt file.

    Returns:
        List of dicts (same schema as load_labels).
    """
    return load_labels(filepath)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def oriented_box_corners(cx, cy, dx, dy, heading):
    """Compute the four corners of an oriented 2D bounding box.

    Args:
        cx, cy: Centre coordinates.
        dx: Length (along heading direction).
        dy: Width  (perpendicular to heading).
        heading: Rotation angle in radians (counter-clockwise from x-axis).

    Returns:
        np.ndarray of shape (5, 2) -- four corners + first corner repeated
        for a closed polygon.
    """
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)

    # Half extents
    hdx = dx / 2.0
    hdy = dy / 2.0

    # Corner offsets in local frame (length along x, width along y)
    corners_local = np.array([
        [ hdx,  hdy],
        [ hdx, -hdy],
        [-hdx, -hdy],
        [-hdx,  hdy],
    ])

    # Rotation matrix
    R = np.array([
        [cos_h, -sin_h],
        [sin_h,  cos_h],
    ])

    corners_world = corners_local @ R.T + np.array([cx, cy])

    # Close the polygon
    corners_closed = np.vstack([corners_world, corners_world[0:1]])
    return corners_closed


def draw_heading_arrow(ax, cx, cy, dx, heading, color, linewidth=1.0):
    """Draw a small arrow from box centre along the heading direction.

    This indicates the front of the object.
    """
    arrow_len = dx * 0.4  # 40% of object length
    end_x = cx + arrow_len * np.cos(heading)
    end_y = cy + arrow_len * np.sin(heading)
    ax.annotate(
        '', xy=(end_x, end_y), xytext=(cx, cy),
        arrowprops=dict(
            arrowstyle='->', color=color, lw=linewidth,
            mutation_scale=8,
        ),
    )


# ---------------------------------------------------------------------------
# Main visualisation
# ---------------------------------------------------------------------------

def draw_bev(
    points: np.ndarray,
    gt_labels: list,
    pred_labels: list = None,
    x_range: tuple = None,
    y_range: tuple = None,
    title: str = 'Bird\'s Eye View',
    output_path: str = None,
    show_heading: bool = True,
    point_size: float = 0.3,
):
    """Generate a publication-quality BEV image.

    Args:
        points:      (N, 4) array with columns [x, y, z, intensity].
        gt_labels:   List of ground truth label dicts.
        pred_labels: Optional list of prediction label dicts.
        x_range:     (x_min, x_max) in metres. Defaults to PC_RANGE.
        y_range:     (y_min, y_max) in metres. Defaults to PC_RANGE.
        title:       Figure title string.
        output_path: If given, save to this path. Otherwise plt.show().
        show_heading: Draw heading arrows on boxes.
        point_size:  Scatter point size.
    """
    if x_range is None:
        x_range = (PC_RANGE[0], PC_RANGE[3])
    if y_range is None:
        y_range = (PC_RANGE[1], PC_RANGE[4])

    # -----------------------------------------------------------------------
    # Set up figure with dark background
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=FIGURE_SIZE,
        facecolor=BACKGROUND_COLOR,
    )
    ax.set_facecolor(BACKGROUND_COLOR)

    # -----------------------------------------------------------------------
    # 1. Draw point cloud (top-down: x vs y, coloured by intensity)
    # -----------------------------------------------------------------------
    mask = (
        (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) &
        (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1])
    )
    pts = points[mask]

    # Normalise intensity to [0, 1] for colormap
    intensity = pts[:, 3]
    i_min, i_max = intensity.min(), intensity.max()
    if i_max > i_min:
        intensity_norm = (intensity - i_min) / (i_max - i_min)
    else:
        intensity_norm = np.ones_like(intensity)

    ax.scatter(
        pts[:, 0], pts[:, 1],
        c=intensity_norm,
        cmap=POINT_COLOR_CMAP,
        s=point_size,
        alpha=0.8,
        edgecolors='none',
        rasterized=True,  # faster rendering in PDF/PNG
    )

    # -----------------------------------------------------------------------
    # 2. Draw range rings centred at ego (0, 0)
    # -----------------------------------------------------------------------
    for radius in RANGE_RING_INTERVALS:
        circle = plt.Circle(
            (0, 0), radius,
            fill=False,
            color=RANGE_RING_COLOR,
            linestyle=RANGE_RING_STYLE,
            linewidth=0.8,
            alpha=RANGE_RING_ALPHA,
        )
        ax.add_patch(circle)
        # Label the ring
        ax.text(
            radius * np.cos(np.pi / 4),
            radius * np.sin(np.pi / 4) + 1.5,
            f'{radius} m',
            color=RANGE_RING_COLOR,
            fontsize=8,
            ha='center',
            alpha=0.7,
        )

    # -----------------------------------------------------------------------
    # 3. Draw ego vehicle marker
    # -----------------------------------------------------------------------
    ax.plot(0, 0, marker='o', color='#ff7f0e', markersize=6, zorder=10)
    ax.text(
        1.5, 1.5, 'EGO', color='#ff7f0e',
        fontsize=9, fontweight='bold', zorder=10,
    )

    # -----------------------------------------------------------------------
    # 4. Draw ground truth bounding boxes
    # -----------------------------------------------------------------------
    gt_class_drawn = set()
    for label in gt_labels:
        cls = label['class_name']
        color = CLASS_COLORS_GT.get(cls, '#ffffff')
        corners = oriented_box_corners(
            label['x'], label['y'],
            label['dx'], label['dy'],
            label['heading'],
        )
        ax.plot(
            corners[:, 0], corners[:, 1],
            color=color,
            linewidth=GT_BOX_LINEWIDTH,
            linestyle=GT_BOX_LINESTYLE,
            zorder=5,
        )
        if show_heading:
            draw_heading_arrow(
                ax, label['x'], label['y'],
                label['dx'], label['heading'],
                color=color, linewidth=1.0,
            )
        gt_class_drawn.add(cls)

    # -----------------------------------------------------------------------
    # 5. Draw prediction bounding boxes (if provided)
    # -----------------------------------------------------------------------
    pred_class_drawn = set()
    if pred_labels:
        for label in pred_labels:
            cls = label['class_name']
            color = CLASS_COLORS_PRED.get(cls, '#aaaaaa')
            corners = oriented_box_corners(
                label['x'], label['y'],
                label['dx'], label['dy'],
                label['heading'],
            )
            ax.plot(
                corners[:, 0], corners[:, 1],
                color=color,
                linewidth=PRED_BOX_LINEWIDTH,
                linestyle=PRED_BOX_LINESTYLE,
                zorder=6,
            )
            if show_heading:
                draw_heading_arrow(
                    ax, label['x'], label['y'],
                    label['dx'], label['heading'],
                    color=color, linewidth=1.0,
                )
            pred_class_drawn.add(cls)

    # -----------------------------------------------------------------------
    # 6. Legend
    # -----------------------------------------------------------------------
    legend_handles = []

    # GT entries
    for cls in sorted(gt_class_drawn):
        color = CLASS_COLORS_GT[cls]
        handle = mlines.Line2D(
            [], [], color=color, linewidth=GT_BOX_LINEWIDTH,
            linestyle=GT_BOX_LINESTYLE,
            label=f'GT {cls}',
        )
        legend_handles.append(handle)

    # Pred entries
    for cls in sorted(pred_class_drawn):
        color = CLASS_COLORS_PRED[cls]
        handle = mlines.Line2D(
            [], [], color=color, linewidth=PRED_BOX_LINEWIDTH,
            linestyle=PRED_BOX_LINESTYLE,
            label=f'Pred {cls}',
        )
        legend_handles.append(handle)

    # Points
    legend_handles.append(
        mlines.Line2D(
            [], [], color='gray', marker='o', linestyle='None',
            markersize=3, label='LiDAR Points',
        )
    )
    # Ego
    legend_handles.append(
        mlines.Line2D(
            [], [], color='#ff7f0e', marker='o', linestyle='None',
            markersize=6, label='Ego Vehicle',
        )
    )
    # Range rings
    legend_handles.append(
        mlines.Line2D(
            [], [], color=RANGE_RING_COLOR, linewidth=0.8,
            linestyle=RANGE_RING_STYLE, alpha=RANGE_RING_ALPHA,
            label='Range Rings',
        )
    )

    legend = ax.legend(
        handles=legend_handles,
        loc='upper right',
        fontsize=LEGEND_SIZE,
        framealpha=0.7,
        facecolor='#222222',
        edgecolor='#555555',
        labelcolor='white',
    )

    # -----------------------------------------------------------------------
    # 7. Axis labels, ticks, title
    # -----------------------------------------------------------------------
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_aspect('equal')

    ax.set_xlabel('X (m)', fontsize=AXIS_LABEL_SIZE, color='white')
    ax.set_ylabel('Y (m)', fontsize=AXIS_LABEL_SIZE, color='white')
    ax.set_title(title, fontsize=TITLE_SIZE, color='white', pad=12)

    ax.tick_params(
        axis='both', colors='white',
        labelsize=TICK_SIZE, direction='in',
    )
    for spine in ax.spines.values():
        spine.set_color('#555555')

    # Grid
    ax.grid(True, color='#333333', linewidth=0.3, alpha=0.5)

    # -----------------------------------------------------------------------
    # 8. Add statistics annotation
    # -----------------------------------------------------------------------
    n_pts = len(pts)
    n_gt_car = sum(1 for l in gt_labels if l['class_name'] == 'Car')
    n_gt_ped = sum(1 for l in gt_labels if l['class_name'] == 'Pedestrian')
    stats_text = f'Points: {n_pts:,}  |  GT Cars: {n_gt_car}  |  GT Peds: {n_gt_ped}'
    if pred_labels:
        n_pred_car = sum(1 for l in pred_labels if l['class_name'] == 'Car')
        n_pred_ped = sum(1 for l in pred_labels if l['class_name'] == 'Pedestrian')
        stats_text += f'  |  Pred Cars: {n_pred_car}  |  Pred Peds: {n_pred_ped}'

    ax.text(
        0.02, 0.02, stats_text,
        transform=ax.transAxes,
        fontsize=8, color='#cccccc',
        verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a1a', alpha=0.8),
    )

    # -----------------------------------------------------------------------
    # 9. Save or show
    # -----------------------------------------------------------------------
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(
            output_path,
            dpi=FIGURE_DPI,
            facecolor=fig.get_facecolor(),
            bbox_inches='tight',
            pad_inches=0.1,
        )
        print(f"  [SAVED] {output_path}")
    else:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate publication-quality BEV images from LiDAR data.',
    )
    parser.add_argument(
        '--frame_id', type=str, required=True,
        help='Frame ID(s), comma-separated (e.g., "000100" or "000100,001000,002000").',
    )
    parser.add_argument(
        '--data_dir', type=str,
        default='/home/xingnan/LidarDetection/OpenPCDet/data/carla',
        help='Root directory of the CARLA dataset.',
    )
    parser.add_argument(
        '--pred_dir', type=str, default=None,
        help='Directory containing prediction label files (same naming as GT).',
    )
    parser.add_argument(
        '--output_dir', type=str,
        default='/home/xingnan/projects/bev_lidar_perception/figures',
        help='Directory to save output images.',
    )
    parser.add_argument(
        '--x_range', type=float, nargs=2, default=[-70.4, 70.4],
        help='X-axis range in metres (min max). Default: -70.4 70.4',
    )
    parser.add_argument(
        '--y_range', type=float, nargs=2, default=[-70.4, 70.4],
        help='Y-axis range in metres (min max). Default: -70.4 70.4',
    )
    parser.add_argument(
        '--point_size', type=float, default=0.3,
        help='Size of LiDAR points in the scatter plot.',
    )
    parser.add_argument(
        '--no_heading', action='store_true',
        help='Disable heading arrows on bounding boxes.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    frame_ids = [fid.strip() for fid in args.frame_id.split(',')]

    print(f"BEV Visualization  --  {len(frame_ids)} frame(s)")
    print(f"  Data directory : {args.data_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  X range: {args.x_range}")
    print(f"  Y range: {args.y_range}")
    if args.pred_dir:
        print(f"  Prediction dir : {args.pred_dir}")

    for fid in frame_ids:
        print(f"\n{'='*60}")
        print(f"  Processing frame: {fid}")
        print(f"{'='*60}")

        # --- Paths ---
        points_path = os.path.join(args.data_dir, 'points', f'{fid}.npy')
        labels_path = os.path.join(args.data_dir, 'labels', f'{fid}.txt')
        output_path = os.path.join(args.output_dir, f'bev_{fid}.png')

        # --- Load point cloud ---
        if not os.path.isfile(points_path):
            print(f"  [ERROR] Point cloud not found: {points_path}")
            continue
        points = load_point_cloud(points_path)
        print(f"  Loaded {points.shape[0]:,} points from {points_path}")

        # --- Load GT labels ---
        gt_labels = load_labels(labels_path)
        n_cars = sum(1 for l in gt_labels if l['class_name'] == 'Car')
        n_peds = sum(1 for l in gt_labels if l['class_name'] == 'Pedestrian')
        print(f"  Loaded {len(gt_labels)} GT labels ({n_cars} Cars, {n_peds} Pedestrians)")

        # --- Load predictions (optional) ---
        pred_labels = None
        if args.pred_dir:
            pred_path = os.path.join(args.pred_dir, f'{fid}.txt')
            if os.path.isfile(pred_path):
                pred_labels = load_predictions(pred_path)
                print(f"  Loaded {len(pred_labels)} predictions from {pred_path}")
            else:
                print(f"  [WARN] Prediction file not found: {pred_path}")

        # --- Draw BEV ---
        title = f'BEV -- Frame {fid}'
        draw_bev(
            points=points,
            gt_labels=gt_labels,
            pred_labels=pred_labels,
            x_range=tuple(args.x_range),
            y_range=tuple(args.y_range),
            title=title,
            output_path=output_path,
            show_heading=not args.no_heading,
            point_size=args.point_size,
        )

    print(f"\n[DONE] All frames processed. Output directory: {args.output_dir}")


if __name__ == '__main__':
    main()
