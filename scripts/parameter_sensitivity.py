#!/usr/bin/env python3
"""
Parameter Sensitivity Analysis for Late-Fusion 3D Detection.

Sweeps each fusion hyper-parameter one at a time (others held at defaults)
and evaluates symmetric fusion mAP@0.5 on seed 0 val split.

Produces:
  - sensitivity_results.json   (all sweep data)
  - fig_sensitivity.pdf / .png (publication-quality 5-panel figure)

Author: Xingnan Zhou
Date: 2026-02-12
"""

import json
import os
import pickle
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup  (import fusion helpers from the project scripts directory)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_fusion import (
    evaluate_config,
    load_all_lidar_detections,
    load_gt_labels,
    run_lidar_only,
    run_symmetric_fusion,
)
from triple_view_fusion import TripleViewFusion

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_DIR = Path("/mnt/hdd12t/outputs/carla_out/dual_camera_lidar_town10hd_v3")
RESULT_PKL = Path(
    "/home/xingnan/LidarDetection/OpenPCDet/output/cfgs/carla_models/"
    "pointpillar/v3_seed0/eval/eval_with_train/epoch_80/val/result.pkl"
)
YOLO_CACHE = Path(
    "/mnt/hdd12t/outputs/carla_out/fusion_eval_v3_pointpillar/yolo_cache_val.pkl"
)
OUTPUT_DIR = Path(
    "/mnt/hdd12t/outputs/carla_out/fusion_eval_v3_pointpillar/sensitivity_analysis"
)

MAX_RANGE = 50.0  # metres — BEV distance filter
CLASS_NAMES = ["Car", "Pedestrian"]
IOU_THRESHOLDS = [0.5]

# Default hyper-parameters (the chosen operating point)
DEFAULTS = OrderedDict(
    [
        ("boost_single", 1.15),
        ("boost_dual", 1.30),
        ("suppress_factor", 0.75),
        ("suppress_conf_gate", 0.45),
        ("match_iou_thresh", 0.30),
    ]
)

# Sweep ranges for each parameter
SWEEP_RANGES = OrderedDict(
    [
        ("boost_single", [1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30]),
        ("boost_dual", [1.00, 1.10, 1.20, 1.30, 1.40, 1.50]),
        ("suppress_factor", [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]),
        ("suppress_conf_gate", [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]),
        ("match_iou_thresh", [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]),
    ]
)

# Pretty labels for plotting (LaTeX math where appropriate)
PARAM_LABELS = {
    "boost_single": r"$\beta_{\mathrm{single}}$",
    "boost_dual": r"$\beta_{\mathrm{dual}}$",
    "suppress_factor": r"$\gamma$",
    "suppress_conf_gate": r"$\theta_{\mathrm{low}}$",
    "match_iou_thresh": r"$\tau_{\mathrm{IoU}}$",
}

# Subplot panel labels
PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)", "(e)"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bev_distance(loc):
    """Euclidean BEV distance from the LiDAR origin."""
    return np.sqrt(loc[0] ** 2 + loc[1] ** 2)


def filter_by_range(det_list, max_range):
    """Keep only detections within max BEV range."""
    return [d for d in det_list if bev_distance(d["location"]) <= max_range]


def run_fusion_with_params(
    lidar_by_frame,
    drone_dets_by_frame,
    ego_dets_by_frame,
    frame_ids,
    fusion_sys,
    params,
):
    """
    Run symmetric fusion for every val frame with the given parameter dict.

    Returns:
        predictions_by_frame: {frame_id: [det, ...]}
    """
    predictions_by_frame = {}
    for fid in frame_ids:
        lidar_dets = lidar_by_frame.get(fid, [])
        drone_dets_2d = drone_dets_by_frame.get(fid, [])
        ego_dets_2d = ego_dets_by_frame.get(fid, [])

        fused = run_symmetric_fusion(
            lidar_dets,
            drone_dets_2d,
            ego_dets_2d,
            fusion_sys.K_drone,
            fusion_sys.K_ego,
            fusion_sys.T_lidar_to_drone,
            fusion_sys.T_lidar_to_ego,
            conf_threshold=0.3,  # base conf threshold is NOT swept
            boost_single=params["boost_single"],
            boost_dual=params["boost_dual"],
            match_iou_thresh=params["match_iou_thresh"],
            suppress_factor=params["suppress_factor"],
            suppress_conf_gate=params["suppress_conf_gate"],
            suppress_classes=("Car",),
        )
        predictions_by_frame[fid] = fused

    return predictions_by_frame


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep():
    """Execute the full sensitivity sweep and return results dict."""

    # ---- Load data --------------------------------------------------------
    print("=" * 70)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 70)

    # Frame IDs
    split_file = DATASET_DIR / "ImageSets" / "val.txt"
    with open(split_file, "r") as f:
        frame_ids = [line.strip() for line in f if line.strip()]
    print(f"Val frames: {len(frame_ids)}")

    # Ground truth
    print("Loading ground truth...")
    gt_by_frame = {}
    for fid in frame_ids:
        labels = load_gt_labels(str(DATASET_DIR / "labels" / f"{fid}.txt"))
        gt_by_frame[fid] = filter_by_range(labels, MAX_RANGE)
    total_gt = sum(len(v) for v in gt_by_frame.values())
    print(f"  GT objects (within {MAX_RANGE}m): {total_gt}")

    # LiDAR detections
    print("Loading LiDAR detections...")
    lidar_by_frame_raw = load_all_lidar_detections(str(RESULT_PKL))
    lidar_by_frame = {}
    for fid in frame_ids:
        dets = lidar_by_frame_raw.get(fid, [])
        lidar_by_frame[fid] = filter_by_range(dets, MAX_RANGE)
    total_lidar = sum(len(v) for v in lidar_by_frame.values())
    print(f"  LiDAR detections (within {MAX_RANGE}m): {total_lidar}")

    # YOLO cache
    print(f"Loading YOLO cache from {YOLO_CACHE}...")
    with open(YOLO_CACHE, "rb") as f:
        yolo_cache = pickle.load(f)
    drone_dets_by_frame = yolo_cache.get("drone", {})
    ego_dets_by_frame = yolo_cache.get("ego", {})
    print(
        f"  Drone frames: {len(drone_dets_by_frame)}, "
        f"Ego frames: {len(ego_dets_by_frame)}"
    )

    # Calibration
    print("Loading calibration...")
    fusion_sys = TripleViewFusion.from_dataset_dir(str(DATASET_DIR))

    # ---- LiDAR-only baseline (constant across all sweeps) -----------------
    print("\nComputing LiDAR-only baseline...")
    preds_baseline = {}
    for fid in frame_ids:
        preds_baseline[fid] = run_lidar_only(lidar_by_frame.get(fid, []), conf_threshold=0.3)
    baseline_results = evaluate_config(preds_baseline, gt_by_frame, CLASS_NAMES, IOU_THRESHOLDS)
    baseline_mAP = baseline_results["mAP@0.5"]
    baseline_car_ap = baseline_results["Car"]["AP@0.5"]
    baseline_ped_ap = baseline_results["Pedestrian"]["AP@0.5"]
    print(f"  LiDAR-only mAP@0.5 = {baseline_mAP:.6f}")
    print(f"    Car AP@0.5 = {baseline_car_ap:.6f}")
    print(f"    Ped AP@0.5 = {baseline_ped_ap:.6f}")

    # ---- Sweep each parameter ---------------------------------------------
    all_sweep_results = {}
    total_evals = sum(len(v) for v in SWEEP_RANGES.values())
    eval_count = 0

    for param_name, values in SWEEP_RANGES.items():
        print(f"\n--- Sweeping {param_name} ({PARAM_LABELS[param_name]}) ---")
        sweep_data = {
            "param_name": param_name,
            "param_label": PARAM_LABELS[param_name],
            "default_value": DEFAULTS[param_name],
            "values": [],
            "mAP_05": [],
            "car_AP_05": [],
            "ped_AP_05": [],
        }

        for val in values:
            eval_count += 1
            # Build parameter dict: sweep this one, keep others at default
            params = dict(DEFAULTS)
            params[param_name] = val

            t0 = time.time()
            preds = run_fusion_with_params(
                lidar_by_frame,
                drone_dets_by_frame,
                ego_dets_by_frame,
                frame_ids,
                fusion_sys,
                params,
            )
            results = evaluate_config(preds, gt_by_frame, CLASS_NAMES, IOU_THRESHOLDS)
            dt = time.time() - t0

            mAP = results["mAP@0.5"]
            car_ap = results["Car"]["AP@0.5"]
            ped_ap = results["Pedestrian"]["AP@0.5"]

            sweep_data["values"].append(val)
            sweep_data["mAP_05"].append(mAP)
            sweep_data["car_AP_05"].append(car_ap)
            sweep_data["ped_AP_05"].append(ped_ap)

            marker = " <-- default" if abs(val - DEFAULTS[param_name]) < 1e-9 else ""
            print(
                f"  [{eval_count}/{total_evals}] {param_name}={val:.2f}  "
                f"mAP@0.5={mAP:.6f}  Car={car_ap:.6f}  Ped={ped_ap:.6f}  "
                f"({dt:.1f}s){marker}"
            )

        all_sweep_results[param_name] = sweep_data

    # ---- Package results --------------------------------------------------
    output = {
        "description": "Parameter sensitivity analysis for symmetric late-fusion 3D detection",
        "dataset": str(DATASET_DIR),
        "lidar_results": str(RESULT_PKL),
        "max_range_m": MAX_RANGE,
        "class_names": CLASS_NAMES,
        "iou_threshold": 0.5,
        "defaults": dict(DEFAULTS),
        "baseline": {
            "config": "lidar_only",
            "mAP_05": baseline_mAP,
            "car_AP_05": baseline_car_ap,
            "ped_AP_05": baseline_ped_ap,
        },
        "sweeps": {k: v for k, v in all_sweep_results.items()},
    }

    return output


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def generate_figure(results, output_dir):
    """
    Create a publication-quality 2x3 multi-panel figure.
    Panels (a)-(e): parameter sweeps with shared y-axis scale.
    Panel (f): horizontal bar chart of mAP swing per parameter (parameter importance).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    # ---- MDPI / journal style settings ------------------------------------
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "lines.linewidth": 1.2,
        "lines.markersize": 5,
    })

    baseline_mAP = results["baseline"]["mAP_05"]
    sweep_keys = list(SWEEP_RANGES.keys())

    # ---- Compute shared y-axis range across all 5 sweep panels -----------
    global_ymin = baseline_mAP
    global_ymax = baseline_mAP
    for pn in sweep_keys:
        vals = results["sweeps"][pn]["mAP_05"]
        global_ymin = min(global_ymin, min(vals))
        global_ymax = max(global_ymax, max(vals))
    y_pad = (global_ymax - global_ymin) * 0.12
    shared_ylim = (global_ymin - y_pad, global_ymax + y_pad)

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4))
    axes = axes.flatten()

    for i, param_name in enumerate(sweep_keys):
        ax = axes[i]
        sweep = results["sweeps"][param_name]

        x_vals = sweep["values"]
        y_vals = sweep["mAP_05"]
        default_val = sweep["default_value"]

        # Find default index for star marker
        default_idx = None
        for j, v in enumerate(x_vals):
            if abs(v - default_val) < 1e-9:
                default_idx = j
                break

        # Light green shaded region above baseline (improvement zone)
        ax.axhspan(
            baseline_mAP, shared_ylim[1],
            color="#e8f5e9", alpha=0.5, zorder=1,
        )

        # Fusion mAP line
        ax.plot(
            x_vals,
            y_vals,
            color="#2166ac",
            marker="o",
            markersize=3.5,
            markerfacecolor="#2166ac",
            markeredgecolor="#2166ac",
            linewidth=1.3,
            zorder=3,
            label="Symmetric fusion",
        )

        # Star marker at default operating point
        if default_idx is not None:
            ax.plot(
                x_vals[default_idx],
                y_vals[default_idx],
                marker="*",
                markersize=11,
                markerfacecolor="#e31a1c",
                markeredgecolor="#e31a1c",
                zorder=5,
                linestyle="None",
                label=f"Default ({default_val})",
            )

        # LiDAR-only baseline dashed line
        ax.axhline(
            y=baseline_mAP,
            color="#666666",
            linestyle="--",
            linewidth=0.9,
            zorder=2,
            label="LiDAR only",
        )

        # Axis labels and panel label
        ax.set_xlabel(PARAM_LABELS[param_name])
        if i % 3 == 0:
            ax.set_ylabel("mAP@0.5")
        else:
            ax.set_ylabel("")

        # Panel label (a), (b), etc. in top-left corner
        ax.text(
            0.04,
            0.96,
            PANEL_LABELS[i],
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
            ha="left",
        )

        # Shared y-axis range for visual comparability
        ax.set_ylim(shared_ylim)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

        # Clean MDPI style
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out")

        # Legend — small, no frame, only on panel (a)
        if i == 0:
            ax.legend(
                loc="lower left",
                frameon=False,
                fontsize=7,
                handlelength=1.5,
            )

    # ---- Panel (f): Parameter importance bar chart -----------------------
    ax_bar = axes[5]

    # Compute mAP swing (max - min) for each parameter
    param_short_labels = []
    swings = []
    for pn in sweep_keys:
        sweep = results["sweeps"][pn]
        mAP_range = max(sweep["mAP_05"]) - min(sweep["mAP_05"])
        swings.append(mAP_range * 100)  # convert to percentage points
        # Short label for bar chart
        label_map = {
            "boost_single": r"$\beta_{\rm s}$",
            "boost_dual": r"$\beta_{\rm d}$",
            "suppress_factor": r"$\gamma$",
            "suppress_conf_gate": r"$\theta_{\rm low}$",
            "match_iou_thresh": r"$\tau_{\rm IoU}$",
        }
        param_short_labels.append(label_map.get(pn, pn))

    # Sort by swing descending
    order = sorted(range(len(swings)), key=lambda k: swings[k], reverse=True)
    sorted_labels = [param_short_labels[k] for k in order]
    sorted_swings = [swings[k] for k in order]

    # Color: dominant parameter in accent color, others in blue
    colors = ["#d32f2f" if s == max(sorted_swings) else "#2166ac" for s in sorted_swings]

    bars = ax_bar.barh(
        range(len(sorted_labels)),
        sorted_swings,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        height=0.55,
        zorder=3,
    )

    ax_bar.set_yticks(range(len(sorted_labels)))
    ax_bar.set_yticklabels(sorted_labels)
    ax_bar.set_xlabel("mAP swing (pp)")
    ax_bar.invert_yaxis()

    # Add value labels on bars
    for bar_obj, val in zip(bars, sorted_swings):
        ax_bar.text(
            bar_obj.get_width() + 0.02,
            bar_obj.get_y() + bar_obj.get_height() / 2,
            f"{val:.2f}",
            va="center",
            ha="left",
            fontsize=7.5,
        )

    ax_bar.text(
        0.04,
        0.96,
        "(f)",
        transform=ax_bar.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )

    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.tick_params(direction="out")
    ax_bar.set_xlim(0, max(sorted_swings) * 1.35)

    fig.tight_layout(pad=0.8, h_pad=1.2, w_pad=1.0)

    # Save
    pdf_path = output_dir / "fig_sensitivity.pdf"
    png_path = output_dir / "fig_sensitivity.png"
    fig.savefig(str(pdf_path), format="pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(str(png_path), format="png", bbox_inches="tight", pad_inches=0.05, dpi=300)
    plt.close(fig)

    print(f"\nFigure saved to:")
    print(f"  {pdf_path}")
    print(f"  {png_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run the sweep
    results = run_sweep()

    # Save JSON
    json_path = OUTPUT_DIR / "sensitivity_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # Generate figure
    generate_figure(results, OUTPUT_DIR)

    # Print summary table
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("=" * 70)
    baseline = results["baseline"]["mAP_05"]
    print(f"LiDAR-only baseline mAP@0.5 = {baseline:.6f}\n")

    print(f"{'Parameter':<22} {'Default':>8} {'mAP@default':>12} {'Best val':>10} {'Best mAP':>12} {'Range':>12}")
    print("-" * 80)

    for pn in SWEEP_RANGES:
        sweep = results["sweeps"][pn]
        dv = sweep["default_value"]
        vals = sweep["values"]
        maps = sweep["mAP_05"]

        # mAP at default
        default_mAP = None
        for j, v in enumerate(vals):
            if abs(v - dv) < 1e-9:
                default_mAP = maps[j]
                break

        best_idx = int(np.argmax(maps))
        best_val = vals[best_idx]
        best_mAP = maps[best_idx]
        mAP_range = max(maps) - min(maps)

        print(
            f"{pn:<22} {dv:>8.2f} {default_mAP:>12.6f} {best_val:>10.2f} {best_mAP:>12.6f} {mAP_range:>12.6f}"
        )

    print()


if __name__ == "__main__":
    main()
