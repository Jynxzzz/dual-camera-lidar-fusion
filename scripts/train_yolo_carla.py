"""
Fine-tune YOLOv8m on CARLA synthetic images for both SDC and drone cameras.

Trains domain-adapted YOLO models using projected 3D GT as 2D training labels.
This eliminates the domain gap between pretrained (real-world) YOLO and CARLA renders.

Usage:
    # Train SDC model
    python train_yolo_carla.py --camera sdc --epochs 50 --batch 16

    # Train Drone model
    python train_yolo_carla.py --camera drone --epochs 50 --batch 16

    # Train both sequentially
    python train_yolo_carla.py --camera both --epochs 50 --batch 16

Author: Xingnan Zhou
Date: 2026-02-10
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def train_camera(camera_name, data_yaml, output_dir, epochs=50, batch=16, imgsz=1280, device=0):
    """Train YOLOv8m for a specific camera view."""
    print(f"\n{'='*60}")
    print(f"Training YOLOv8m for {camera_name.upper()} camera")
    print(f"{'='*60}")
    print(f"  Data: {data_yaml}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch}")
    print(f"  Image size: {imgsz}")
    print(f"  Output: {output_dir}")

    # Load pretrained YOLOv8m
    model = YOLO('yolov8m.pt')

    # Train
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project=str(output_dir),
        name=f'yolov8m_{camera_name}',
        patience=15,
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
        workers=4,
        # Augmentation suited for synthetic data
        hsv_h=0.01,     # Less color augmentation (synthetic colors are consistent)
        hsv_s=0.3,
        hsv_v=0.2,
        degrees=0.0,    # No rotation (camera is fixed)
        translate=0.1,
        scale=0.3,
        flipud=0.0,     # No vertical flip (gravity matters)
        fliplr=0.5,     # Horizontal flip is fine
        mosaic=0.5,     # Moderate mosaic
        mixup=0.1,      # Light mixup
    )

    # Validate
    best_model = Path(output_dir) / f'yolov8m_{camera_name}' / 'weights' / 'best.pt'
    print(f"\nBest model saved to: {best_model}")

    return best_model


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8m on CARLA data")
    parser.add_argument('--camera', type=str, default='both',
                        choices=['sdc', 'drone', 'both'],
                        help='Which camera to train for')
    parser.add_argument('--data-dir', type=str,
                        default='/mnt/hdd12t/outputs/carla_out/yolo_carla',
                        help='YOLO dataset directory')
    parser.add_argument('--output', type=str,
                        default='/mnt/hdd12t/outputs/carla_out/yolo_training',
                        help='Training output directory')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=1280,
                        help='Training image size')
    parser.add_argument('--device', type=str, default='0',
                        help='CUDA device')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    cameras = []
    if args.camera in ('sdc', 'both'):
        cameras.append('sdc')
    if args.camera in ('drone', 'both'):
        cameras.append('drone')

    best_models = {}
    for cam in cameras:
        data_yaml = data_dir / cam / 'data.yaml'
        if not data_yaml.exists():
            print(f"ERROR: {data_yaml} not found. Run generate_yolo_labels.py first.")
            continue

        best = train_camera(
            cam, data_yaml, output_dir,
            epochs=args.epochs, batch=args.batch,
            imgsz=args.imgsz, device=args.device
        )
        best_models[cam] = best

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    for cam, path in best_models.items():
        print(f"  {cam.upper()}: {path}")


if __name__ == '__main__':
    main()
