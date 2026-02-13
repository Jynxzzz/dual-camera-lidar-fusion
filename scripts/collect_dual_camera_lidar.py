"""
Collect Dual-Camera + LiDAR Dataset from CARLA

Sensors:
1. SDC Camera (ego vehicle front-facing camera)
2. Drone Camera (bird's eye view camera, fixed position above scene)
3. LiDAR (on ego vehicle roof)

All sensors synchronized at frame level.
Spawns background traffic (vehicles + pedestrians) for realistic scenes.
"""

import argparse
import json
import os
import sys
import glob
import time
import math
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict

# Add CARLA Python API
sys.path.append("/home/xingnan/Carla/PythonAPI/carla/dist")
try:
    import glob as _glob
    carla_eggs = _glob.glob("/home/xingnan/Carla/PythonAPI/carla/dist/carla-*-py3.*-linux_x86_64.egg")
    if carla_eggs:
        sys.path.insert(0, carla_eggs[0])
except Exception:
    pass
sys.path.append("/home/xingnan/Carla/PythonAPI/carla")
sys.path.append("/home/xingnan/Carla/PythonAPI")

import carla

# Configuration
OUTPUT_DIR = Path("/mnt/hdd12t/outputs/carla_out/dual_camera_lidar_dataset")
TARGET_FRAMES = 650  # Collect 650 frames
NUM_VEHICLES = 50    # Background traffic vehicles
NUM_WALKERS = 30     # Background pedestrians

# SDC Camera (ego vehicle front camera)
SDC_CAMERA_CONFIG = {
    "width": 1920,
    "height": 1280,
    "fov": 90.0,
    "x": 1.5,        # 1.5m forward from vehicle center
    "y": 0.0,
    "z": 2.4,        # 2.4m above ground (dashcam height)
    "pitch": 0.0,
    "yaw": 0.0,
    "roll": 0.0,
}

# Drone Camera (bird's eye view, follows vehicle from above)
DRONE_CAMERA_CONFIG = {
    "width": 1920,
    "height": 1280,
    "fov": 90.0,
    "x": 0.0,        # Directly above vehicle
    "y": 0.0,
    "z": 40.0,       # 40m above vehicle
    "pitch": -90.0,  # Looking straight down
    "yaw": 0.0,
    "roll": 0.0,
}

# LiDAR
LIDAR_CONFIG = {
    "range": 70.0,
    "channels": 64,
    "points_per_second": 2000000,
    "rotation_frequency": 10.0,
    "upper_fov": 10.0,
    "lower_fov": -30.0,
    "x": 0.0,
    "y": 0.0,
    "z": 2.5,
    "pitch": 0.0,
    "yaw": 0.0,
    "roll": 0.0,
}

# Global buffers for synchronized data
sdc_camera_buffer = {"data": None, "frame": None}
drone_camera_buffer = {"data": None, "frame": None}
lidar_buffer = {"data": None, "frame": None}

frame_counter = 0


def setup_output_dirs():
    """Create output directory structure."""
    dirs = [
        OUTPUT_DIR / "points",
        OUTPUT_DIR / "images_sdc",      # SDC camera images
        OUTPUT_DIR / "images_drone",    # Drone camera images
        OUTPUT_DIR / "labels",
        OUTPUT_DIR / "calib",
        OUTPUT_DIR / "pose",
        OUTPUT_DIR / "ImageSets",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    print(f"✅ Output directories created at: {OUTPUT_DIR}")


def camera_callback(image, camera_type):
    """Callback for camera data."""
    global sdc_camera_buffer, drone_camera_buffer

    # Convert image to numpy array
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))  # BGRA
    array = array[:, :, :3]  # Drop alpha channel
    array = array[:, :, ::-1]  # BGR to RGB

    if camera_type == "sdc":
        sdc_camera_buffer["data"] = array
        sdc_camera_buffer["frame"] = image.frame
    elif camera_type == "drone":
        drone_camera_buffer["data"] = array
        drone_camera_buffer["frame"] = image.frame


def lidar_callback(point_cloud):
    """Callback for LiDAR data."""
    global lidar_buffer

    # Convert to numpy array
    points = np.frombuffer(point_cloud.raw_data, dtype=np.float32)
    points = points.reshape(-1, 4)  # (x, y, z, intensity)

    lidar_buffer["data"] = points
    lidar_buffer["frame"] = point_cloud.frame


def calculate_camera_intrinsics(width, height, fov):
    """Calculate camera intrinsic matrix from FOV."""
    focal = width / (2.0 * np.tan(fov * np.pi / 360.0))
    cx = width / 2.0
    cy = height / 2.0

    K = np.array([
        [focal, 0.0, cx],
        [0.0, focal, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    return K


def get_transform_matrix(sensor_config, is_drone=False):
    """
    Calculate transformation matrix from sensor to LiDAR frame.

    For drone camera: since it's attached to vehicle, we still use relative transform
    """
    # Camera to LiDAR transform (in CARLA coordinates)
    dx = LIDAR_CONFIG["x"] - sensor_config["x"]
    dy = LIDAR_CONFIG["y"] - sensor_config["y"]
    dz = LIDAR_CONFIG["z"] - sensor_config["z"]

    # For drone camera, the transform represents drone_cam -> lidar
    # Since drone is at z=40m above vehicle, this gives correct relative position
    T_cam_to_lidar = np.eye(4, dtype=np.float32)
    T_cam_to_lidar[0, 3] = dx
    T_cam_to_lidar[1, 3] = dy
    T_cam_to_lidar[2, 3] = dz

    return T_cam_to_lidar


def save_calibration(frame_id):
    """Save calibration data for both cameras."""
    calib_file = OUTPUT_DIR / "calib" / f"{frame_id:06d}.txt"

    # SDC Camera intrinsics
    K_sdc = calculate_camera_intrinsics(
        SDC_CAMERA_CONFIG["width"],
        SDC_CAMERA_CONFIG["height"],
        SDC_CAMERA_CONFIG["fov"]
    )

    # Drone Camera intrinsics
    K_drone = calculate_camera_intrinsics(
        DRONE_CAMERA_CONFIG["width"],
        DRONE_CAMERA_CONFIG["height"],
        DRONE_CAMERA_CONFIG["fov"]
    )

    # Extrinsics
    T_sdc_to_lidar = get_transform_matrix(SDC_CAMERA_CONFIG, is_drone=False)
    T_drone_to_lidar = get_transform_matrix(DRONE_CAMERA_CONFIG, is_drone=True)

    with open(calib_file, 'w') as f:
        # SDC Camera
        f.write(f"K_sdc: {' '.join(map(str, K_sdc.flatten()))}\n")
        f.write(f"T_sdc_to_lidar: {' '.join(map(str, T_sdc_to_lidar[:3, :].flatten()))}\n")

        # Drone Camera
        f.write(f"K_drone: {' '.join(map(str, K_drone.flatten()))}\n")
        f.write(f"T_drone_to_lidar: {' '.join(map(str, T_drone_to_lidar[:3, :].flatten()))}\n")


def transform_to_matrix(transform):
    """Convert carla.Transform to 4x4 numpy matrix."""
    loc = transform.location
    rot = transform.rotation
    pitch = np.radians(rot.pitch)
    yaw = np.radians(rot.yaw)
    roll = np.radians(rot.roll)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [loc.x, loc.y, loc.z]
    return T


# Minimum bounding box sizes (KITTI conventions)
MIN_BBOX_SIZE = {
    "Car": (3.5, 1.5, 1.4),
    "Pedestrian": (0.8, 0.6, 1.7),
    "Cyclist": (1.8, 0.6, 1.7),
}


def collect_ground_truth(world, vehicle, lidar_sensor, frame_id):
    """
    Collect ground truth bounding boxes using proper matrix transforms.
    Uses lidar sensor's actual world transform for accurate coordinates.

    Format: x y z dx dy dz heading_angle class_name
    (OpenPCDet custom dataset format, in LiDAR coordinate frame)
    """
    # Get lidar world transform and its inverse
    lidar_transform = lidar_sensor.get_transform()
    lidar_world = transform_to_matrix(lidar_transform)
    lidar_world_inv = np.linalg.inv(lidar_world)

    ego_id = vehicle.id
    labels = []

    # Get all vehicles and pedestrians
    actors = world.get_actors()
    vehicles_list = actors.filter("vehicle.*")
    walkers_list = actors.filter("walker.*")

    for actor in list(vehicles_list) + list(walkers_list):
        if actor.id == ego_id:
            continue

        # Classify actor
        type_id = actor.type_id
        if "walker" in type_id or "pedestrian" in type_id:
            class_name = "Pedestrian"
        elif "vehicle.bicycle" in type_id or "vehicle.motorcycle" in type_id:
            class_name = "Cyclist"
        elif "vehicle" in type_id:
            class_name = "Car"
        else:
            continue

        # Bounding box (CARLA gives half-extents)
        bbox = actor.bounding_box
        extent = bbox.extent
        l = extent.x * 2
        w = extent.y * 2
        h = extent.z * 2

        # Enforce minimum bbox sizes (handles -inf and tiny boxes)
        if class_name in MIN_BBOX_SIZE:
            min_l, min_w, min_h = MIN_BBOX_SIZE[class_name]
            if not np.isfinite(l) or l < min_l:
                l = min_l
            if not np.isfinite(w) or w < min_w:
                w = min_w
            if not np.isfinite(h) or h < min_h:
                h = min_h

        # Actor transform
        actor_transform = actor.get_transform()

        # Bbox center in world frame (includes bbox.location offset)
        bbox_center_world = np.array([
            actor_transform.location.x + bbox.location.x,
            actor_transform.location.y + bbox.location.y,
            actor_transform.location.z + bbox.location.z,
            1.0,
        ])

        # Transform to LiDAR frame
        bbox_center_lidar = lidar_world_inv @ bbox_center_world
        x = float(bbox_center_lidar[0])
        y = float(bbox_center_lidar[1])
        z = float(bbox_center_lidar[2])

        # Skip if any coordinate is invalid
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
            continue

        # Distance filter
        dist = math.sqrt(x**2 + y**2)
        if dist > 70.0:
            continue

        # Rotation: actor yaw relative to lidar frame, normalized to [-pi, pi]
        actor_yaw = np.radians(actor_transform.rotation.yaw)
        lidar_yaw = np.radians(lidar_transform.rotation.yaw)
        heading = actor_yaw - lidar_yaw
        heading = (heading + np.pi) % (2 * np.pi) - np.pi

        # Format: x y z dx dy dz heading_angle class_name
        labels.append(f"{x:.2f} {y:.2f} {z:.2f} {l:.2f} {w:.2f} {h:.2f} {heading:.4f} {class_name}")

    # Always create label file (even if empty)
    label_file = OUTPUT_DIR / "labels" / f"{frame_id:06d}.txt"
    with open(label_file, 'w') as f:
        f.write('\n'.join(labels))

    return len(labels)


def save_frame(frame_id, vehicle):
    """Save synchronized frame data."""
    # Save SDC camera image
    img_sdc = cv2.cvtColor(sdc_camera_buffer["data"], cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(OUTPUT_DIR / "images_sdc" / f"{frame_id:06d}.jpg"), img_sdc)

    # Save Drone camera image
    img_drone = cv2.cvtColor(drone_camera_buffer["data"], cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(OUTPUT_DIR / "images_drone" / f"{frame_id:06d}.jpg"), img_drone)

    # Save LiDAR points
    np.save(str(OUTPUT_DIR / "points" / f"{frame_id:06d}.npy"), lidar_buffer["data"])

    # Save calibration
    save_calibration(frame_id)

    # Save ego pose as 4x4 matrix
    ego_transform = vehicle.get_transform()
    ego_matrix = transform_to_matrix(ego_transform)
    np.save(str(OUTPUT_DIR / "pose" / f"{frame_id:06d}.npy"), ego_matrix)


def spawn_traffic(client, world, traffic_manager, num_vehicles=50, num_walkers=30):
    """Spawn background traffic with proper walker AI setup."""
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    vehicles = []
    vehicle_bps = bp_lib.filter("vehicle.*")

    # Spawn vehicles
    np.random.shuffle(spawn_points)
    for i, sp in enumerate(spawn_points[:num_vehicles]):
        bp = np.random.choice(vehicle_bps)
        if bp.has_attribute("color"):
            color = np.random.choice(bp.get_attribute("color").recommended_values)
            bp.set_attribute("color", color)
        bp.set_attribute("role_name", "autopilot")

        v = world.try_spawn_actor(bp, sp)
        if v is not None:
            v.set_autopilot(True, traffic_manager.get_port())
            vehicles.append(v)

    print(f"  Spawned {len(vehicles)} background vehicles")

    # Spawn walkers with diverse locations
    walker_bps = bp_lib.filter("walker.pedestrian.*")
    walkers = []
    controllers = []
    spawn_locs = set()

    attempts = num_walkers * 5
    for _ in range(attempts):
        if len(walkers) >= num_walkers:
            break
        spawn_point = carla.Transform()
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        loc_key = (round(loc.x, 1), round(loc.y, 1))
        if loc_key in spawn_locs:
            continue
        spawn_locs.add(loc_key)
        spawn_point.location = loc
        bp = np.random.choice(walker_bps)
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")
        walker = world.try_spawn_actor(bp, spawn_point)
        if walker is not None:
            walkers.append(walker)

    # Tick before starting walker controllers
    world.tick()

    # Spawn walker AI controllers
    controller_bp = bp_lib.find("controller.ai.walker")
    for walker in walkers:
        controller = world.spawn_actor(controller_bp, carla.Transform(), walker)
        controllers.append(controller)

    world.tick()

    # Start controllers and assign destinations
    for controller in controllers:
        controller.start()
    world.tick()

    for controller in controllers:
        dest = world.get_random_location_from_navigation()
        if dest is not None:
            controller.go_to_location(dest)
        controller.set_max_speed(1.0 + np.random.random() * 1.5)

    # Verify walker diversity
    unique_positions = set()
    for w in walkers:
        loc = w.get_location()
        unique_positions.add((round(loc.x, 0), round(loc.y, 0)))

    print(f"  Spawned {len(walkers)} pedestrians at {len(unique_positions)} unique positions")
    return vehicles, walkers, controllers


def main():
    global frame_counter

    parser = argparse.ArgumentParser(description="Dual-Camera + LiDAR CARLA Data Collection")
    parser.add_argument("--frames", type=int, default=TARGET_FRAMES, help="Number of frames")
    parser.add_argument("--vehicles", type=int, default=NUM_VEHICLES, help="Background vehicles")
    parser.add_argument("--walkers", type=int, default=NUM_WALKERS, help="Background pedestrians")
    parser.add_argument("--output", type=str, default=None, help="Output directory override")
    parser.add_argument("--fps", type=float, default=20.0, help="Simulation tick rate")
    args = parser.parse_args()

    global OUTPUT_DIR
    if args.output:
        OUTPUT_DIR = Path(args.output)

    setup_output_dirs()

    # Connect to CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(60.0)
    world = client.get_world()

    print(f"✅ Connected to CARLA server")
    print(f"🗺️  Map: {world.get_map().name}")
    print(f"📊 Config: {args.frames} frames, {args.vehicles} vehicles, {args.walkers} walkers")

    # Set synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / args.fps
    world.apply_settings(settings)

    # Setup Traffic Manager (MUST be before any autopilot)
    print("\nSetting up Traffic Manager...")
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_random_device_seed(42)

    blueprint_library = world.get_blueprint_library()

    # Spawn ego vehicle
    print("Spawning ego vehicle...")
    spawn_points = world.get_map().get_spawn_points()
    ego_bp = blueprint_library.find('vehicle.tesla.model3')
    ego_bp.set_attribute("role_name", "ego")
    vehicle = None
    for sp_idx, sp in enumerate(spawn_points[:10]):
        vehicle = world.try_spawn_actor(ego_bp, sp)
        if vehicle is not None:
            print(f"  Ego vehicle spawned at spawn_point[{sp_idx}]: {sp.location}")
            break
    if vehicle is None:
        print("ERROR: Failed to spawn ego vehicle!")
        return
    vehicle.set_autopilot(True, traffic_manager.get_port())

    # Track all spawned actors for cleanup
    bg_vehicles, bg_walkers, bg_controllers = [], [], []
    sensors = []

    try:
        # Create sensors
        # 1. SDC Camera
        sdc_cam_bp = blueprint_library.find('sensor.camera.rgb')
        sdc_cam_bp.set_attribute('image_size_x', str(SDC_CAMERA_CONFIG["width"]))
        sdc_cam_bp.set_attribute('image_size_y', str(SDC_CAMERA_CONFIG["height"]))
        sdc_cam_bp.set_attribute('fov', str(SDC_CAMERA_CONFIG["fov"]))

        sdc_cam_transform = carla.Transform(
            carla.Location(x=SDC_CAMERA_CONFIG["x"], y=SDC_CAMERA_CONFIG["y"], z=SDC_CAMERA_CONFIG["z"]),
            carla.Rotation(pitch=SDC_CAMERA_CONFIG["pitch"], yaw=SDC_CAMERA_CONFIG["yaw"], roll=SDC_CAMERA_CONFIG["roll"])
        )
        sdc_camera = world.spawn_actor(sdc_cam_bp, sdc_cam_transform, attach_to=vehicle)
        sdc_camera.listen(lambda image: camera_callback(image, "sdc"))
        sensors.append(sdc_camera)
        print(f"📸 SDC Camera spawned at (x={SDC_CAMERA_CONFIG['x']}, z={SDC_CAMERA_CONFIG['z']})")

        # 2. Drone Camera
        drone_cam_bp = blueprint_library.find('sensor.camera.rgb')
        drone_cam_bp.set_attribute('image_size_x', str(DRONE_CAMERA_CONFIG["width"]))
        drone_cam_bp.set_attribute('image_size_y', str(DRONE_CAMERA_CONFIG["height"]))
        drone_cam_bp.set_attribute('fov', str(DRONE_CAMERA_CONFIG["fov"]))

        drone_cam_transform = carla.Transform(
            carla.Location(x=DRONE_CAMERA_CONFIG["x"], y=DRONE_CAMERA_CONFIG["y"], z=DRONE_CAMERA_CONFIG["z"]),
            carla.Rotation(pitch=DRONE_CAMERA_CONFIG["pitch"], yaw=DRONE_CAMERA_CONFIG["yaw"], roll=DRONE_CAMERA_CONFIG["roll"])
        )
        drone_camera = world.spawn_actor(drone_cam_bp, drone_cam_transform, attach_to=vehicle)
        drone_camera.listen(lambda image: camera_callback(image, "drone"))
        sensors.append(drone_camera)
        print(f"🚁 Drone Camera spawned at (x={DRONE_CAMERA_CONFIG['x']}, z={DRONE_CAMERA_CONFIG['z']}, pitch={DRONE_CAMERA_CONFIG['pitch']}°)")

        # 3. LiDAR
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', str(LIDAR_CONFIG["range"]))
        lidar_bp.set_attribute('channels', str(LIDAR_CONFIG["channels"]))
        lidar_bp.set_attribute('points_per_second', str(int(LIDAR_CONFIG["points_per_second"])))
        lidar_bp.set_attribute('rotation_frequency', str(LIDAR_CONFIG["rotation_frequency"]))
        lidar_bp.set_attribute('upper_fov', str(LIDAR_CONFIG["upper_fov"]))
        lidar_bp.set_attribute('lower_fov', str(LIDAR_CONFIG["lower_fov"]))

        lidar_transform = carla.Transform(
            carla.Location(x=LIDAR_CONFIG["x"], y=LIDAR_CONFIG["y"], z=LIDAR_CONFIG["z"]),
            carla.Rotation(pitch=LIDAR_CONFIG["pitch"], yaw=LIDAR_CONFIG["yaw"], roll=LIDAR_CONFIG["roll"])
        )
        lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
        lidar.listen(lidar_callback)
        sensors.append(lidar)
        print(f"📡 LiDAR spawned at (x={LIDAR_CONFIG['x']}, z={LIDAR_CONFIG['z']})")

        # Spawn background traffic
        print("\nSpawning background traffic...")
        bg_vehicles, bg_walkers, bg_controllers = spawn_traffic(
            client, world, traffic_manager, args.vehicles, args.walkers
        )

        # Warmup - let traffic settle and sensors stabilize
        print("Warming up simulation (150 ticks)...")
        for i in range(150):
            world.tick()
            if i == 100:
                for ctrl in bg_controllers:
                    dest = world.get_random_location_from_navigation()
                    if dest is not None:
                        ctrl.go_to_location(dest)

        # Verify ego is moving
        ego_vel = vehicle.get_velocity()
        ego_speed = math.sqrt(ego_vel.x**2 + ego_vel.y**2 + ego_vel.z**2) * 3.6
        ego_loc = vehicle.get_location()
        print(f"  Ego speed: {ego_speed:.1f} km/h, location: ({ego_loc.x:.1f}, {ego_loc.y:.1f})")
        if ego_speed < 1.0:
            print("  WARNING: Ego appears stuck, trying different spawn point...")
            vehicle.set_transform(spawn_points[1])
            for _ in range(50):
                world.tick()

        target_frames = args.frames
        print(f"\n🎬 Starting data collection... (Target: {target_frames} frames)")
        print(f"=" * 70)

        # Stats tracking
        stats = {"total_annotations": 0, "class_counts": {}, "frames_with_zero_labels": 0}

        # Collection loop
        tick_count = 0
        while frame_counter < target_frames:
            world.tick()
            tick_count += 1

            # Refresh walker destinations periodically
            if tick_count % 500 == 0:
                for ctrl in bg_controllers:
                    dest = world.get_random_location_from_navigation()
                    if dest is not None:
                        ctrl.go_to_location(dest)

            # Check if all sensors have data for the same frame
            if (sdc_camera_buffer["data"] is not None and
                drone_camera_buffer["data"] is not None and
                lidar_buffer["data"] is not None):

                # Verify frame synchronization
                sdc_frame = sdc_camera_buffer["frame"]
                drone_frame = drone_camera_buffer["frame"]
                lidar_frame = lidar_buffer["frame"]

                if sdc_frame == drone_frame == lidar_frame:
                    # Collect ground truth
                    num_labels = collect_ground_truth(world, vehicle, lidar, frame_counter)

                    # Track stats
                    if num_labels == 0:
                        stats["frames_with_zero_labels"] += 1

                    # Save frame
                    save_frame(frame_counter, vehicle)

                    frame_counter += 1

                    if frame_counter % 10 == 0:
                        ego_loc = vehicle.get_location()
                        ego_vel = vehicle.get_velocity()
                        ego_speed = math.sqrt(ego_vel.x**2 + ego_vel.y**2) * 3.6
                        print(f"✅ Frame {frame_counter}/{target_frames} | "
                              f"Labels: {num_labels} | "
                              f"LiDAR: {len(lidar_buffer['data'])} pts | "
                              f"Ego: ({ego_loc.x:.0f},{ego_loc.y:.0f}) {ego_speed:.0f}km/h")

                    # Clear buffers
                    sdc_camera_buffer["data"] = None
                    drone_camera_buffer["data"] = None
                    lidar_buffer["data"] = None
                else:
                    if frame_counter < 5:  # Only warn early on
                        print(f"⚠️  Frame mismatch: SDC={sdc_frame}, Drone={drone_frame}, LiDAR={lidar_frame}")

            time.sleep(0.01)

        print(f"\n{'=' * 70}")
        print(f"✅ Data collection complete! Collected {frame_counter} frames")
        print(f"📁 Output directory: {OUTPUT_DIR}")
        print(f"📊 Zero-label frames: {stats['frames_with_zero_labels']}/{frame_counter}")

        # Generate ImageSets (train/val/test splits)
        print("\nGenerating train/val/test splits...")
        np.random.seed(42)
        frame_ids = [f"{i:06d}" for i in range(frame_counter)]
        indices = np.random.permutation(len(frame_ids))
        n_train = int(len(indices) * 0.7)
        n_val = int(len(indices) * 0.15)

        train_ids = sorted([frame_ids[i] for i in indices[:n_train]])
        val_ids = sorted([frame_ids[i] for i in indices[n_train:n_train + n_val]])
        test_ids = sorted([frame_ids[i] for i in indices[n_train + n_val:]])

        for split_name, split_ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
            with open(OUTPUT_DIR / "ImageSets" / f"{split_name}.txt", "w") as f:
                for fid in split_ids:
                    f.write(f"{fid}\n")
            print(f"  {split_name}: {len(split_ids)} frames")

        # Save metadata
        metadata = {
            "map": world.get_map().name,
            "total_frames": frame_counter,
            "sensors": {
                "sdc_camera": SDC_CAMERA_CONFIG,
                "drone_camera": DRONE_CAMERA_CONFIG,
                "lidar": LIDAR_CONFIG,
            },
            "traffic": {"vehicles": args.vehicles, "walkers": args.walkers},
            "zero_label_frames": stats["frames_with_zero_labels"],
        }
        with open(OUTPUT_DIR / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"\n📋 Metadata saved to {OUTPUT_DIR / 'metadata.json'}")

    finally:
        # Cleanup ALL actors
        print(f"\n🧹 Cleaning up...")
        for sensor in sensors:
            try:
                sensor.stop()
                sensor.destroy()
            except Exception:
                pass

        for ctrl in bg_controllers:
            try:
                ctrl.stop()
                ctrl.destroy()
            except Exception:
                pass
        for w in bg_walkers:
            try:
                w.destroy()
            except Exception:
                pass
        for v in bg_vehicles:
            try:
                v.destroy()
            except Exception:
                pass

        try:
            vehicle.destroy()
        except Exception:
            pass

        # Restore async mode
        try:
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
        except Exception:
            pass

        print(f"✅ All actors destroyed")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Interrupted by user")
        print(f"📊 Collected {frame_counter} frames before interruption")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
