"""
nuScenes data loader for the BEV Attribution Debug Tool.

Loads multi-camera samples from the nuScenes mini split, returning images,
calibration matrices, and ground-truth annotations in a format ready for
BEV model inference and Captum attribution.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# nuScenes surround-camera names — matches nuscenes-devkit create_nuscenes_infos.py
# order used for TPVFormer / GaussianFormer training.
CAMERA_NAMES = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

# Default image normalisation (ImageNet statistics).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Default target resolution for the model pipeline.
DEFAULT_IMG_H = 256
DEFAULT_IMG_W = 704


def _find_dataroot(dataroot: str) -> str:
    """Resolve the nuScenes data root, checking common layouts.

    The nuScenes mini split can be extracted in several ways:
      1. <dataroot>/v1.0-mini/  (metadata) + <dataroot>/samples/ + <dataroot>/sweeps/
      2. <dataroot>/  directly containing v1.0-mini/, samples/, sweeps/

    Returns the path that should be passed to NuScenes(dataroot=...).

    Raises FileNotFoundError if no valid layout is found.
    """
    dataroot = os.path.abspath(dataroot)

    # Check if dataroot itself contains the expected sub-dirs.
    if os.path.isdir(os.path.join(dataroot, "v1.0-mini")):
        return dataroot

    # Maybe the user pointed directly at the version directory.
    parent = os.path.dirname(dataroot)
    if os.path.isdir(os.path.join(parent, "v1.0-mini")):
        return parent

    raise FileNotFoundError(
        f"Could not find nuScenes v1.0-mini metadata under '{dataroot}'. "
        "Expected a 'v1.0-mini/' subdirectory.  Run scripts/download_nuscenes.sh "
        "first, then ensure the archive is extracted so that "
        "<dataroot>/v1.0-mini/ exists."
    )


def _get_nuscenes(dataroot: str, version: str = "v1.0-mini"):
    """Lazy-load the NuScenes database object (cached per dataroot)."""
    # Import here so the rest of the module works even without nuscenes.
    from nuscenes.nuscenes import NuScenes

    resolved = _find_dataroot(dataroot)
    logger.info("Loading NuScenes %s from %s …", version, resolved)
    nusc = NuScenes(version=version, dataroot=resolved, verbose=False)
    return nusc


def _load_image(filepath: str) -> Image.Image:
    """Load a single image from disk as RGB PIL Image."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Image file not found: {filepath}")
    return Image.open(filepath).convert("RGB")


def _image_to_tensor(
    img: Image.Image,
    target_h: int = DEFAULT_IMG_H,
    target_w: int = DEFAULT_IMG_W,
) -> torch.Tensor:
    """Resize, normalise, and convert a PIL image to a [3, H, W] float tensor."""
    img_resized = img.resize((target_w, target_h), Image.BILINEAR)
    arr = np.asarray(img_resized, dtype=np.float32) / 255.0  # [H, W, 3]
    # Normalise with ImageNet statistics.
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
    arr = (arr - mean) / std
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]
    return tensor


def _parse_calibration(nusc, cam_data: dict):
    """Extract calibration for a camera.

    nuScenes stores calibration as:
      - calibrated_sensor: sensor-to-ego (translation + rotation)
      - ego_pose: ego-to-world (translation + rotation)

    Returns:
      intrinsic:       3x3 camera intrinsic matrix
      world_to_camera: 4x4 = inv(sensor_to_ego) @ inv(ego_to_world)
      ego_to_camera:   4x4 = inv(sensor_to_ego)  — for projecting BEV points
      cam_to_global:   4x4 = ego_to_world @ sensor_to_ego  — for lidar2img via global frame
    """
    from pyquaternion import Quaternion

    # sensor → ego
    cs = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
    intrinsic = np.array(cs["camera_intrinsic"], dtype=np.float64)  # 3x3

    sensor_to_ego = np.eye(4, dtype=np.float64)
    sensor_to_ego[:3, :3] = Quaternion(cs["rotation"]).rotation_matrix
    sensor_to_ego[:3, 3] = np.array(cs["translation"])

    # ego → camera  (for BEV projection — BEV points are in ego frame)
    ego_to_camera = np.linalg.inv(sensor_to_ego)

    # ego → world (camera's ego pose at camera timestamp)
    ep = nusc.get("ego_pose", cam_data["ego_pose_token"])
    ego_to_world = np.eye(4, dtype=np.float64)
    ego_to_world[:3, :3] = Quaternion(ep["rotation"]).rotation_matrix
    ego_to_world[:3, 3] = np.array(ep["translation"])

    # world → camera  =  inv(sensor_to_ego) @ inv(ego_to_world)
    world_to_camera = ego_to_camera @ np.linalg.inv(ego_to_world)

    # cam_to_global = ego_to_world @ sensor_to_ego (full chain, camera-timestamp ego)
    cam_to_global = ego_to_world @ sensor_to_ego

    return intrinsic, world_to_camera, ego_to_camera, cam_to_global


def _parse_annotations(nusc, sample_token: str) -> List[Dict]:
    """Parse ground-truth 3D bounding boxes for a sample.

    Returns a list of dicts, each with:
      - center: [x, y, z] in global frame
      - size: [w, l, h] (width, length, height)
      - rotation: [w, x, y, z] quaternion
      - class: str (e.g. 'car', 'pedestrian')
    """
    sample = nusc.get("sample", sample_token)
    gt_boxes = []
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        gt_boxes.append(
            {
                "center": np.array(ann["translation"], dtype=np.float64),
                "size": np.array(ann["size"], dtype=np.float64),
                "rotation": np.array(ann["rotation"], dtype=np.float64),
                "class": ann["category_name"],
            }
        )
    return gt_boxes


def load_sample(
    scene_idx: int = 0,
    sample_idx: int = 0,
    dataroot: str = "data/",
    img_h: int = DEFAULT_IMG_H,
    img_w: int = DEFAULT_IMG_W,
) -> Dict:
    """Load a single nuScenes sample with 6 surround-camera images.

    Parameters
    ----------
    scene_idx : int
        Index into the list of scenes (0-9 for mini split).
    sample_idx : int
        Index of the keyframe sample within the chosen scene.
    dataroot : str
        Path to the nuScenes data root (containing v1.0-mini/).
    img_h, img_w : int
        Target image resolution after resize.

    Returns
    -------
    dict with keys:
        images          : list of 6 PIL.Image.Image (original resolution)
        image_tensors   : torch.Tensor [6, 3, img_h, img_w] normalised
        intrinsics      : list of 6 np.ndarray (3x3)
        extrinsics      : list of 6 np.ndarray (4x4), world-to-camera
        camera_names    : list of 6 str
        gt_boxes        : list of annotation dicts
        sample_token    : str
        scene_name      : str
        original_sizes  : list of 6 (W, H) tuples (original image sizes)

    Raises
    ------
    IndexError  – if scene_idx or sample_idx is out of range.
    FileNotFoundError – if data directory or images are missing.
    """
    nusc = _get_nuscenes(dataroot)

    # --- Resolve scene and sample ---
    if scene_idx < 0 or scene_idx >= len(nusc.scene):
        raise IndexError(
            f"scene_idx={scene_idx} out of range. "
            f"Available: 0..{len(nusc.scene) - 1}"
        )
    scene = nusc.scene[scene_idx]

    # Walk through samples in the scene to reach sample_idx.
    sample_token = scene["first_sample_token"]
    for _ in range(sample_idx):
        sample_record = nusc.get("sample", sample_token)
        if sample_record["next"] == "":
            raise IndexError(
                f"sample_idx={sample_idx} out of range for scene "
                f"'{scene['name']}' (only {_ + 1} samples available)."
            )
        sample_token = sample_record["next"]

    sample_record = nusc.get("sample", sample_token)

    # --- Extract lidar-to-ego transform (needed for TPVFormer/3D occ models) ---
    lidar_token = sample_record["data"]["LIDAR_TOP"]
    lidar_data = nusc.get("sample_data", lidar_token)
    lidar_cs = nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
    from pyquaternion import Quaternion as Q_lidar
    lidar_to_ego = np.eye(4, dtype=np.float64)
    lidar_to_ego[:3, :3] = Q_lidar(lidar_cs["rotation"]).rotation_matrix
    lidar_to_ego[:3, 3] = np.array(lidar_cs["translation"])

    # Lidar ego pose (at lidar timestamp) → lidar_to_global
    lidar_ep = nusc.get("ego_pose", lidar_data["ego_pose_token"])
    ego_to_global_lidar = np.eye(4, dtype=np.float64)
    ego_to_global_lidar[:3, :3] = Q_lidar(lidar_ep["rotation"]).rotation_matrix
    ego_to_global_lidar[:3, 3] = np.array(lidar_ep["translation"])
    lidar_to_global = ego_to_global_lidar @ lidar_to_ego

    # --- Load camera data ---
    images: List[Image.Image] = []
    image_tensors: List[torch.Tensor] = []
    intrinsics: List[np.ndarray] = []
    extrinsics: List[np.ndarray] = []
    ego_to_cameras: List[np.ndarray] = []
    cam_to_globals: List[np.ndarray] = []
    original_sizes: List[Tuple[int, int]] = []

    for cam_name in CAMERA_NAMES:
        cam_token = sample_record["data"][cam_name]
        cam_data = nusc.get("sample_data", cam_token)

        # Image
        img_path = os.path.join(nusc.dataroot, cam_data["filename"])
        img = _load_image(img_path)
        images.append(img)
        original_sizes.append(img.size)  # (W, H)
        image_tensors.append(_image_to_tensor(img, img_h, img_w))

        # Calibration
        K, extrinsic, ego_to_cam, cam_to_global = _parse_calibration(nusc, cam_data)
        intrinsics.append(K)
        extrinsics.append(extrinsic)
        ego_to_cameras.append(ego_to_cam)
        cam_to_globals.append(cam_to_global)

    # Stack image tensors: [6, 3, H, W]
    stacked_tensors = torch.stack(image_tensors, dim=0)

    # --- Ground-truth annotations ---
    gt_boxes = _parse_annotations(nusc, sample_token)

    return {
        "images": images,
        "image_tensors": stacked_tensors,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "ego_to_cameras": ego_to_cameras,
        "cam_to_globals": cam_to_globals,
        "lidar_to_ego": lidar_to_ego,
        "lidar_to_global": lidar_to_global,
        "camera_names": list(CAMERA_NAMES),
        "gt_boxes": gt_boxes,
        "sample_token": sample_token,
        "scene_name": scene["name"],
        "original_sizes": original_sizes,
    }


def get_scene_info(dataroot: str = "data/") -> List[Dict]:
    """Return metadata about all available scenes.

    Useful for populating a scene selector in the UI.
    """
    nusc = _get_nuscenes(dataroot)
    info = []
    for i, scene in enumerate(nusc.scene):
        info.append(
            {
                "index": i,
                "name": scene["name"],
                "description": scene["description"],
                "num_samples": scene["nbr_samples"],
            }
        )
    return info


def get_num_samples_in_scene(scene_idx: int = 0, dataroot: str = "data/") -> int:
    """Count the number of keyframe samples in a given scene."""
    nusc = _get_nuscenes(dataroot)
    scene = nusc.scene[scene_idx]
    count = 0
    token = scene["first_sample_token"]
    while token:
        count += 1
        sample = nusc.get("sample", token)
        token = sample["next"] if sample["next"] != "" else None
    return count
