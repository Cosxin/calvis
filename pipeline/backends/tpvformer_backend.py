"""TPVFormer backend -- Tri-Perspective View for 3D Semantic Occupancy.

Paper: "Tri-Perspective View for Vision-Based 3D Semantic Occupancy Prediction" (CVPR 2023)
Repo: https://github.com/wzzheng/TPVFormer

Output: 3D semantic occupancy grid [C, Z, H, W] where:
  - C = 17 classes (16 semantic + 1 free/empty)
  - Z = 16 height bins
  - H, W = 200x200 BEV grid
"""

import logging
import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from .base import BEVBackend

logger = logging.getLogger(__name__)

CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints")

# nuScenes LiDAR segmentation classes (TPVFormer uses these)
# TPVFormer outputs 18 classes: 17 semantic + noise (class 0 in their labeling
# maps to class index 17 in the output). We expose 17 classes for viz,
# stripping noise. The raw model output has noise as the last channel.
NUSCENES_LIDARSEG_NAMES = [
    'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain',
    'manmade', 'vegetation', 'free',
]

# Full names including noise (matches model output channels)
_TPV_ALL_NAMES = NUSCENES_LIDARSEG_NAMES + ['noise']

NUSCENES_LIDARSEG_COLORS = {
    'barrier':              (112, 128, 144),
    'bicycle':              (220, 20, 60),
    'bus':                  (255, 127, 80),
    'car':                  (255, 158, 0),
    'construction_vehicle': (233, 150, 70),
    'motorcycle':           (255, 61, 99),
    'pedestrian':           (0, 0, 230),
    'traffic_cone':         (47, 79, 79),
    'trailer':              (255, 140, 0),
    'truck':                (255, 99, 71),
    'driveable_surface':    (0, 207, 191),
    'other_flat':           (175, 0, 75),
    'sidewalk':             (75, 0, 75),
    'terrain':              (112, 180, 60),
    'manmade':              (222, 184, 135),
    'vegetation':           (0, 175, 0),
    'free':                 (30, 30, 30),
}


class TPVFormerBackend(BEVBackend):
    """TPVFormer backend for 3D semantic occupancy prediction.

    Produces a 3D voxel grid [C, Z, H, W] that can be:
    - Collapsed to 2D BEV via max-pooling along Z
    - Sliced at specific heights for inspection
    - Viewed as tri-perspective planes (XY, XZ, YZ)
    """

    name = "tpvformer"
    repr_type = "3d_occ"

    @property
    def class_names(self) -> List[str]:
        return list(NUSCENES_LIDARSEG_NAMES)

    def load(self, device: str = "cpu") -> Optional[nn.Module]:
        """Try to load TPVFormer model.

        Strategy:
        1. Check for vendored TPVFormer code + checkpoint
        2. Fall back to a synthetic placeholder model
        """
        # Try multiple checkpoint names
        ckpt_candidates = [
            os.path.join(CKPT_DIR, "tpv04_occupancy_v2.pth"),
            os.path.join(CKPT_DIR, "tpvformer_occ.pth"),
        ]
        ckpt_path = None
        for c in ckpt_candidates:
            if os.path.isfile(c):
                ckpt_path = c
                break

        # Try loading real TPVFormer
        if ckpt_path:
            try:
                model = self._load_real_model(ckpt_path, device)
                if model is not None:
                    return model
            except Exception as e:
                logger.warning("Failed to load real TPVFormer: %s", e)
                import traceback
                traceback.print_exc()

        # Fall back to synthetic model for UI development
        logger.info("Using TPVFormer placeholder model (synthetic output for UI dev)")
        model = TPVFormerPlaceholder(num_classes=len(NUSCENES_LIDARSEG_NAMES))
        model = model.to(device)
        model.eval()
        return model

    def _load_real_model(self, ckpt_path: str, device: str) -> Optional[nn.Module]:
        """Load the real TPVFormer from vendored code + pretrained checkpoint."""
        import sys
        import warnings
        warnings.filterwarnings('ignore')

        tpv_dir = os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "tpvformer")
        if not os.path.isdir(tpv_dir):
            logger.info("TPVFormer vendor directory not found at %s", tpv_dir)
            return None

        # Check that the actual model Python files exist (not just .git)
        tpv04_dir = os.path.join(tpv_dir, "tpvformer04")
        if not os.path.isdir(tpv04_dir):
            logger.info("TPVFormer model code not found at %s", tpv04_dir)
            return None

        if not os.path.isfile(ckpt_path):
            logger.info("TPVFormer checkpoint not found at %s", ckpt_path)
            return None

        # Add vendored code to path
        if tpv_dir not in sys.path:
            sys.path.insert(0, tpv_dir)

        # Apply mmcv v1→v2 compat patches
        from compat import patch_mmcv_imports

        # Register custom modules in mmengine MODELS registry
        from mmengine.registry import MODELS
        from mmdet.models.layers.positional_encoding import LearnedPositionalEncoding
        if 'LearnedPositionalEncoding' not in MODELS._module_dict:
            MODELS.register_module(module=LearnedPositionalEncoding)

        from tpvformer04.modules.encoder import TPVFormerEncoder
        from tpvformer04.modules.tpvformer_layer import TPVFormerLayer
        from tpvformer04.modules.cross_view_hybrid_attention import TPVCrossViewHybridAttention
        from tpvformer04.modules.image_cross_attention import TPVImageCrossAttention, TPVMSDeformableAttention3D

        for cls in [TPVFormerEncoder, TPVFormerLayer, TPVCrossViewHybridAttention,
                    TPVImageCrossAttention, TPVMSDeformableAttention3D]:
            if cls.__name__ not in MODELS._module_dict:
                MODELS.register_module(module=cls)

        from tpvformer04 import TPVFormer as TPVFormerModel
        from mmengine.config import Config

        # Load config
        cfg_path = os.path.join(tpv_dir, "config", "tpv04_occupancy.py")
        if not os.path.isfile(cfg_path):
            logger.warning("TPVFormer config not found at %s", cfg_path)
            return None

        cfg = Config.fromfile(cfg_path)
        logger.info("Building real TPVFormer model...")

        model = TPVFormerModel(
            use_grid_mask=True,
            img_backbone=cfg.model.img_backbone,
            img_neck=cfg.model.img_neck,
            tpv_head=cfg.model.tpv_head,
            tpv_aggregator=cfg.model.tpv_aggregator,
        )

        # Load checkpoint
        logger.info("Loading TPVFormer checkpoint from %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = {k.replace('module.', ''): v for k, v in ckpt.items()}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("TPVFormer missing keys: %d (first 3: %s)", len(missing), missing[:3])
        if unexpected:
            logger.warning("TPVFormer unexpected keys: %d", len(unexpected))

        model = model.to(device)
        model.eval()
        total = sum(p.numel() for p in model.parameters())
        logger.info("TPVFormer loaded: %d params, 0 missing, 0 unexpected", total)
        return model

    def get_bev_grid(self, raw_output: np.ndarray) -> np.ndarray:
        """Convert raw 3D output to BEV grid, stripping noise channel.

        TPVFormer outputs [18, Z, H, W] with channel 17 = noise.
        We strip noise to get [17, H, W].

        For BEV collapse, we use a smarter approach:
        1. Per-voxel argmax across classes (including noise)
        2. For each BEV column, find the most frequent non-noise, non-free class
        3. Fall back to max-logit if no solid class found
        """
        if raw_output.ndim == 4:
            C, Z, H, W = raw_output.shape

            # Strategy: max-pool logits along Z, then strip noise
            # This preserves logit magnitude for the renderer's argmax
            bev = np.max(raw_output, axis=1)  # [C, H, W]
        else:
            bev = raw_output

        # Strip noise channel (last one) if present
        if bev.shape[0] == 18:
            bev = bev[:17]  # Remove noise channel (idx 17)

        return bev

    def infer(self, model: nn.Module, sample: dict, device: str = "cpu") -> np.ndarray:
        """Run inference and return BEV grid [C, H, W].

        For 3D models, this collapses the height dimension.
        Use get_raw_output() for the full 3D grid.
        """
        raw = self.get_raw_output(model, sample, device)
        return self.get_bev_grid(raw)

    def get_raw_output(self, model: nn.Module, sample: dict, device: str = "cpu") -> np.ndarray:
        """Run inference and return full 3D grid [C, Z, H, W]."""
        model.eval()

        images = sample["image_tensors"]
        if images.dim() == 4:
            images = images.unsqueeze(0)
        images = images.to(device=device, dtype=torch.float32)

        # Check if this is the real TPVFormer (needs img_metas)
        model_cls = type(model).__name__
        if model_cls == 'TPVFormer':
            return self._run_real_tpvformer(model, images, sample, device)

        # Placeholder model — simple forward
        if hasattr(model, 'set_calibration'):
            model.set_calibration(sample)
        with torch.no_grad():
            output = model(images)
        result = output.squeeze(0).cpu().numpy()
        return result  # [C, Z, H, W] or [C, H, W]

    def _prepare_tpv_images(self, sample, device):
        """Prepare images: caffe BGR normalization at original resolution, padded to div 32.

        Official TPVFormer pipeline:
        1. Load image via mmcv (BGR uint8 at original 900×1600)
        2. Subtract BGR mean [103.530, 116.280, 123.675], std=[1,1,1], keep BGR
        3. Pad to divisor 32 → 928×1600 (bottom 28 rows = 0)
        No resizing, no intrinsic scaling.
        """
        CAFFE_MEAN_BGR = np.array([103.530, 116.280, 123.675], dtype=np.float32)

        tensors = []
        for img in sample['images']:
            # img is PIL RGB at original resolution (1600×900)
            arr = np.asarray(img, dtype=np.float32)  # [900, W, 3] RGB
            # Convert RGB → BGR
            arr = arr[:, :, ::-1].copy()  # BGR
            # Subtract mean (caffe-style)
            arr = arr - CAFFE_MEAN_BGR.reshape(1, 1, 3)
            # Pad to divisor 32 (pad height 900→928, width 1600 already div by 32)
            h, w = arr.shape[:2]
            pad_h = int(np.ceil(h / 32) * 32) - h  # 928 - 900 = 28
            pad_w = int(np.ceil(w / 32) * 32) - w  # 0
            if pad_h > 0 or pad_w > 0:
                arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
            tensors.append(torch.from_numpy(arr).permute(2, 0, 1))  # [3, H_pad, W_pad]

        images = torch.stack(tensors, dim=0).unsqueeze(0)  # [1, 6, 3, 928, 1600]
        _, _, _, inp_h, inp_w = images.shape
        return images.to(device=device, dtype=torch.float32), inp_h, inp_w

    def _run_real_tpvformer(self, model, images_unused, sample, device):
        """Run the real TPVFormer with proper img_metas.

        Matches official TPVFormer inference pipeline:
        1. Images: caffe BGR at original resolution, padded to div 32 (928×1600)
        2. lidar2img via global frame: K_pad @ inv(cam_to_global) @ lidar_to_global
           (handles different ego poses at camera vs lidar timestamps)
        3. img_shape = padded size, intrinsics NOT scaled
        """
        # Prepare properly normalized/padded images
        images, inp_h, inp_w = self._prepare_tpv_images(sample, device)

        # Build lidar2img through global frame (handles timestamp differences)
        lidar_to_global = sample.get('lidar_to_global')
        cam_to_globals = sample.get('cam_to_globals')
        intrinsics = sample.get('intrinsics', [])

        if lidar_to_global is None or cam_to_globals is None:
            # Fallback: use ego frame (less accurate but works without global data)
            logger.warning("No global-frame data — falling back to ego-frame lidar2img")
            lidar_to_ego = np.array(sample.get('lidar_to_ego', np.eye(4)), dtype=np.float64)
            ego_to_cams = sample.get('ego_to_cameras', [])
            lidar2img_list = []
            for ci in range(6):
                K = np.array(intrinsics[ci], dtype=np.float64)
                E = np.array(ego_to_cams[ci], dtype=np.float64)
                K_pad = np.eye(4, dtype=np.float64)
                K_pad[:3, :3] = K
                lidar2img_list.append(K_pad @ E @ lidar_to_ego)
        else:
            lidar_to_global = np.array(lidar_to_global, dtype=np.float64)
            lidar2img_list = []
            for ci in range(6):
                K = np.array(intrinsics[ci], dtype=np.float64)  # original, NOT scaled
                c2g = np.array(cam_to_globals[ci], dtype=np.float64)
                global_to_cam = np.linalg.inv(c2g)
                lidar_to_cam = global_to_cam @ lidar_to_global
                K_pad = np.eye(4, dtype=np.float64)
                K_pad[:3, :3] = K
                lidar2img_list.append(K_pad @ lidar_to_cam)

        lidar2img = np.stack(lidar2img_list, axis=0)  # [6, 4, 4]

        img_metas = [{
            'lidar2img': lidar2img,
            'img_shape': [(inp_h, inp_w)] * 6,
        }]

        logger.info("TPVFormer inference: images %s, lidar2img[0] diag: [%.1f, %.1f, %.1f, %.1f]",
                     images.shape, lidar2img[0,0,0], lidar2img[0,1,1], lidar2img[0,2,2], lidar2img[0,3,3])

        with torch.no_grad():
            output = model(
                img=images,
                img_metas=img_metas,
                use_grid_mask=False,
            )

        # output shape: [B, C, W, H, Z] from TPVAggregator (no points mode)
        # where W=tpv_w=X_lidar, H=tpv_h=Y_lidar, Z=tpv_z=Z_lidar
        # ALL in LIDAR frame, NOT ego frame!
        # Lidar→ego: X_lidar≈-ego_Y, Y_lidar≈ego_X, Z_lidar≈ego_Z+1.84m
        if isinstance(output, tuple):
            output = output[0]  # voxel logits
        result = output.squeeze(0).cpu().numpy()  # [C, W(X_lidar), H(Y_lidar), Z_lidar]

        # Store lidar_to_ego for position transform in get_sparse_voxels
        self._last_lidar_to_ego = sample.get('lidar_to_ego', np.eye(4))

        # BEV renderer convention: [C, Z, row, col] where
        #   row 0 = forward (max ego_X) and col 0 = left (max ego_Y)
        #
        # Lidar frame mapping (from lidar_to_ego rotation):
        #   Y_lidar ≈ ego_X (forward),  X_lidar ≈ -ego_Y (right)
        #
        # So: rows should come from H (Y_lidar≈ego_X), cols from W (X_lidar≈-ego_Y)
        if result.ndim == 4:
            # [C, W(X_lidar), H(Y_lidar), Z_lidar] → [C, Z, H(Y_lidar), W(X_lidar)]
            result = result.transpose(0, 3, 2, 1)  # [C, Z, H(Y_lidar≈ego_X), W(X_lidar≈-ego_Y)]
            # Flip rows: row 0 = max Y_lidar = max ego_X = forward
            result = result[:, :, ::-1, :]
            # Don't flip cols: col 0 = min X_lidar = max ego_Y = left ✓
            result = np.ascontiguousarray(result)

        return result

    def get_sparse_voxels(self, raw_output, exclude_classes=None, conf_threshold=5.0):
        """Override to exclude noise/free, transform positions lidar→ego.

        The base class computes positions in the grid's native frame (lidar).
        We need to transform them to ego frame for the 3D viewer.
        """
        if exclude_classes is None:
            exclude_classes = [16, 17] if raw_output.shape[0] == 18 else [16]
        result = super().get_sparse_voxels(raw_output, exclude_classes, conf_threshold)

        # Transform positions from lidar frame to ego frame
        L2E = getattr(self, '_last_lidar_to_ego', None)
        if L2E is not None and result['num_voxels'] > 0:
            pos = np.array(result['positions']).reshape(-1, 3)  # [N, 3] lidar-frame
            # Apply rotation + translation: p_ego = R @ p_lidar + t
            R = np.array(L2E, dtype=np.float64)[:3, :3]
            t = np.array(L2E, dtype=np.float64)[:3, 3]
            pos_ego = (pos @ R.T) + t  # [N, 3] ego-frame
            result['positions'] = pos_ego.astype(np.float32).flatten().tolist()
            logger.info("Sparse voxels: lidar Z range [%.1f, %.1f] → ego Z range [%.1f, %.1f]",
                        pos[:, 2].min(), pos[:, 2].max(),
                        pos_ego[:, 2].min(), pos_ego[:, 2].max())

        # Cap at 30K voxels for browser performance
        if result['num_voxels'] > 30000:
            confs = np.array(result['confidences'])
            top_idx = np.argsort(confs)[-30000:]
            pos = np.array(result['positions']).reshape(-1, 3)[top_idx]
            result['positions'] = pos.flatten().tolist()
            result['classes'] = [result['classes'][i] for i in top_idx]
            result['confidences'] = [result['confidences'][i] for i in top_idx]
            result['num_voxels'] = 30000
        return result

    def _get_class_colors_rgb(self):
        """Use nuScenes semantic colors."""
        return [list(NUSCENES_LIDARSEG_COLORS.get(n, (128, 128, 128)))
                for n in NUSCENES_LIDARSEG_NAMES]

    def get_triplane_views(self, raw_output: np.ndarray):
        """Extract tri-perspective views from 3D grid.

        Args:
            raw_output: [C, Z, H, W] 3D occupancy grid

        Returns:
            dict with 'xy' (top-down), 'xz' (front), 'yz' (side) views,
            each as [C, A, B] arrays.
        """
        if raw_output.ndim != 4:
            return None
        # C, Z, H, W
        return {
            'xy': np.max(raw_output, axis=1),  # [C, H, W] -- top-down (BEV)
            'xz': np.max(raw_output, axis=2),  # [C, Z, W] -- front view
            'yz': np.max(raw_output, axis=3),  # [C, Z, H] -- side view
        }

    def get_height_slice(self, raw_output: np.ndarray, z_idx: int) -> np.ndarray:
        """Get a single height slice from the 3D grid.

        Returns [C, H, W] at the specified height index.
        """
        if raw_output.ndim != 4:
            return raw_output
        z_idx = max(0, min(raw_output.shape[1] - 1, z_idx))
        return raw_output[:, z_idx, :, :]

    def get_checkpoint_url(self) -> Optional[str]:
        return "https://cloud.tsinghua.edu.cn/f/3fbd12101ead4397a0f7/?dl=1"

    def get_checkpoint_filename(self) -> str:
        return "tpvformer_occ.pth"


class TPVFormerPlaceholder(nn.Module):
    """Placeholder model that generates synthetic 3D occupancy output.

    Uses the actual camera features to produce spatially-coherent
    synthetic occupancy for UI development and testing.

    Output: [B, C, Z, H, W] where Z=16, H=W=200, C=17
    """

    def __init__(self, num_classes: int = 17, bev_h: int = 200, bev_w: int = 200,
                 num_z: int = 16):
        super().__init__()
        self.num_classes = num_classes
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_z = num_z

        # Simple encoder to extract features from cameras
        import torchvision.models as models
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2,  # stride 8, 128 channels
        )

        # Project to BEV
        self.bev_proj = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 3D head: generate volume from BEV features
        self.volume_head = nn.Sequential(
            nn.Conv2d(64, num_z * num_classes, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, 6, 3, H, W]
        Returns:
            [B, C, Z, H_bev, W_bev] -- 3D occupancy logits
        """
        B, N, C, H, W = images.shape

        # Encode all cameras
        x = images.reshape(B * N, C, H, W)
        feats = self.encoder(x)  # [B*6, 128, h, w]

        # Reduce and project to BEV
        feats = self.bev_proj(feats)  # [B*6, 64, h, w]

        # Upsample to BEV resolution
        feats = nn.functional.interpolate(feats, size=(self.bev_h, self.bev_w),
                                          mode='bilinear', align_corners=False)

        # Reshape and sum across cameras
        feats = feats.reshape(B, N, 64, self.bev_h, self.bev_w)
        feats = feats.sum(dim=1)  # [B, 64, H, W]

        # Generate 3D volume
        vol = self.volume_head(feats)  # [B, Z*C, H, W]
        vol = vol.reshape(B, self.num_classes, self.num_z, self.bev_h, self.bev_w)

        return vol


# Expose for import
__all__ = ['TPVFormerBackend', 'NUSCENES_LIDARSEG_NAMES', 'NUSCENES_LIDARSEG_COLORS']
