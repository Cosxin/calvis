"""GaussianFormer backend — 3D Gaussian scene representation.

Paper: "GaussianFormer: Scene as Gaussians for Vision-Based 3D Semantic Occupancy Prediction" (ECCV 2024)
Repo: https://github.com/huang-yh/GaussianFormer

Output: Set of 3D Gaussians, each with:
  - center (x, y, z)
  - covariance (3x3 or 6 params)
  - semantic logits (C classes)

These are splatted to a voxel grid for visualization.
"""

import logging
import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from .base import BEVBackend

logger = logging.getLogger(__name__)

CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints")

# Same classes as TPVFormer (nuScenes LiDAR seg)
NUSCENES_OCC_NAMES = [
    'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain',
    'manmade', 'vegetation', 'free',
]


class GaussianFormerBackend(BEVBackend):
    """GaussianFormer backend for 3D occupancy via Gaussian splatting.

    Outputs are 3D Gaussians that get rasterized to a voxel grid [C, Z, H, W]
    for BEV visualization.
    """

    name = "gaussianformer"
    repr_type = "gaussian"

    @property
    def class_names(self) -> List[str]:
        return list(NUSCENES_OCC_NAMES)

    def load(self, device: str = "cpu") -> Optional[nn.Module]:
        """Load GaussianFormer model.

        Strategy:
        1. Check for vendored code + checkpoint
        2. Fall back to placeholder model
        """
        ckpt_path = os.path.join(CKPT_DIR, "gaussianformer.pth")

        try:
            model = self._load_real_model(ckpt_path, device)
            if model is not None:
                return model
        except Exception as e:
            logger.warning("Failed to load real GaussianFormer: %s", e)

        logger.info("Using GaussianFormer placeholder model (synthetic output)")
        model = GaussianFormerPlaceholder(num_classes=len(NUSCENES_OCC_NAMES))
        model = model.to(device)
        model.eval()
        return model

    def _load_real_model(self, ckpt_path: str, device: str) -> Optional[nn.Module]:
        """Attempt to load real GaussianFormer from vendored code."""
        gf_dir = os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "gaussianformer")
        if not os.path.isdir(gf_dir):
            logger.info("GaussianFormer vendor directory not found at %s", gf_dir)
            return None
        if not os.path.isfile(ckpt_path):
            logger.info("GaussianFormer checkpoint not found at %s", ckpt_path)
            return None
        return None

    def infer(self, model: nn.Module, sample: dict, device: str = "cpu") -> np.ndarray:
        """Run inference and return BEV grid [C, H, W]."""
        raw = self.get_raw_output(model, sample, device)
        return self.get_bev_grid(raw)

    def get_raw_output(self, model: nn.Module, sample: dict, device: str = "cpu") -> np.ndarray:
        """Run inference and return raw output.

        For real GaussianFormer: returns Gaussian parameters + splatted voxel grid.
        For placeholder: returns [C, Z, H, W] directly.
        """
        model.eval()
        if hasattr(model, 'set_calibration'):
            model.set_calibration(sample)

        images = sample["image_tensors"]
        if images.dim() == 4:
            images = images.unsqueeze(0)
        images = images.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            output = model(images)

        # If model returns a dict with gaussians and voxels
        if isinstance(output, dict):
            voxels = output.get('voxels', output.get('occ', None))
            if voxels is not None:
                return voxels.squeeze(0).cpu().numpy()
            # Convert gaussians to voxel grid
            return self._gaussians_to_voxels(output).cpu().numpy()

        return output.squeeze(0).cpu().numpy()

    def _gaussians_to_voxels(self, gaussian_output: dict,
                              grid_size=(200, 200, 16),
                              grid_range=51.2) -> torch.Tensor:
        """Convert Gaussian parameters to a voxel occupancy grid.

        This is a simplified version — the real GaussianFormer uses
        CUDA kernels for efficient splatting.
        """
        centers = gaussian_output['centers']      # [N, 3]
        semantics = gaussian_output['semantics']   # [N, C]

        H, W, Z = grid_size
        C = semantics.shape[-1]

        # Discretize Gaussian centers to voxel indices
        # Assume ego frame: X=fwd, Y=left, Z=up
        # Grid: X in [-range, range], Y in [-range, range], Z in [-5, 3]
        voxels = torch.zeros(C, Z, H, W, device=centers.device)

        xi = ((centers[:, 0] + grid_range) / (2 * grid_range) * W).long().clamp(0, W-1)
        yi = ((centers[:, 1] + grid_range) / (2 * grid_range) * H).long().clamp(0, H-1)
        zi = ((centers[:, 2] + 5.0) / 8.0 * Z).long().clamp(0, Z-1)

        for n in range(centers.shape[0]):
            voxels[:, zi[n], yi[n], xi[n]] += semantics[n]

        return voxels

    def get_gaussians(self, model: nn.Module, sample: dict, device: str = "cpu"):
        """Get raw Gaussian parameters for 3D visualization.

        Returns dict with 'centers', 'covariances', 'semantics', 'opacities'.
        Returns None if model doesn't expose Gaussians.
        """
        if hasattr(model, 'get_gaussians'):
            model.eval()
            images = sample["image_tensors"]
            if images.dim() == 4:
                images = images.unsqueeze(0)
            images = images.to(device=device, dtype=torch.float32)
            with torch.no_grad():
                return model.get_gaussians(images)
        return None

    def get_sparse_voxels(self, raw_output, exclude_classes=None, conf_threshold=0.5):
        """Override: exclude free (class 16), threshold at 0.5 for tighter selection.

        GaussianFormer logit range is [-2.3, 1.85] — much narrower than TPVFormer.
        A threshold of 0.5 keeps ~300K voxels; cap at 30K for browser performance.
        """
        if exclude_classes is None:
            exclude_classes = [16]  # free
        result = super().get_sparse_voxels(raw_output, exclude_classes, conf_threshold)
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
        from pipeline.backends.tpvformer_backend import NUSCENES_LIDARSEG_COLORS, NUSCENES_LIDARSEG_NAMES
        return [list(NUSCENES_LIDARSEG_COLORS.get(n, (128, 128, 128)))
                for n in NUSCENES_OCC_NAMES]

    def get_checkpoint_url(self) -> Optional[str]:
        # NonEmpty variant (25600 Gaussians, 19.31 mIoU) — stable /f/ link format
        return "https://cloud.tsinghua.edu.cn/f/d1766fff8ad74756920b/?dl=1"

    def get_checkpoint_filename(self) -> str:
        return "gaussianformer.pth"


class GaussianFormerPlaceholder(nn.Module):
    """Placeholder model that generates synthetic 3D occupancy via pseudo-Gaussians.

    Uses camera features to produce spatially-coherent output for UI testing.
    Mimics the GaussianFormer output format.

    Output: [B, C, Z, H, W] voxel grid (same as TPVFormer for viz compatibility)
    """

    def __init__(self, num_classes: int = 17, bev_h: int = 200, bev_w: int = 200,
                 num_z: int = 16):
        super().__init__()
        self.num_classes = num_classes
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_z = num_z

        import torchvision.models as models
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2,
        )

        # Gaussian parameter prediction head
        self.bev_reduce = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Volume generation (simulates Gaussian splatting)
        self.volume_head = nn.Sequential(
            nn.Conv2d(64, num_z * num_classes, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = images.shape
        x = images.reshape(B * N, C, H, W)
        feats = self.encoder(x)
        feats = self.bev_reduce(feats)
        feats = nn.functional.interpolate(feats, size=(self.bev_h, self.bev_w),
                                          mode='bilinear', align_corners=False)
        feats = feats.reshape(B, N, 64, self.bev_h, self.bev_w).sum(dim=1)
        vol = self.volume_head(feats)
        vol = vol.reshape(B, self.num_classes, self.num_z, self.bev_h, self.bev_w)
        return vol


__all__ = ['GaussianFormerBackend', 'NUSCENES_OCC_NAMES']
