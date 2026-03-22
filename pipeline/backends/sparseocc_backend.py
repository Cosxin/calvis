"""SparseOcc backend — Sparse 3D Occupancy Prediction.

Paper: "Fully Sparse 3D Occupancy Prediction" (ECCV 2024)
Repo: https://github.com/MCG-NJU/SparseOcc

Output: Sparse 3D occupancy predictions that are converted to a dense
voxel grid [C, Z, H, W] for visualization.
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

# Occ3D-nuScenes classes (17 classes including free space)
OCC3D_CLASS_NAMES = [
    'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain',
    'manmade', 'vegetation', 'free',
]


class SparseOccBackend(BEVBackend):
    """SparseOcc backend for fully sparse 3D occupancy prediction.

    The model predicts sparse voxel occupancy which is then densified
    into a [C, Z, H, W] grid for BEV visualization.
    """

    name = "sparseocc"
    repr_type = "3d_occ"

    @property
    def class_names(self) -> List[str]:
        return list(OCC3D_CLASS_NAMES)

    def load(self, device: str = "cpu") -> Optional[nn.Module]:
        ckpt_path = os.path.join(CKPT_DIR, "sparseocc_r50.pth")

        try:
            model = self._load_real_model(ckpt_path, device)
            if model is not None:
                return model
        except Exception as e:
            logger.warning("Failed to load real SparseOcc: %s", e)

        logger.info("Using SparseOcc placeholder model (synthetic output)")
        model = SparseOccPlaceholder(num_classes=len(OCC3D_CLASS_NAMES))
        model = model.to(device)
        model.eval()
        return model

    def _load_real_model(self, ckpt_path: str, device: str) -> Optional[nn.Module]:
        """Attempt to load real SparseOcc from vendored code."""
        so_dir = os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "sparseocc")
        if not os.path.isdir(so_dir):
            logger.info("SparseOcc vendor directory not found at %s", so_dir)
            return None
        if not os.path.isfile(ckpt_path):
            logger.info("SparseOcc checkpoint not found at %s", ckpt_path)
            return None
        return None

    def infer(self, model: nn.Module, sample: dict, device: str = "cpu") -> np.ndarray:
        raw = self.get_raw_output(model, sample, device)
        return self.get_bev_grid(raw)

    def get_raw_output(self, model: nn.Module, sample: dict, device: str = "cpu") -> np.ndarray:
        """Run inference and return dense 3D grid [C, Z, H, W]."""
        model.eval()
        if hasattr(model, 'set_calibration'):
            model.set_calibration(sample)

        images = sample["image_tensors"]
        if images.dim() == 4:
            images = images.unsqueeze(0)
        images = images.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            output = model(images)

        # Handle sparse output format
        if isinstance(output, dict) and 'sparse_voxels' in output:
            return self._sparse_to_dense(output).squeeze(0).cpu().numpy()

        return output.squeeze(0).cpu().numpy()

    def _sparse_to_dense(self, sparse_output: dict,
                          grid_size=(200, 200, 16)) -> torch.Tensor:
        """Convert sparse voxel predictions to dense grid."""
        coords = sparse_output['sparse_voxels']   # [N, 3] (x, y, z indices)
        logits = sparse_output['sparse_logits']    # [N, C] class logits

        H, W, Z = grid_size
        C = logits.shape[-1]

        dense = torch.zeros(1, C, Z, H, W, device=logits.device)

        # Clamp indices
        xi = coords[:, 0].long().clamp(0, W-1)
        yi = coords[:, 1].long().clamp(0, H-1)
        zi = coords[:, 2].long().clamp(0, Z-1)

        for n in range(coords.shape[0]):
            dense[0, :, zi[n], yi[n], xi[n]] = logits[n]

        return dense

    def get_checkpoint_url(self) -> Optional[str]:
        # Google Drive link — user may need to download manually
        return None

    def get_checkpoint_filename(self) -> str:
        return "sparseocc_r50.pth"


class SparseOccPlaceholder(nn.Module):
    """Placeholder that generates synthetic sparse-then-dense occupancy output.

    Output: [B, C, Z, H, W] dense voxel grid
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

        self.bev_reduce = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Sparse-style: predict occupancy probability + class logits
        self.occ_head = nn.Sequential(
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
        vol = self.occ_head(feats)
        vol = vol.reshape(B, self.num_classes, self.num_z, self.bev_h, self.bev_w)
        return vol


__all__ = ['SparseOccBackend', 'OCC3D_CLASS_NAMES']
