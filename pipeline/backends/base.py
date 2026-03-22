"""Abstract base class for model backends."""

import abc
import logging
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BEVBackend(abc.ABC):
    """Abstract interface for all model backends.

    Each backend wraps a specific model architecture and provides:
    - Model loading with pretrained weights
    - Inference producing a standardized BEV grid [C, H, W]
    - Captum-compatible forward function for attribution
    - Metadata (class names, representation type, etc.)
    """

    name: str = "base"
    repr_type: str = "bev_seg"  # 'bev_seg' | '3d_occ' | 'gaussian'

    @property
    @abc.abstractmethod
    def class_names(self) -> List[str]:
        """List of class names this model predicts."""
        ...

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @abc.abstractmethod
    def load(self, device: str = "cpu") -> nn.Module:
        """Load model with pretrained weights. Returns nn.Module."""
        ...

    @abc.abstractmethod
    def infer(self, model: nn.Module, sample: dict, device: str = "cpu") -> np.ndarray:
        """Run inference. Returns [C, H, W] BEV grid (logits or probabilities)."""
        ...

    def get_bev_grid(self, raw_output: np.ndarray) -> np.ndarray:
        """Convert raw model output to [C, H, W] BEV grid.

        Default: pass through (assumes output is already 2D BEV).
        Override for 3D occupancy models to collapse height dimension.
        """
        if raw_output.ndim == 4:
            # [C, D, H, W] -> [C, H, W] via max pooling along depth/height
            return np.max(raw_output, axis=1)
        return raw_output

    def make_captum_forward(self, model: nn.Module, sample: dict,
                            cell_i: int, cell_j: int, class_idx: int = 0):
        """Create Captum-compatible forward function.

        Returns a callable: images [B, 6, 3, H, W] -> scalar
        """
        model.eval()
        if hasattr(model, 'set_calibration'):
            model.set_calibration(sample)

        def _forward(images: torch.Tensor) -> torch.Tensor:
            logits = model(images)
            if logits.dim() == 5:  # [B, C, D, H, W] -> collapse D
                logits = logits.max(dim=2).values
            return logits[0, class_idx, cell_i, cell_j]

        return _forward

    def get_raw_output(self, model: nn.Module, sample: dict, device: str = "cpu"):
        """Get the raw (possibly 3D) model output before BEV collapse.

        Returns the full output tensor as numpy. Useful for 3D visualization.
        Default implementation calls infer().
        """
        return self.infer(model, sample, device)

    @property
    def num_cameras(self) -> int:
        """Expected number of input cameras. Override for single-image models."""
        return 6

    @property
    def task_type(self) -> str:
        """Task type: 'bev_seg' | '3d_occ' | 'img_seg'. Default matches repr_type."""
        return self.repr_type

    def get_sparse_voxels(self, raw_output: np.ndarray,
                          exclude_classes: Optional[List[int]] = None,
                          conf_threshold: float = 0.0) -> dict:
        """Convert raw 3D output to sparse voxel representation.

        Args:
            raw_output: [C, Z, H, W] voxel grid (logits)
            exclude_classes: class indices to skip (e.g., noise, free)
            conf_threshold: minimum logit to include

        Returns:
            dict with positions, classes, confidences, metadata
        """
        if raw_output.ndim != 4:
            return {'num_voxels': 0, 'positions': [], 'classes': [], 'confidences': []}

        C, Z, H, W = raw_output.shape

        # Strip excluded classes BEFORE argmax so they don't dominate
        if exclude_classes:
            keep = [i for i in range(C) if i not in exclude_classes]
            filtered = raw_output[keep]  # [C', Z, H, W]
            class_map_local = np.argmax(filtered, axis=0)  # indices into keep[]
            class_map = np.array(keep)[class_map_local]    # remap to original indices
            conf_map = np.max(filtered, axis=0)
        else:
            class_map = np.argmax(raw_output, axis=0)
            conf_map = np.max(raw_output, axis=0)

        # Build mask: above threshold
        mask = conf_map > conf_threshold

        # Extract sparse positions
        zz, hh, ww = np.where(mask)
        classes = class_map[zz, hh, ww].astype(np.int32)
        confs = conf_map[zz, hh, ww].astype(np.float32)

        # Convert grid indices to world coordinates
        # Convention: H=rows=X(forward), W=cols=Y(left), Z=height
        grid_range = 51.2  # default, override in subclass if needed
        z_min, z_max = -5.0, 3.0
        res_xy = 2 * grid_range / max(H, W)
        res_z = (z_max - z_min) / Z

        # X: row 0 = max_x (forward), row H-1 = min_x
        x_coords = grid_range - (hh + 0.5) * res_xy
        # Y: col 0 = max_y (left), col W-1 = min_y
        y_coords = grid_range - (ww + 0.5) * res_xy
        # Z: bin 0 = z_min, bin Z-1 = z_max
        z_coords = z_min + (zz + 0.5) * res_z

        positions = np.stack([x_coords, y_coords, z_coords], axis=1).astype(np.float32)

        return {
            'num_voxels': int(len(zz)),
            'positions': positions.flatten().tolist(),
            'classes': classes.tolist(),
            'confidences': confs.tolist(),
            'grid_dims': [int(Z), int(H), int(W)],
            'voxel_size': [float(res_z), float(res_xy), float(res_xy)],
            'class_names': self.class_names,
            'class_colors': self._get_class_colors_rgb(),
        }

    def _get_class_colors_rgb(self) -> List[List[int]]:
        """Return RGB colors for each class. Override for custom palettes."""
        import colorsys
        colors = []
        for i in range(self.num_classes):
            hue = (i * 0.618033988749895) % 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.9)
            colors.append([int(r * 255), int(g * 255), int(b * 255)])
        return colors

    def get_checkpoint_url(self) -> Optional[str]:
        """URL to download pretrained checkpoint, if available."""
        return None

    def get_checkpoint_filename(self) -> str:
        """Expected checkpoint filename in checkpoints/ directory."""
        return f"{self.name}_model.pt"
