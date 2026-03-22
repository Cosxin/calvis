"""
BEV model loading and inference for the BEV Attribution Debug Tool.

Two strategies:
  1. **MMDet3D path** — load BEVFormer-tiny via mmdet3d's init_model() API.
     Requires mmdet3d, mmcv, and mmengine to be installed and a valid config +
     checkpoint.
  2. **Fallback SimpleBEVModel** — a lightweight ResNet-18 backbone with a
     learned BEV projection (lift-splat style).  Works out-of-the-box with
     only torch/torchvision and produces differentiable BEV grids suitable
     for Captum attribution.
"""

import logging
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Constants
# --------------------------------------------------------------------------- #

NUM_CLASSES = 10          # vehicle, pedestrian, barrier, etc.
BEV_H = 200              # BEV grid height  (200 cells @ 0.5 m = 100 m)
BEV_W = 200              # BEV grid width
NUM_CAMERAS = 6
FEATURE_DIM = 256         # intermediate BEV feature channels
IMG_H = 256               # expected input image height
IMG_W = 704               # expected input image width

# Default checkpoint directory.
CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")


# --------------------------------------------------------------------------- #
#  SimpleBEVModel  (fallback — always available)
# --------------------------------------------------------------------------- #

class CameraEncoder(nn.Module):
    """Shared per-camera encoder: ResNet-18 backbone → compact feature map."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        import torchvision.models as models

        resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        # Take everything up to (and including) layer3 → feature stride 16.
        self.backbone = nn.Sequential(
            resnet.conv1,   # /2
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,  # /4
            resnet.layer1,
            resnet.layer2,   # /8
            resnet.layer3,   # /16  → channels = 256
        )
        self.out_channels = 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B*N, 3, H, W] where N = number of cameras.
        Returns:
            [B*N, 256, H/16, W/16]
        """
        return self.backbone(x)


class BEVProjection(nn.Module):
    """Project per-camera image features into a shared BEV grid.

    Each camera's feature map is reduced via 1x1 conv, upsampled to
    BEV grid resolution, and then all cameras are summed together.
    A fusion conv refines the combined BEV features.

    This is intentionally lightweight — accuracy is not the goal; producing
    meaningful gradients for attribution visualisation is.
    """

    def __init__(
        self,
        in_channels: int = 256,
        bev_channels: int = FEATURE_DIM,
        bev_h: int = BEV_H,
        bev_w: int = BEV_W,
        num_cameras: int = NUM_CAMERAS,
    ):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_cameras = num_cameras
        self.bev_channels = bev_channels

        # Per-camera: reduce channels and prepare for BEV projection.
        self.cam_reduce = nn.Sequential(
            nn.Conv2d(in_channels, bev_channels, 1),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU(inplace=True),
        )

        # Fusion conv after summing all cameras.
        self.fuse = nn.Sequential(
            nn.Conv2d(bev_channels, bev_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, cam_feats: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """
        Args:
            cam_feats: [B*N, C, h, w] — per-camera backbone features.
            batch_size: B
        Returns:
            [B, bev_channels, bev_h, bev_w]
        """
        BN, C, h, w = cam_feats.shape
        N = self.num_cameras

        # Reduce channels: [B*N, C, h, w] → [B*N, bev_channels, h, w]
        reduced = self.cam_reduce(cam_feats)

        # Upsample each camera's features to BEV grid size.
        # [B*N, bev_channels, bev_h, bev_w]
        upsampled = F.interpolate(
            reduced, size=(self.bev_h, self.bev_w),
            mode='bilinear', align_corners=False,
        )

        # Reshape to [B, N, bev_channels, bev_h, bev_w] and sum across cameras.
        upsampled = upsampled.reshape(
            batch_size, N, self.bev_channels, self.bev_h, self.bev_w
        )
        bev = upsampled.sum(dim=1)  # [B, bev_channels, bev_h, bev_w]

        # Refine with fusion conv.
        bev = self.fuse(bev)
        return bev


class BEVHead(nn.Module):
    """Classification head: BEV features → per-cell class logits."""

    def __init__(
        self,
        in_channels: int = FEATURE_DIM,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, 1),
        )

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bev: [B, in_channels, H, W]
        Returns:
            [B, num_classes, H, W]
        """
        return self.head(bev)


class SimpleBEVModel(nn.Module):
    """Fallback BEV model: ResNet-18 encoder + learned BEV projection + head.

    Input : [B, 6, 3, H, W]  multi-camera images
    Output: [B, num_classes, 200, 200]  BEV class-logit grid
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        bev_h: int = BEV_H,
        bev_w: int = BEV_W,
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.num_cameras = NUM_CAMERAS
        self.num_classes = num_classes
        self.bev_h = bev_h
        self.bev_w = bev_w

        self.encoder = CameraEncoder(pretrained=pretrained_backbone)
        self.bev_projection = BEVProjection(
            in_channels=self.encoder.out_channels,
            bev_channels=FEATURE_DIM,
            bev_h=bev_h,
            bev_w=bev_w,
            num_cameras=NUM_CAMERAS,
        )
        self.bev_head = BEVHead(
            in_channels=FEATURE_DIM,
            num_classes=num_classes,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, 6, 3, H, W] — six surround-camera images, normalised.
        Returns:
            [B, num_classes, bev_h, bev_w] — per-cell class logits.
        """
        B, N, C, H, W = images.shape
        assert N == self.num_cameras, (
            f"Expected {self.num_cameras} cameras, got {N}"
        )

        # Encode all cameras in a single batched pass.
        x = images.reshape(B * N, C, H, W)          # [B*6, 3, H, W]
        cam_feats = self.encoder(x)                   # [B*6, 256, h, w]

        # Project to BEV.
        bev_feats = self.bev_projection(cam_feats, B)  # [B, 256, 200, 200]

        # Classification head.
        logits = self.bev_head(bev_feats)              # [B, C_cls, 200, 200]
        return logits


# --------------------------------------------------------------------------- #
#  MMDet3D BEVFormer loader (optional)
# --------------------------------------------------------------------------- #

def _try_load_mmdet3d(
    config_path: str, checkpoint_path: str, device: str
) -> Optional[nn.Module]:
    """Attempt to load a BEVFormer model via mmdet3d.

    Returns the model on success, or None if mmdet3d is unavailable or the
    config/checkpoint is invalid.
    """
    try:
        from mmdet3d.apis import init_model as mmdet3d_init_model
    except ImportError:
        logger.info("mmdet3d not installed — skipping MMDet3D model path.")
        return None

    if not os.path.isfile(config_path):
        logger.warning("MMDet3D config not found: %s", config_path)
        return None
    if not os.path.isfile(checkpoint_path):
        logger.warning("MMDet3D checkpoint not found: %s", checkpoint_path)
        return None

    try:
        logger.info("Loading BEVFormer via mmdet3d from %s", checkpoint_path)
        model = mmdet3d_init_model(config_path, checkpoint_path, device=device)
        model.eval()
        return model
    except Exception as exc:
        logger.warning("Failed to load MMDet3D model: %s", exc)
        return None


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #

def load_model(
    checkpoint_path: Optional[str] = None,
    config_path: Optional[str] = None,
    device: str = "cpu",
    force_simple: bool = False,
) -> nn.Module:
    """Load a BEV model for inference and attribution.

    Strategy:
      1. If ``force_simple`` is False and mmdet3d is available, try loading
         BEVFormer-tiny from ``config_path`` / ``checkpoint_path``.
      2. Otherwise, instantiate :class:`SimpleBEVModel` (always works).
         If ``checkpoint_path`` points to a SimpleBEVModel state-dict, load it.

    Parameters
    ----------
    checkpoint_path : str, optional
        Path to a model checkpoint (.pth).  If None, default locations are
        checked under ``checkpoints/``.
    config_path : str, optional
        MMDet3D config file for BEVFormer.
    device : str
        'cpu' or 'cuda' / 'cuda:0' etc.
    force_simple : bool
        If True, skip the mmdet3d path entirely and use SimpleBEVModel.

    Returns
    -------
    nn.Module in eval mode, on the requested device.
    """
    # --- Resolve default paths ---
    ckpt_dir = os.path.abspath(CKPT_DIR)
    if checkpoint_path is None:
        # Try common checkpoint names.
        candidates = [
            os.path.join(ckpt_dir, "bevformer_tiny_epoch_24.pth"),
            os.path.join(ckpt_dir, "bevformer_tiny.pth"),
            os.path.join(ckpt_dir, "simple_bev.pth"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                checkpoint_path = c
                break

    if config_path is None:
        config_path = os.path.join(ckpt_dir, "bevformer_tiny.py")

    # --- Strategy 1: MMDet3D ---
    if not force_simple and checkpoint_path is not None:
        model = _try_load_mmdet3d(config_path, checkpoint_path, device)
        if model is not None:
            logger.info("Loaded BEVFormer via mmdet3d successfully.")
            return model

    # --- Strategy 2: LSS (Lift-Splat-Shoot) with pretrained checkpoint ---
    if not force_simple:
        lss_ckpt = os.path.join(ckpt_dir, "lss_model.pt")
        if os.path.isfile(lss_ckpt):
            from pipeline.lss_model import create_lss_model

            lss_model = create_lss_model(
                checkpoint_path=lss_ckpt, device=device,
            )
            if lss_model is not None:
                logger.info("Loaded LSS (Lift-Splat-Shoot) model with pretrained weights.")
                return lss_model

    # --- Strategy 3: SimpleBEVModel (fallback — always works) ---
    logger.info("Using SimpleBEVModel fallback.")
    model = SimpleBEVModel(pretrained_backbone=True)

    if checkpoint_path is not None and os.path.isfile(checkpoint_path):
        try:
            state_dict = torch.load(
                checkpoint_path, map_location=device, weights_only=True
            )
            # Handle state dicts wrapped in a 'state_dict' key.
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict, strict=False)
            logger.info(
                "Loaded SimpleBEVModel weights from %s", checkpoint_path
            )
        except Exception as exc:
            logger.warning(
                "Could not load checkpoint into SimpleBEVModel "
                "(this is fine for untrained inference): %s",
                exc,
            )

    model = model.to(device)
    model.eval()
    return model
