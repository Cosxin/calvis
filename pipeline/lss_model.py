"""
Lift-Splat-Shoot (LSS) model for BEV vehicle segmentation.

Vendored from nv-tlabs/lift-splat-shoot (BSD License).
https://github.com/nv-tlabs/lift-splat-shoot

Original authors: Jonah Philion and Sanja Fidler (NVIDIA, ECCV 2020).

This file contains:
  - The core LSS architecture (CamEncode, BevEncode, LiftSplatShoot)
  - LSSWrapper: an adapter that matches the SimpleBEVModel interface
    (forward(images) -> [B, 10, 200, 200]) and handles calibration injection.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  LSS helpers (from tools.py)
# --------------------------------------------------------------------------- #

def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([int((row[1] - row[0]) / row[2]) for row in [xbound, ybound, zbound]])
    return dx, bx, nx


def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])
    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))
    return x, geom_feats


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])
        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))
        ctx.save_for_backward(kept)
        ctx.mark_non_differentiable(geom_feats)
        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1
        val = gradx[back]
        return val, None, None


# --------------------------------------------------------------------------- #
#  LSS model components (from models.py)
# --------------------------------------------------------------------------- #

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class CamEncode(nn.Module):
    def __init__(self, D, C, downsample):
        super().__init__()
        self.D = D
        self.C = C
        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")
        self.up1 = Up(320 + 112, 512)
        self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, x):
        x = self.get_eff_depth(x)
        x = self.depthnet(x)
        depth = self.get_depth_dist(x[:, :self.D])
        new_x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)
        return depth, new_x

    def get_eff_depth(self, x):
        endpoints = dict()
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            prev_x = x
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])
        return x

    def forward(self, x):
        depth, x = self.get_depth_feat(x)
        return x


class BevEncode(nn.Module):
    def __init__(self, inC, outC):
        super().__init__()
        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu
        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3
        self.up1 = Up(64 + 256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)
        x = self.up1(x, x1)
        x = self.up2(x)
        return x


class LiftSplatShoot(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, outC):
        super().__init__()
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf

        dx, bx, nx = gen_dx_bx(
            self.grid_conf['xbound'],
            self.grid_conf['ybound'],
            self.grid_conf['zbound'],
        )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.downsample = 16
        self.camC = 64
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
        self.camencode = CamEncode(self.D, self.camC, self.downsample)
        self.bevencode = BevEncode(inC=self.camC, outC=outC)

        # Use cumsum_trick (autograd-friendly) instead of QuickCumsum
        # for Captum attribution compatibility.
        self.use_quickcumsum = False

    def create_frustum(self):
        ogfH, ogfW = self.data_aug_conf['final_dim']
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(
            *self.grid_conf['dbound'], dtype=torch.float
        ).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape

        # undo post-transformation
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((
            points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
            points[:, :, :, :, :, 2:3]
        ), 5)
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        return points

    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C"""
        B, N, C, imH, imW = x.shape
        x = x.view(B * N, C, imH, imW)
        x = self.camencode(x)
        x = x.view(B, N, self.camC, self.D, imH // self.downsample, imW // self.downsample)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([
            torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
            for ix in range(B)
        ])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (
            (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        )
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = (
            geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)
            + geom_feats[:, 1] * (self.nx[2] * B)
            + geom_feats[:, 2] * B
            + geom_feats[:, 3]
        )
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        x = self.get_cam_feats(x)
        x = self.voxel_pooling(geom, x)
        return x

    def forward(self, x, rots, trans, intrins, post_rots, post_trans):
        x = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)
        x = self.bevencode(x)
        return x


# --------------------------------------------------------------------------- #
#  Default LSS configuration (matches the official pretrained checkpoint)
# --------------------------------------------------------------------------- #

DEFAULT_GRID_CONF = {
    'xbound': [-50.0, 50.0, 0.5],
    'ybound': [-50.0, 50.0, 0.5],
    'zbound': [-10.0, 10.0, 20.0],
    'dbound': [4.0, 45.0, 1.0],
}

# The official LSS checkpoint was trained with 128x352 input images.
DEFAULT_DATA_AUG_CONF = {
    'final_dim': (128, 352),
    'H': 900,
    'W': 1600,
}

NUM_CLASSES_PAD = 10  # Pad to 10 classes to match existing BEV viz pipeline


# --------------------------------------------------------------------------- #
#  LSSWrapper — adapts LSS to the SimpleBEVModel interface
# --------------------------------------------------------------------------- #

class LSSWrapper(nn.Module):
    """Wraps LiftSplatShoot to match the SimpleBEVModel interface.

    - Accepts ``forward(images)`` with ``[B, 6, 3, H, W]`` input.
    - Returns ``[B, 10, 200, 200]`` (padded from 1-ch vehicle segmentation).
    - Calibration is injected via ``set_calibration(sample)`` before forward.
    """

    def __init__(self, lss_model: LiftSplatShoot):
        super().__init__()
        self.lss = lss_model
        # Calibration state (set before each forward pass via set_calibration).
        self._rots: Optional[torch.Tensor] = None
        self._trans: Optional[torch.Tensor] = None
        self._intrins: Optional[torch.Tensor] = None
        self._post_rots: Optional[torch.Tensor] = None
        self._post_trans: Optional[torch.Tensor] = None

    def set_calibration(self, sample: dict):
        """Extract and store calibration from a data sample dict.

        LSS expects:
          rots:       [B, N, 3, 3]  sensor-to-ego rotation
          trans:      [B, N, 3]     sensor-to-ego translation
          intrins:    [B, N, 3, 3]  camera intrinsic matrices
          post_rots:  [B, N, 3, 3]  image augmentation rotation (identity)
          post_trans: [B, N, 3]     image augmentation translation (zeros)

        The sample dict provides:
          ego_to_cameras: list of 6 [4,4] numpy arrays  (ego → camera)
          intrinsics:     list of 6 [3,3] numpy arrays  (camera intrinsics)
          original_sizes: list of 6 (W,H) tuples
        """
        device = next(self.lss.parameters()).device
        N = 6  # number of cameras

        ego_to_cams = sample.get("ego_to_cameras", sample.get("extrinsics", []))
        raw_intrinsics = sample["intrinsics"]

        rots_list = []
        trans_list = []
        intrins_list = []
        post_rots_list = []
        post_trans_list = []

        # LSS was trained with images resized to (128, 352) from original (900, 1600).
        # We need to account for this resize in the calibration.
        final_h, final_w = DEFAULT_DATA_AUG_CONF['final_dim']
        orig_h, orig_w = DEFAULT_DATA_AUG_CONF['H'], DEFAULT_DATA_AUG_CONF['W']

        for i in range(N):
            # ego_to_camera = inv(sensor_to_ego)
            # So sensor_to_ego = inv(ego_to_camera)
            ego_to_cam = np.array(ego_to_cams[i], dtype=np.float64)
            sensor_to_ego = np.linalg.inv(ego_to_cam)

            rot = torch.tensor(sensor_to_ego[:3, :3], dtype=torch.float32)
            tran = torch.tensor(sensor_to_ego[:3, 3], dtype=torch.float32)

            # Camera intrinsics — use original resolution intrinsics.
            K = torch.tensor(np.array(raw_intrinsics[i], dtype=np.float64), dtype=torch.float32)

            rots_list.append(rot)
            trans_list.append(tran)
            intrins_list.append(K)

            # The post_rots/post_trans encode how the image was resized/cropped
            # from the original resolution to the final input resolution.
            # For a simple resize (no crop, no flip, no rotation):
            #   post_rot = diag(sx, sy, 1), post_trans = 0
            # where sx = final_w / orig_w, sy = final_h / orig_h.
            sx = final_w / orig_w
            sy = final_h / orig_h
            post_rot = torch.tensor([
                [sx, 0, 0],
                [0, sy, 0],
                [0, 0, 1],
            ], dtype=torch.float32)
            post_tran = torch.zeros(3, dtype=torch.float32)

            post_rots_list.append(post_rot)
            post_trans_list.append(post_tran)

        # Stack and add batch dimension: [1, N, ...]
        self._rots = torch.stack(rots_list).unsqueeze(0).to(device)       # [1,6,3,3]
        self._trans = torch.stack(trans_list).unsqueeze(0).to(device)      # [1,6,3]
        self._intrins = torch.stack(intrins_list).unsqueeze(0).to(device)  # [1,6,3,3]
        self._post_rots = torch.stack(post_rots_list).unsqueeze(0).to(device)   # [1,6,3,3]
        self._post_trans = torch.stack(post_trans_list).unsqueeze(0).to(device)  # [1,6,3]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, 6, 3, H, W] — normalised surround-camera images.
        Returns:
            [B, 10, 200, 200] — BEV class logits (channel 0 = vehicle).
        """
        if self._rots is None:
            raise RuntimeError(
                "LSSWrapper.set_calibration(sample) must be called before forward(). "
                "The LSS model requires camera calibration matrices."
            )

        B = images.shape[0]

        # LSS was trained at 128x352 — resize input if needed.
        _, N, C, H, W = images.shape
        target_h, target_w = DEFAULT_DATA_AUG_CONF['final_dim']
        if H != target_h or W != target_w:
            # Reshape to [B*N, C, H, W], resize, reshape back.
            imgs_flat = images.reshape(B * N, C, H, W)
            imgs_flat = F.interpolate(
                imgs_flat, size=(target_h, target_w),
                mode='bilinear', align_corners=False,
            )
            images = imgs_flat.reshape(B, N, C, target_h, target_w)

        # Expand calibration to match batch size if needed.
        rots = self._rots.expand(B, -1, -1, -1)
        trans = self._trans.expand(B, -1, -1)
        intrins = self._intrins.expand(B, -1, -1, -1)
        post_rots = self._post_rots.expand(B, -1, -1, -1)
        post_trans = self._post_trans.expand(B, -1, -1)

        # Run LSS: output is [B, 1, 200, 200] (vehicle segmentation logits).
        logits_1ch = self.lss(images, rots, trans, intrins, post_rots, post_trans)

        # Align axes with the visualization convention.
        # LSS output grid: dim2 index 0 = ego X = -50m (backward),
        #                   dim3 index 0 = ego Y = -50m (rightward).
        # Viz expects:      row 0 = top = forward (+50m),
        #                   col 0 = left = leftward (+50m).
        # Flip both spatial dims so forward is at top and left is at left.
        logits_1ch = logits_1ch.flip(dims=[2, 3])

        # Pad to [B, 10, 200, 200] for compatibility with existing pipeline.
        _, _, bev_h, bev_w = logits_1ch.shape
        padded = torch.full(
            (B, NUM_CLASSES_PAD, bev_h, bev_w),
            -100.0,
            device=logits_1ch.device,
            dtype=logits_1ch.dtype,
        )
        padded[:, 0:1, :, :] = logits_1ch  # Channel 0 = vehicle/car

        return padded


# --------------------------------------------------------------------------- #
#  Factory function
# --------------------------------------------------------------------------- #

def create_lss_model(
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
) -> Optional[LSSWrapper]:
    """Create an LSS model with pretrained weights.

    Returns LSSWrapper on success, or None if loading fails.
    """
    try:
        lss = LiftSplatShoot(
            grid_conf=DEFAULT_GRID_CONF,
            data_aug_conf=DEFAULT_DATA_AUG_CONF,
            outC=1,  # binary vehicle segmentation
        )

        if checkpoint_path is not None:
            logger.info("Loading LSS checkpoint from %s", checkpoint_path)
            state_dict = torch.load(
                checkpoint_path, map_location=device, weights_only=False,
            )
            # The official checkpoint may have keys like 'model_state_dict'
            # or be a raw state dict.
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            lss.load_state_dict(state_dict, strict=False)
            logger.info("LSS checkpoint loaded successfully.")

        wrapper = LSSWrapper(lss)
        wrapper = wrapper.to(device)
        wrapper.eval()
        return wrapper

    except Exception as exc:
        logger.warning("Failed to create LSS model: %s", exc)
        return None
