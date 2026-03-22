"""Compatibility shim for TPVFormer — maps old mmcv v1 APIs to mmcv v2 / mmengine.

TPVFormer was written for mmcv-full 1.x + mmdet 2.x + mmseg 0.x
We have mmcv 2.x + mmdet 3.x + mmseg 1.x + mmengine 0.x

This module patches the old import paths so TPVFormer code can import unchanged.
Must be called BEFORE importing any TPVFormer modules.
"""

import sys
import types
import logging

logger = logging.getLogger(__name__)


def patch_mmcv_imports():
    """Monkey-patch old mmcv v1 import paths to work with mmcv v2 + mmengine."""

    import mmcv
    import mmengine

    # ── mmcv.ConfigDict → mmengine.config.ConfigDict ──
    from mmengine.config import ConfigDict
    mmcv.ConfigDict = ConfigDict

    # ── mmcv.runner → mmengine ──
    # Create a fake mmcv.runner module
    if 'mmcv.runner' not in sys.modules:
        runner_mod = types.ModuleType('mmcv.runner')
        sys.modules['mmcv.runner'] = runner_mod
    else:
        runner_mod = sys.modules['mmcv.runner']

    # BaseModule
    from mmengine.model import BaseModule
    runner_mod.BaseModule = BaseModule

    # force_fp32 / auto_fp16 — these are no longer needed in mmcv v2, make them no-ops
    def _noop_decorator(*args, **kwargs):
        """No-op decorator replacing force_fp32/auto_fp16."""
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def wrapper(fn):
            return fn
        return wrapper

    runner_mod.force_fp32 = _noop_decorator
    runner_mod.auto_fp16 = _noop_decorator

    # mmcv.runner.base_module
    if 'mmcv.runner.base_module' not in sys.modules:
        base_mod = types.ModuleType('mmcv.runner.base_module')
        sys.modules['mmcv.runner.base_module'] = base_mod
    else:
        base_mod = sys.modules['mmcv.runner.base_module']

    base_mod.BaseModule = BaseModule

    from torch.nn import ModuleList
    base_mod.ModuleList = ModuleList

    # ── mmcv.cnn patches ──
    # xavier_init, constant_init
    from mmengine.model import xavier_init, constant_init
    import mmcv.cnn as mmcv_cnn
    mmcv_cnn.xavier_init = xavier_init
    mmcv_cnn.constant_init = constant_init

    # ── mmcv.cnn.bricks.registry → mmengine.registry ──
    # These registries moved
    try:
        from mmcv.cnn.bricks.registry import ATTENTION
    except ImportError:
        from mmengine import Registry
        ATTENTION = Registry('attention')
        if 'mmcv.cnn.bricks.registry' not in sys.modules:
            reg_mod = types.ModuleType('mmcv.cnn.bricks.registry')
            sys.modules['mmcv.cnn.bricks.registry'] = reg_mod
        else:
            reg_mod = sys.modules['mmcv.cnn.bricks.registry']
        reg_mod.ATTENTION = ATTENTION

    try:
        from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER, TRANSFORMER_LAYER_SEQUENCE
    except ImportError:
        from mmengine import Registry
        TRANSFORMER_LAYER = Registry('transformer_layer')
        TRANSFORMER_LAYER_SEQUENCE = Registry('transformer_layer_sequence')
        reg_mod = sys.modules.get('mmcv.cnn.bricks.registry')
        if reg_mod:
            reg_mod.TRANSFORMER_LAYER = TRANSFORMER_LAYER
            reg_mod.TRANSFORMER_LAYER_SEQUENCE = TRANSFORMER_LAYER_SEQUENCE

    # ── mmcv.cnn.bricks.transformer ──
    # build_attention, build_feedforward_network, build_positional_encoding
    try:
        from mmcv.cnn.bricks.transformer import build_attention
    except ImportError:
        pass  # May need manual patching

    # ── mmcv.utils ──
    try:
        from mmcv.utils import ext_loader
    except ImportError:
        # Create a shim
        if 'mmcv.utils' not in sys.modules:
            utils_mod = types.ModuleType('mmcv.utils')
            sys.modules['mmcv.utils'] = utils_mod
        else:
            utils_mod = sys.modules['mmcv.utils']

        class _ExtLoader:
            @staticmethod
            def load_ext(name, funcs):
                return types.ModuleType(name)
        utils_mod.ext_loader = _ExtLoader()

    try:
        from mmcv.utils import TORCH_VERSION, digit_version
    except ImportError:
        import torch
        utils_mod = sys.modules.get('mmcv.utils', types.ModuleType('mmcv.utils'))
        utils_mod.TORCH_VERSION = torch.__version__
        utils_mod.digit_version = lambda v: tuple(int(x) for x in v.split('.')[:3])
        sys.modules['mmcv.utils'] = utils_mod

    # ── mmseg.models.HEADS registry ──
    try:
        from mmseg.registry import MODELS as MMSEG_MODELS
        import mmseg.models as mmseg_models_mod
        if not hasattr(mmseg_models_mod, 'HEADS'):
            mmseg_models_mod.HEADS = MMSEG_MODELS
    except Exception:
        pass

    # ── mmseg.models.SEGMENTORS ──
    try:
        from mmseg.registry import MODELS as MMSEG_MODELS
        import mmseg.models as mmseg_models_mod
        if not hasattr(mmseg_models_mod, 'SEGMENTORS'):
            mmseg_models_mod.SEGMENTORS = MMSEG_MODELS
        if not hasattr(mmseg_models_mod, 'builder'):
            # Create a fake builder
            class _Builder:
                @staticmethod
                def build_backbone(cfg):
                    from mmdet.registry import MODELS
                    return MODELS.build(cfg)
                @staticmethod
                def build_neck(cfg):
                    from mmdet.registry import MODELS
                    return MODELS.build(cfg)
            mmseg_models_mod.builder = _Builder()
    except Exception:
        pass

    # ── mmdet.models.utils.positional_encoding → mmdet.models.layers ──
    try:
        from mmdet.models.utils.positional_encoding import LearnedPositionalEncoding
    except (ImportError, ModuleNotFoundError):
        try:
            from mmdet.models.layers.positional_encoding import LearnedPositionalEncoding
            # Create the old module path so TPVFormer can import from it
            import mmdet.models
            if not hasattr(mmdet.models, 'utils'):
                mmdet.models.utils = types.ModuleType('mmdet.models.utils')
                sys.modules['mmdet.models.utils'] = mmdet.models.utils
            pos_mod = types.ModuleType('mmdet.models.utils.positional_encoding')
            pos_mod.LearnedPositionalEncoding = LearnedPositionalEncoding
            sys.modules['mmdet.models.utils.positional_encoding'] = pos_mod
            mmdet.models.utils.positional_encoding = pos_mod
        except ImportError:
            pass

    logger.info("Applied mmcv v1→v2 compatibility patches for TPVFormer")


# Auto-apply on import
patch_mmcv_imports()
