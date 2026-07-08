"""VLM model runner with pytorch-grad-cam integration for token→image attribution."""

import logging
import math
import os
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
# Local checkpoint path — used when available for offline support
LOCAL_CHECKPOINT = os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints', 'smolvlm-256m')

# All CAM methods available from pytorch-grad-cam (14 classes)
CAM_METHODS = {
    'gradcam': 'GradCAM',
    'hirescam': 'HiResCAM',
    'gradcampp': 'GradCAMPlusPlus',
    'xgradcam': 'XGradCAM',
    'eigencam': 'EigenCAM',
    'eigengradcam': 'EigenGradCAM',
    'layercam': 'LayerCAM',
    'scorecam': 'ScoreCAM',
    'ablationcam': 'AblationCAM',
    'finercam': 'FinerCAM',
    'shapleycam': 'ShapleyCAM',
    'kpcacam': 'KPCA_CAM',
    'gradcam_ew': 'GradCAMElementWise',
    'randomcam': 'RandomCAM',
}


def _get_cam_class(method: str):
    """Import and return the CAM class for the given method key."""
    import pytorch_grad_cam
    mapping = {
        'gradcam': pytorch_grad_cam.GradCAM,
        'hirescam': pytorch_grad_cam.HiResCAM,
        'gradcampp': pytorch_grad_cam.GradCAMPlusPlus,
        'xgradcam': pytorch_grad_cam.XGradCAM,
        'eigencam': pytorch_grad_cam.EigenCAM,
        'eigengradcam': pytorch_grad_cam.EigenGradCAM,
        'layercam': pytorch_grad_cam.LayerCAM,
        'scorecam': pytorch_grad_cam.ScoreCAM,
        'ablationcam': pytorch_grad_cam.AblationCAM,
        'finercam': pytorch_grad_cam.FinerCAM,
        'shapleycam': pytorch_grad_cam.ShapleyCAM,
        'kpcacam': pytorch_grad_cam.KPCA_CAM,
        'gradcam_ew': pytorch_grad_cam.GradCAMElementWise,
        'randomcam': pytorch_grad_cam.RandomCAM,
    }
    return mapping.get(method, pytorch_grad_cam.GradCAM)


class VLMRunner:
    """Manages SmolVLM loading, text generation, and CAM-based attribution."""

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cpu"
        # Cached state from last generate() call
        self._input_ids = None
        self._attention_mask = None
        self._pixel_values = None
        self._generated_ids = None
        self._full_ids = None  # prompt + generated
        self._tokens = None
        self._image = None
        self._image_grid_shape = None  # (H_patches, W_patches)
        # Word-level attention cache (computed once after generate)
        self._words = None       # list of {text, token_indices, strength, heatmap}
        self._word_heatmaps = None  # list of [H, W] numpy arrays

    @property
    def loaded(self):
        return self.model is not None

    def load(self, device: str = "cpu"):
        """Load SmolVLM-256M-Instruct with eager attention (required for grad-cam).

        On macOS Apple Silicon, use device='mps' for ~8-10x speedup over CPU.
        """
        # Prefer local checkpoint for model weights (avoids re-download)
        model_path = LOCAL_CHECKPOINT if os.path.isdir(LOCAL_CHECKPOINT) else MODEL_ID
        logger.info("Loading VLM weights from: %s on %s", model_path, device)
        self.device = device

        # SmolVLM-256M is built on Idefics3. AutoProcessor auto-detection fails
        # because preprocessor_config.json (both local and on Hub) lacks the
        # image_processor_type key required by newer transformers.
        # Load Idefics3Processor directly to bypass auto-detection entirely.
        from transformers import Idefics3Processor
        processor_source = model_path if os.path.isdir(model_path) else MODEL_ID
        logger.info("Loading Idefics3Processor from: %s", processor_source)
        self.processor = Idefics3Processor.from_pretrained(processor_source)

        # Use bfloat16 on MPS/CUDA for speed+memory, float32 on CPU
        dtype = torch.bfloat16 if device != "cpu" else torch.float32
        # Use AutoModel to respect model_type in config.json:
        # - Local checkpoint has model_type="smolvlm" → SmolVLMForConditionalGeneration
        # - Hub snapshot may have model_type="idefics3" → Idefics3ForConditionalGeneration
        # Both are architecturally identical but have different layer name prefixes.
        # Hardcoding SmolVLM class against an idefics3 config causes weight misloading.
        from transformers import AutoModelForVision2Seq
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            _attn_implementation="eager",  # flash_attention_2 is CUDA-only
            torch_dtype=dtype,
        ).to(device).eval()
        nparams = sum(p.numel() for p in self.model.parameters())
        logger.info("VLM loaded: class=%s  model_type=%s  params=%s  dtype=%s",
                     type(self.model).__name__,
                     getattr(self.model.config, 'model_type', '?'),
                     f"{nparams:,}", dtype)

    def generate(self, image: Image.Image, prompt: str = "Describe this driving scene.",
                 max_new_tokens: int = 128) -> dict:
        """Generate text from an image and cache state for attribution.

        Returns:
            dict with keys:
                text: full generated text string
                tokens: list of {text: str, index: int} for each generated token
        """
        self._image = image.copy()

        # Build chat messages and tokenize
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text_prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.processor(
            text=text_prompt,
            images=[image],
            return_tensors="pt",
            do_image_splitting=False,  # single 512x512 patch = 64 tokens (vs 13 patches = 832 tokens)
        ).to(self.device)

        self._input_ids = inputs["input_ids"]
        self._attention_mask = inputs["attention_mask"]
        self._pixel_values = inputs["pixel_values"]

        # Infer image patch grid shape from pixel_values and model config
        self._infer_grid_shape()

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # The output includes the prompt tokens; extract only generated part
        prompt_len = self._input_ids.shape[1]
        self._generated_ids = output_ids[0, prompt_len:]
        self._full_ids = output_ids[0]  # full sequence for teacher forcing

        # Decode tokens individually
        tokens = []
        for i, tid in enumerate(self._generated_ids):
            token_text = self.processor.tokenizer.decode(
                [tid.item()], skip_special_tokens=False
            )
            # Skip special tokens like <end_of_turn>, </s>, etc.
            if tid.item() in self.processor.tokenizer.all_special_ids:
                continue
            tokens.append({"text": token_text, "index": i})

        full_text = self.processor.tokenizer.decode(
            self._generated_ids, skip_special_tokens=True
        )

        self._tokens = tokens
        logger.info("VLM generated %d tokens: %s...", len(tokens),
                     full_text[:80])
        return {"text": full_text, "tokens": tokens}

    def _infer_grid_shape(self):
        """Infer the spatial grid shape of vision encoder output patches."""
        # SmolVLM's vision encoder (SigLIP) processes image into patches
        # The processor resizes the image and splits into patches
        # We need to figure out H_patches x W_patches from the pixel_values shape
        if self._pixel_values is None:
            self._image_grid_shape = (16, 16)
            return

        # pixel_values shape: [B, C, H, W] or [B, num_patches, C, patch_H, patch_W]
        pv = self._pixel_values
        if pv.ndim == 4:
            # Standard [B, C, H, W]
            _, _, h, w = pv.shape
            # SmolVLM uses SigLIP with patch_size typically 14 or 16
            try:
                patch_size = self.model.config.vision_config.patch_size
            except AttributeError:
                patch_size = 14
            gh = h // patch_size
            gw = w // patch_size
            self._image_grid_shape = (gh, gw)
        else:
            # Fallback
            self._image_grid_shape = (16, 16)

        logger.info("Vision patch grid: %s", self._image_grid_shape)

    def get_cam_heatmap(self, token_index: int, method: str = "attention") -> np.ndarray:
        """Compute attribution heatmap for a generated token over the image.

        Args:
            token_index: index into self._tokens (the generated token list).
            method: 'attention' (LLM self-attention, fast, default),
                    or a CAM method key ('gradcam', 'eigencam', etc.).

        Returns:
            [H, W] numpy array in [0, 1] at the original image resolution.
        """
        if self._generated_ids is None:
            raise RuntimeError("Call generate() first")

        # Dispatch: attention-based (fast) vs gradient-based CAM
        if method == "attention":
            return self._get_attention_heatmap(token_index)
        else:
            return self._get_gradcam_heatmap(token_index, method)

    def _find_image_positions(self, input_ids_flat):
        """Find which positions in the input sequence are image tokens."""
        from collections import Counter
        id_counts = Counter(input_ids_flat)
        # Image tokens appear 64 times (image_seq_len=64)
        candidates = [(tid, count) for tid, count in id_counts.most_common(10)
                      if count >= 60]
        if candidates:
            image_token_id = candidates[0][0]
            return [i for i, tid in enumerate(input_ids_flat) if tid == image_token_id]
        logger.warning("Could not identify image token positions, using fallback")
        return list(range(5, 69))

    def _attn_to_heatmap(self, attn_flat: np.ndarray) -> np.ndarray:
        """Convert flat [64] attention values → full-resolution [H, W] heatmap in [0,1]."""
        grid_size = int(math.sqrt(len(attn_flat)))
        attn_grid = attn_flat[:grid_size * grid_size].reshape(grid_size, grid_size)
        vmin, vmax = attn_grid.min(), attn_grid.max()
        if vmax > vmin:
            attn_grid = (attn_grid - vmin) / (vmax - vmin)
        else:
            attn_grid = np.zeros_like(attn_grid)
        img_w, img_h = self._image.size
        hm_pil = Image.fromarray((attn_grid * 255).astype(np.uint8), mode='L')
        hm_pil = hm_pil.resize((img_w, img_h), Image.BILINEAR)
        return np.array(hm_pil).astype(np.float32) / 255.0

    def _group_tokens_into_words(self):
        """Group subword tokens into words based on leading space.

        SmolVLM tokens that start a new word begin with ' ' (space).
        E.g. [' The', ' image', ' dep', 'icts'] → ['The', 'image', 'depicts']
        """
        words = []
        current_word_text = ""
        current_indices = []
        for tok in self._tokens:
            text = tok["text"]
            idx = tok["index"]  # index into _generated_ids
            if text.startswith(' ') and current_indices:
                # New word starts — flush previous word
                words.append({
                    "text": current_word_text.strip(),
                    "token_indices": list(current_indices),
                })
                current_word_text = text
                current_indices = [idx]
            else:
                current_word_text += text
                current_indices.append(idx)
        # Flush last word
        if current_indices:
            words.append({
                "text": current_word_text.strip(),
                "token_indices": list(current_indices),
            })
        return words

    def compute_word_attentions(self, method: str = "avg"):
        """Compute attention for ALL generated tokens in ONE forward pass,
        then aggregate subwords into words.

        Args:
            method: 'avg' (all-layers average, default — recommended) or
                    'rollout' (attention rollout — demonstrates attention sink problem).

        Sets self._words (list of {text, token_indices, strength}) and
        self._word_heatmaps (list of [H, W] numpy arrays).
        """
        if self._generated_ids is None:
            raise RuntimeError("Call generate() first")

        prompt_len = self._input_ids.shape[1]
        num_generated = len(self._generated_ids)

        # Build teacher-forced input with ALL generated tokens
        full_input_ids = torch.cat([
            self._input_ids,
            self._generated_ids.unsqueeze(0)
        ], dim=1)
        full_attention_mask = torch.ones_like(full_input_ids)
        seq_len = full_input_ids.shape[1]

        logger.info("Computing word attentions (%s): %d tokens in 1 forward pass",
                     method, num_generated)
        self.model.to("cpu").float()
        try:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=full_input_ids.to("cpu"),
                    attention_mask=full_attention_mask.to("cpu"),
                    pixel_values=self._pixel_values.to("cpu").float(),
                    output_attentions=True,
                )

            image_positions = self._find_image_positions(full_input_ids[0].tolist())

            # outputs.attentions: tuple of num_layers, each [B, heads, S, S]
            # Average heads within each layer first
            all_layers = torch.stack([layer_attn[0].mean(dim=0)  # [S, S]
                                      for layer_attn in outputs.attentions])  # [L, S, S]

            if method == "rollout":
                # Attention Rollout (Abnar & Zuidema 2020):
                # Multiply attention matrices across layers with residual connections.
                # Known issue: collapses to attention sinks (BOS) in deep models (30+ layers).
                # Included as educational demonstration of the sink phenomenon.
                rollout = torch.eye(seq_len)
                for layer_attn in all_layers:
                    # Add residual connection: 0.5 * A + 0.5 * I
                    attn_aug = 0.5 * layer_attn + 0.5 * torch.eye(seq_len)
                    # Re-normalize rows to sum to 1
                    attn_aug = attn_aug / attn_aug.sum(dim=-1, keepdim=True)
                    rollout = attn_aug @ rollout
                final_attn = rollout  # [S, S]
            else:
                # All-layers average (recommended):
                # Simple mean across layers. Each layer votes independently.
                # Noise cancels out; consistent spatial signal survives.
                final_attn = all_layers.mean(dim=0)  # [S, S]

            # Extract attention to image patches for EVERY generated token
            all_token_attn = {}  # gen_idx → [64] numpy array
            for tok in self._tokens:
                gen_idx = tok["index"]
                target_pos = prompt_len + gen_idx
                attn_to_images = final_attn[target_pos, image_positions].numpy()  # [64]
                all_token_attn[gen_idx] = attn_to_images

        finally:
            self.model.to(self.device)
            if self.device != "cpu":
                self.model.to(torch.bfloat16)
            if self.device == "mps":
                torch.mps.empty_cache()

        # Group tokens into words
        words = self._group_tokens_into_words()

        # Aggregate subwords → words with MAX (Inseq default)
        # For each word, take element-wise max across its subword attention maps
        word_heatmaps = []
        for word in words:
            subword_attns = [all_token_attn[idx] for idx in word["token_indices"]
                             if idx in all_token_attn]
            if not subword_attns:
                word["strength"] = 0.0
                word_heatmaps.append(np.zeros_like(next(iter(all_token_attn.values()))))
                continue
            # Max across subwords (element-wise) — picks strongest signal per patch
            combined = np.maximum.reduce(subword_attns)  # [64]
            # Strength = peak attention value (how focused this word is)
            word["strength"] = float(combined.max())
            word_heatmaps.append(combined)

        # Normalize strengths to [0, 1] across all words
        max_strength = max(w["strength"] for w in words) if words else 1.0
        if max_strength > 0:
            for w in words:
                w["strength"] = w["strength"] / max_strength

        # Convert raw [64] attention arrays to full-resolution heatmaps
        self._word_heatmaps = [self._attn_to_heatmap(attn) for attn in word_heatmaps]
        self._words = words

        logger.info("Computed %d word attentions via %s (from %d tokens)",
                     len(words), method, len(self._tokens))
        return words

    def get_word_heatmap(self, word_index: int) -> np.ndarray:
        """Get precomputed heatmap for a word. No forward pass needed."""
        if self._word_heatmaps is None:
            raise RuntimeError("Call compute_word_attentions() first")
        return self._word_heatmaps[word_index]

    def _get_attention_heatmap(self, token_index: int) -> np.ndarray:
        """All-layers average attention from generated token to image patches.

        Averages attention across all layers and heads. This produces
        token-specific distributions (unlike rollout which collapses to
        attention sinks after matrix multiplication).
        """
        tok = self._tokens[token_index]
        gen_idx = tok["index"]
        prompt_len = self._input_ids.shape[1]

        full_input_ids = torch.cat([
            self._input_ids,
            self._generated_ids[:gen_idx + 1].unsqueeze(0)
        ], dim=1)
        full_attention_mask = torch.ones_like(full_input_ids)

        logger.info("Attention for token[%d]='%s'", token_index, tok["text"])
        self.model.to("cpu").float()
        try:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=full_input_ids.to("cpu"),
                    attention_mask=full_attention_mask.to("cpu"),
                    pixel_values=self._pixel_values.to("cpu").float(),
                    output_attentions=True,
                )

            image_positions = self._find_image_positions(full_input_ids[0].tolist())
            target_pos = prompt_len + gen_idx

            # All-layers average: avg across layers and heads
            all_layers = torch.stack([layer_attn[0].mean(dim=0)
                                      for layer_attn in outputs.attentions])  # [L, S, S]
            avg_attn = all_layers.mean(dim=0)  # [S, S]
            attn_to_images = avg_attn[target_pos, image_positions].numpy()  # [64]

        finally:
            self.model.to(self.device)
            if self.device != "cpu":
                self.model.to(torch.bfloat16)
            if self.device == "mps":
                torch.mps.empty_cache()

        return self._attn_to_heatmap(attn_to_images)

    def _get_gradcam_heatmap(self, token_index: int, method: str = "gradcam") -> np.ndarray:
        """Compute GradCAM on the connector output (vision→LLM projection).

        Uses the connector's modality_projection as target layer — this is where
        vision features get projected into LLM space, so gradients are meaningful
        for per-token attribution.
        """
        from .wrapper import SmolVLMForCAM
        from .targets import VLMTokenTarget

        tok = self._tokens[token_index]
        gen_idx = tok["index"]
        token_id = self._generated_ids[gen_idx].item()
        prompt_len = self._input_ids.shape[1]

        full_input_ids = torch.cat([
            self._input_ids,
            self._generated_ids[:gen_idx + 1].unsqueeze(0)
        ], dim=1)
        full_attention_mask = torch.ones_like(full_input_ids)
        target_position = prompt_len + gen_idx - 1

        wrapper = SmolVLMForCAM(
            model=self.model,
            processor=self.processor,
            input_ids=full_input_ids,
            attention_mask=full_attention_mask,
            pixel_values_shape=self._pixel_values.shape,
        )

        # Target layer: connector output (vision→LLM projection)
        # This is where per-token gradients are most meaningful
        try:
            target_layer = self.model.model.connector.modality_projection.proj
        except AttributeError:
            # Fallback to vision post_layernorm
            target_layer = self.model.model.vision_model.post_layernorm

        # reshape_transform: connector output is [B, 64, 576] → [B, 576, 8, 8]
        def reshape_transform(tensor):
            if tensor.ndim != 3:
                return tensor
            num_tokens = tensor.shape[1]
            side = int(math.sqrt(num_tokens))
            if side * side != num_tokens:
                # Trim to nearest square
                tensor = tensor[:, :side * side, :]
            result = tensor.reshape(tensor.size(0), side, side, tensor.size(2))
            result = result.permute(0, 3, 1, 2)  # [B, C, H, W]
            return result

        target = VLMTokenTarget(target_position, token_id)

        CAMClass = _get_cam_class(method)
        logger.info("Running %s for token[%d]='%s' (id=%d, pos=%d)",
                     method, token_index, tok["text"], token_id, target_position)

        cam_device = "cpu"
        wrapper = wrapper.to(cam_device).float()
        pv = self._pixel_values
        if pv.ndim == 5:
            pv = pv.squeeze(1)
        cam_input = pv.to(cam_device).float()
        wrapper._input_ids = wrapper._input_ids.to(cam_device)
        wrapper._attention_mask = wrapper._attention_mask.to(cam_device)

        try:
            with CAMClass(model=wrapper, target_layers=[target_layer],
                           reshape_transform=reshape_transform) as cam:
                grayscale_cam = cam(
                    input_tensor=cam_input,
                    targets=[target],
                )
        except Exception:
            self.model.to(self.device)
            del wrapper, cam_input
            if self.device == "mps":
                torch.mps.empty_cache()
            raise
        self.model.to(self.device)
        del wrapper, cam_input
        if self.device == "mps":
            torch.mps.empty_cache()

        heatmap = grayscale_cam[0]

        img_w, img_h = self._image.size
        hm_pil = Image.fromarray((heatmap * 255).astype(np.uint8), mode='L')
        hm_pil = hm_pil.resize((img_w, img_h), Image.BILINEAR)
        heatmap_full = np.array(hm_pil).astype(np.float32) / 255.0

        return heatmap_full
