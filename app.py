"""BEV Attribution Debug Tool — FastAPI + custom HTML frontend."""

import os, base64, io, json, logging, time, traceback, collections
import numpy as np
import torch
from PIL import Image

# Ensure CWD is the project root (where this file lives) so relative paths work
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.responses import StreamingResponse
from pydantic import BaseModel

# ── In-memory log buffer for the frontend log viewer ─────────────────────────
LOG_BUFFER = collections.deque(maxlen=500)

class _FrontendLogHandler(logging.Handler):
    """Captures log records into a ring buffer for SSE streaming."""
    def emit(self, record):
        try:
            msg = self.format(record)
            LOG_BUFFER.append(msg)
        except Exception:
            pass

_log_handler = _FrontendLogHandler()
_log_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-5s %(name)s — %(message)s', datefmt='%H:%M:%S'))
logging.root.addHandler(_log_handler)
logging.root.setLevel(logging.INFO)
logger = logging.getLogger("app")

# ── Project imports ──────────────────────────────────────────────────────────
_pipeline_ok = _attribution_ok = False
try:
    from pipeline.data import load_sample, gt_boxes_to_bev, find_first_available_scene
    from pipeline.wrapper import infer, forward_fn, make_captum_forward
    _pipeline_ok = True
except Exception as e:
    logging.getLogger("app").warning("Pipeline import failed: %s", e)

_backends_ok = False
try:
    from pipeline.backends import BACKENDS, get_backend, list_backends
    _backends_ok = True
except Exception as e:
    logging.getLogger("app").warning("Backends import failed: %s", e)
try:
    from attribution import attribute
    _attribution_ok = True
except Exception: pass
from viz.bev import render_bev, render_occupancy_bev, CLASS_NAMES, BEV_IMAGE_SIZE

CAMERA_NAMES = ['CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']
ATTR_MAP = {'GradCAM':'gradcam','Integrated Gradients':'ig','Attention':'attention','Occlusion':'occlusion'}
GRID_RANGE, RESOLUTION = 51.2, 0.512
GRID_CELLS = int(2*GRID_RANGE/RESOLUTION)

def _get_repr_types():
    """Return the single available backend (LSS)."""
    return [{'id': 'lss', 'label': 'LSS (Lift-Splat-Shoot)', 'available': True, 'repr_type': 'bev_seg'}]

_st = dict(model=None, sample=None, bev_grid=None, heatmaps=None,
           backend=None, backend_name='lss', raw_output=None)
_attr_cache = {}

# ── VLM state ────────────────────────────────────────────────────────────────
_vlm_ok = False
try:
    from pipeline.vlm import VLMRunner
    _vlm_ok = True
except Exception as e:
    logging.getLogger("app").warning("VLM import failed: %s\n%s", e, traceback.format_exc())

_vlm_st = dict(runner=None)
_vlm_attr_cache = {}

VLM_METHODS = {
    'Attention': 'attention',
    # GradCAM methods kept in backend (pipeline/vlm/model.py) but not exposed
    # in VLM UI — they're not well suited for early-fusion VLM attribution.
    # They'll be used for other tasks (BEV, CNN classifiers) later.
}

def _pil_uri(img, fmt='JPEG', q=82):
    buf = io.BytesIO()
    if fmt=='JPEG': img.convert('RGB').save(buf, format='JPEG', quality=q)
    else: img.save(buf, format='PNG')
    return f'data:{"image/jpeg" if fmt=="JPEG" else "image/png"};base64,{base64.b64encode(buf.getvalue()).decode()}'

def _get_model_class_info(model):
    """Extract class names from backend or model metadata."""
    backend = _st.get('backend')
    if backend is not None:
        return backend.class_names
    if model is None:
        return list(CLASS_NAMES)
    names = getattr(model, 'class_names', None)
    if names:
        return list(names)
    if type(model).__name__ == 'LSSWrapper':
        return ['vehicle']
    nc = getattr(model, 'num_classes', None)
    if nc and nc < len(CLASS_NAMES):
        return list(CLASS_NAMES[:nc])
    return list(CLASS_NAMES)

def _ensure_model(backend_name=None):
    """Load model using the specified backend."""
    if backend_name and _backends_ok:
        if _st.get('backend_name') != backend_name or _st['model'] is None:
            try:
                backend = get_backend(backend_name)
                model = backend.load(device='cpu')
                if model is not None:
                    _st['model'] = model
                    _st['backend'] = backend
                    _st['backend_name'] = backend_name
                    logger.info("Loaded model via %s backend", backend_name)
                    return model
            except Exception as e:
                logger.error("Failed to load backend %s: %s", backend_name, e)

    # Fall back to existing load_model path
    if _st['model'] is None and _pipeline_ok:
        from pipeline.model import load_model
        _st['model'] = load_model()
        _st['backend_name'] = 'lss'
    return _st['model']

# ── FastAPI ──────────────────────────────────────────────────────────────────
from fastapi import Request
from fastapi.exception_handlers import http_exception_handler
from starlette.exceptions import HTTPException as StarletteHTTPException

server = FastAPI()

@server.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception):
    """Convert all unhandled exceptions to JSON so the frontend can display them."""
    tb = traceback.format_exc()
    logger.error("Unhandled exception on %s: %s\n%s", request.url.path, exc, tb)
    return JSONResponse(status_code=500, content={"error": str(exc), "traceback": tb})

class LoadReq(BaseModel):
    scene_idx: int = 0
    sample_idx: int = 0
    backend: str = 'lss'

class BevReq(BaseModel):
    mode: str = 'argmax'
    class_name: str = 'car'

class AttrReq(BaseModel):
    cell_i: int
    cell_j: int
    method: str = 'GradCAM'
    class_name: str = 'car'

@server.get("/", response_class=HTMLResponse)
async def index():
    return FRONTEND_HTML

@server.get("/api/logs")
async def api_logs():
    """Return all buffered log lines as JSON array."""
    return JSONResponse(list(LOG_BUFFER))

@server.get("/api/logs/stream")
async def api_logs_stream():
    """SSE stream of new log lines."""
    import asyncio
    async def gen():
        sent = len(LOG_BUFFER)
        yield "data: " + json.dumps(list(LOG_BUFFER)) + "\n\n"
        while True:
            await asyncio.sleep(0.5)
            cur = len(LOG_BUFFER)
            if cur > sent:
                new_lines = list(LOG_BUFFER)[sent:]
                for line in new_lines:
                    yield "data: " + json.dumps(line) + "\n\n"
                sent = cur
    return StreamingResponse(gen(), media_type="text/event-stream")

@server.post("/api/load-scene")
async def api_load_scene(req: LoadReq):
    _st['heatmaps'] = None; _attr_cache.clear()
    logger.info("Loading scene %d, sample %d…", req.scene_idx, req.sample_idx)
    t0 = time.time(); parts = []
    if _pipeline_ok:
        try:
            scene_idx, sample_idx = req.scene_idx, req.sample_idx
            # If the requested scene/sample has missing images, auto-find one that exists
            try:
                _st['sample'] = load_sample(scene_idx, sample_idx)
            except FileNotFoundError:
                logger.warning("Images missing for scene %d / sample %d — scanning for available data…", scene_idx, sample_idx)
                scene_idx, sample_idx = find_first_available_scene()
                logger.info("Auto-selected scene %d / sample %d", scene_idx, sample_idx)
                _st['sample'] = load_sample(scene_idx, sample_idx)
            parts.append(f"Loaded {time.time()-t0:.2f}s")
            logger.info("Sample loaded in %.2fs", time.time()-t0)
            model = _ensure_model(backend_name=req.backend)
            if model and _st['sample']:
                t1 = time.time()
                backend = _st.get('backend')
                if backend:
                    _st['raw_output'] = backend.get_raw_output(model, _st['sample'])
                    _st['bev_grid'] = backend.get_bev_grid(_st['raw_output'])
                else:
                    _st['bev_grid'] = infer(model, _st['sample'])
                parts.append(f"Inference {time.time()-t1:.2f}s")
                logger.info("Inference done in %.2fs, BEV grid shape: %s", time.time()-t1, _st['bev_grid'].shape)
        except Exception as e:
            traceback.print_exc(); parts.append(f"Error: {e}")
            _st['sample'] = _st['bev_grid'] = None
    else:
        parts.append("Pipeline unavailable"); _st['sample'] = _st['bev_grid'] = None

    sample = _st.get('sample')
    cam_uris, intrinsics, extrinsics, img_sizes = [], [], [], []
    for ci in range(6):
        if sample and 'images' in sample and len(sample['images'])==6:
            cam_uris.append(_pil_uri(sample['images'][ci]))
            img_sizes.append(list(sample['images'][ci].size))
        else:
            cam_uris.append(''); img_sizes.append([1600,900])
        if sample and 'intrinsics' in sample:
            intrinsics.append(sample['intrinsics'][ci].flatten().tolist())
        else:
            intrinsics.append([800,0,800,0,800,450,0,0,1])
        # Use ego_to_camera (not world_to_camera) — BEV points are in ego frame
        if sample and 'ego_to_cameras' in sample:
            extrinsics.append(sample['ego_to_cameras'][ci].flatten().tolist())
        elif sample and 'extrinsics' in sample:
            extrinsics.append(sample['extrinsics'][ci].flatten().tolist())
        else:
            extrinsics.append([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1])

    # Get dynamic class info from loaded model/backend
    model_classes = _get_model_class_info(_st.get('model'))
    backend = _st.get('backend')
    bev_class_colors = None

    # Compute GT bounding boxes in ego-frame BEV coordinates
    bev_gt_boxes = None
    if sample and 'gt_boxes' in sample and 'ego_to_global' in sample:
        try:
            bev_gt_boxes = gt_boxes_to_bev(sample['gt_boxes'], sample['ego_to_global'])
            logger.info("GT boxes: %d annotations → %d BEV boxes", len(sample['gt_boxes']), len(bev_gt_boxes))
        except Exception as e:
            logger.warning("Failed to compute GT BEV boxes: %s", e)
    _st['bev_gt_boxes'] = bev_gt_boxes

    bev_imgs = {}
    bev_imgs_gt = {}  # versions with GT boxes
    for m in ('argmax',):
        bev_imgs[m] = _pil_uri(render_bev(_st['bev_grid'], mode=m, target_class=0,
                                           grid_range=GRID_RANGE, resolution=RESOLUTION,
                                           class_names=model_classes, class_colors=bev_class_colors), fmt='PNG')
        bev_imgs_gt[m] = _pil_uri(render_bev(_st['bev_grid'], mode=m, target_class=0,
                                              gt_boxes=bev_gt_boxes,
                                              grid_range=GRID_RANGE, resolution=RESOLUTION,
                                              class_names=model_classes, class_colors=bev_class_colors), fmt='PNG')
    cell_classes, cell_confs = [], []
    actual_grid_cells = GRID_CELLS
    actual_resolution = RESOLUTION
    if _st['bev_grid'] is not None:
        cell_classes = np.argmax(_st['bev_grid'], axis=0).flatten().tolist()
        cell_confs = [round(float(v),4) for v in np.max(_st['bev_grid'], axis=0).flatten()]
        actual_grid_cells = _st['bev_grid'].shape[-1]  # Use actual grid size
        actual_resolution = 2 * GRID_RANGE / actual_grid_cells  # Compute matching resolution

    # Serialize GT boxes for frontend hit-testing (corners in BEV world coords)
    gt_boxes_json = []
    if bev_gt_boxes:
        for b in bev_gt_boxes:
            gt_boxes_json.append({
                'corners': b['corners'],  # list of 4 (wx, wz) tuples
                'class_idx': b['class_idx'],
                'class_name': b['class_name'],
            })

    return JSONResponse({
        'camera_images': cam_uris, 'intrinsics': intrinsics, 'extrinsics': extrinsics,
        'image_sizes': img_sizes, 'camera_names': list(CAMERA_NAMES),
        'bev_images': bev_imgs,
        'bev_images_gt': bev_imgs_gt,
        'gt_boxes': gt_boxes_json,
        'bev_info': {'grid_range':GRID_RANGE,'resolution':actual_resolution,'grid_cells':actual_grid_cells,
                     'class_names':model_classes,'cell_classes':cell_classes,'cell_confs':cell_confs},
        'repr_types': _get_repr_types(),
        'repr_type': backend.repr_type if backend else 'bev_seg',
        'num_cameras': len(sample.get('images', [])) if sample else 6,
        'status': ' | '.join(parts) or 'Ready',
    })

@server.post("/api/render-bev")
async def api_render_bev(req: BevReq):
    logger.info("Rendering BEV mode=%s class=%s", req.mode, req.class_name)
    model_classes = _get_model_class_info(_st.get('model'))
    ci = model_classes.index(req.class_name) if req.class_name in model_classes else 0
    backend = _st.get('backend')
    bev_class_colors = None
    img = render_bev(_st['bev_grid'], mode=req.mode, target_class=ci,
                     gt_boxes=_st.get('bev_gt_boxes'),
                     grid_range=GRID_RANGE, resolution=RESOLUTION,
                     class_names=model_classes, class_colors=bev_class_colors)
    return JSONResponse({'bev_image': _pil_uri(img, fmt='PNG')})

@server.get("/api/backends")
async def api_backends():
    return JSONResponse(_get_repr_types())

@server.post("/api/attribute")
async def api_attribute(req: AttrReq):
    logger.info("Attribution: method=%s cell=[%d,%d] class=%s", req.method, req.cell_i, req.cell_j, req.class_name)
    mk = ATTR_MAP.get(req.method, 'gradcam')
    model_classes = _get_model_class_info(_st.get('model'))
    ci = model_classes.index(req.class_name) if req.class_name in model_classes else 0
    ck = (req.cell_i, req.cell_j, ci, mk)
    if ck in _attr_cache:
        hm = _attr_cache[ck]
        elapsed = 0
    elif not _attribution_ok:
        return JSONResponse({'error':'Attribution unavailable','heatmaps':[]})
    elif not _st.get('model') or not _st.get('sample'):
        return JSONResponse({'error':'Load scene first','heatmaps':[]})
    else:
        try:
            t0 = time.time()
            hm = attribute(_st['model'], _st['sample'], req.cell_i, req.cell_j, ci, method=mk)
            elapsed = time.time()-t0
            _attr_cache[ck] = hm
        except Exception as e:
            traceback.print_exc()
            return JSONResponse({'error':str(e),'heatmaps':[]})
    _st['heatmaps'] = hm
    uris = []
    for k in range(6):
        h = hm[k] if k < hm.shape[0] else np.zeros((64,64))
        uris.append(_pil_uri(Image.fromarray((np.clip(h,0,1)*255).astype(np.uint8), mode='L'), fmt='PNG'))
    return JSONResponse({'heatmaps':uris,'status':f'{req.method} [{req.cell_i},{req.cell_j}] {req.class_name} {elapsed:.2f}s'})

# ── VLM Endpoints ────────────────────────────────────────────────────────────

NUSCENES_SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'samples')

@server.get("/api/vlm/images")
async def api_vlm_images():
    """List available camera images from data/samples/ (no nuscenes package needed)."""
    cams = ['CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']
    result = {}
    for cam in cams:
        cam_dir = os.path.join(NUSCENES_SAMPLES_DIR, cam)
        if os.path.isdir(cam_dir):
            files = sorted([f for f in os.listdir(cam_dir) if f.endswith('.jpg')])
            result[cam] = files[:50]  # limit to 50 per camera for UI
    return JSONResponse(result)

@server.get("/api/vlm/image/{camera}/{filename}")
async def api_vlm_image(camera: str, filename: str):
    """Serve a camera image directly from data/samples/."""
    import re
    # Sanitize to prevent path traversal
    if not re.match(r'^CAM_[A-Z_]+$', camera) or '..' in filename:
        return JSONResponse({'error': 'Invalid path'}, status_code=400)
    path = os.path.join(NUSCENES_SAMPLES_DIR, camera, filename)
    if not os.path.isfile(path):
        return JSONResponse({'error': 'Image not found'}, status_code=404)
    return StreamingResponse(open(path, 'rb'), media_type='image/jpeg')

class VLMGenerateReq(BaseModel):
    camera: str = 'CAM_FRONT'
    filename: str = ''
    prompt: str = "Describe this driving scene."
    attn_method: str = 'avg'  # 'avg' (all-layers average) or 'rollout'

class VLMAttentionReq(BaseModel):
    token_index: int
    method: str = 'GradCAM'

@server.get("/api/vlm/debug")
async def api_vlm_debug():
    """Diagnostic endpoint — dump everything relevant to VLM loading."""
    info = {}
    try:
        import transformers
        info['transformers_version'] = transformers.__version__
    except: info['transformers_version'] = 'IMPORT FAILED'
    # Check preprocessor_config.json
    import pipeline.vlm.model as vm
    ckpt = os.path.normpath(vm.LOCAL_CHECKPOINT)
    info['local_checkpoint'] = ckpt
    info['checkpoint_exists'] = os.path.isdir(ckpt)
    pp_path = os.path.join(ckpt, 'preprocessor_config.json')
    info['preprocessor_config_exists'] = os.path.isfile(pp_path)
    if os.path.isfile(pp_path):
        info['preprocessor_config'] = json.load(open(pp_path))
    cfg_path = os.path.join(ckpt, 'config.json')
    if os.path.isfile(cfg_path):
        cfg = json.load(open(cfg_path))
        info['config_model_type'] = cfg.get('model_type')
    # Check which image processor classes are registered
    try:
        from transformers.models.auto.image_processing_auto import IMAGE_PROCESSOR_MAPPING_NAMES
        info['registered_image_processors'] = {k: v for k, v in IMAGE_PROCESSOR_MAPPING_NAMES.items() if 'idefics' in k.lower() or 'smol' in k.lower()}
    except: info['registered_image_processors'] = 'UNAVAILABLE'
    # Check if Idefics3ImageProcessor exists
    try:
        from transformers import Idefics3ImageProcessor
        info['Idefics3ImageProcessor'] = 'EXISTS'
    except ImportError:
        info['Idefics3ImageProcessor'] = 'NOT FOUND'
    try:
        from transformers import SmolVLMImageProcessor
        info['SmolVLMImageProcessor'] = 'EXISTS'
    except ImportError:
        info['SmolVLMImageProcessor'] = 'NOT FOUND'
    info['vlm_ok'] = _vlm_ok
    info['checkpoint_files'] = os.listdir(ckpt) if os.path.isdir(ckpt) else []
    return JSONResponse(info)

@server.post("/api/vlm/load")
async def api_vlm_load():
    if not _vlm_ok:
        return JSONResponse({'ok': False, 'error': 'VLM module not available'})
    if _vlm_st['runner'] and _vlm_st['runner'].loaded:
        return JSONResponse({'ok': True, 'status': 'Already loaded'})
    try:
        runner = VLMRunner()
        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        runner.load(device=device)
        _vlm_st['runner'] = runner
        return JSONResponse({'ok': True, 'status': f'Loaded on {device}'})
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({'ok': False, 'error': str(e)})

@server.post("/api/vlm/generate")
async def api_vlm_generate(req: VLMGenerateReq):
    runner = _vlm_st.get('runner')
    if not runner or not runner.loaded:
        return JSONResponse({'error': 'Load VLM first'})
    # Load image directly from disk (no nuscenes package needed)
    if not req.filename:
        return JSONResponse({'error': 'No image selected'})
    img_path = os.path.join(NUSCENES_SAMPLES_DIR, req.camera, req.filename)
    if not os.path.isfile(img_path):
        return JSONResponse({'error': f'Image not found: {req.camera}/{req.filename}'})
    try:
        _vlm_attr_cache.clear()
        image = Image.open(img_path).convert('RGB')
        t0 = time.time()
        result = runner.generate(image, req.prompt)
        gen_elapsed = time.time() - t0
        # Compute word-level attention (one forward pass for all words)
        attn_method = req.attn_method if req.attn_method in ('avg', 'rollout') else 'avg'
        t1 = time.time()
        words = runner.compute_word_attentions(method=attn_method)
        attn_elapsed = time.time() - t1
        _vlm_attr_cache.clear()
        image_uri = _pil_uri(image, fmt='JPEG', q=75)
        # Return words with strength scores (frontend renders color-coded text)
        word_data = [{"text": w["text"], "strength": round(w["strength"], 3)} for w in words]
        return JSONResponse({
            'text': result['text'],
            'tokens': result['tokens'],
            'words': word_data,
            'image_uri': image_uri,
            'status': f'Generated {len(result["tokens"])} tokens, {len(words)} words | '
                       f'gen {gen_elapsed:.1f}s + {attn_method} {attn_elapsed:.1f}s',
        })
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({'error': str(e)})

@server.post("/api/vlm/attention")
async def api_vlm_attention(req: VLMAttentionReq):
    runner = _vlm_st.get('runner')
    if not runner or not runner.loaded:
        return JSONResponse({'error': 'Load VLM first'})
    if runner._tokens is None:
        return JSONResponse({'error': 'Generate text first'})
    method_key = VLM_METHODS.get(req.method, 'gradcam')
    cache_key = (req.token_index, method_key)
    if cache_key in _vlm_attr_cache:
        hm = _vlm_attr_cache[cache_key]
    else:
        try:
            t0 = time.time()
            hm = runner.get_cam_heatmap(req.token_index, method=method_key)
            elapsed = time.time() - t0
            _vlm_attr_cache[cache_key] = hm
            logger.info("VLM CAM %s token[%d] in %.2fs", req.method, req.token_index, elapsed)
        except Exception as e:
            traceback.print_exc()
            return JSONResponse({'error': str(e)})
    # Render heatmap overlay using viz/camera.py
    from viz.camera import render_camera
    image = runner._image
    token_text = runner._tokens[req.token_index]['text'] if req.token_index < len(runner._tokens) else '?'
    overlay = render_camera(image, heatmap=hm, camera_name=f'Token: {token_text.strip()}')
    return JSONResponse({
        'heatmap': _pil_uri(overlay, fmt='JPEG', q=82),
        'token_text': token_text,
    })

class VLMWordReq(BaseModel):
    word_index: int

@server.post("/api/vlm/word-attention")
async def api_vlm_word_attention(req: VLMWordReq):
    runner = _vlm_st.get('runner')
    if not runner or not runner.loaded:
        return JSONResponse({'error': 'Load VLM first'})
    if runner._words is None:
        return JSONResponse({'error': 'Generate text first'})
    if req.word_index >= len(runner._words):
        return JSONResponse({'error': f'Word index {req.word_index} out of range'})
    cache_key = ("word", req.word_index)
    if cache_key in _vlm_attr_cache:
        uri = _vlm_attr_cache[cache_key]
    else:
        from viz.camera import render_camera
        hm = runner.get_word_heatmap(req.word_index)
        word = runner._words[req.word_index]
        overlay = render_camera(runner._image, heatmap=hm, camera_name=f'{word["text"]}')
        uri = _pil_uri(overlay, fmt='JPEG', q=82)
        _vlm_attr_cache[cache_key] = uri
    word = runner._words[req.word_index]
    return JSONResponse({
        'heatmap': uri,
        'word': word['text'],
        'strength': word['strength'],
    })

# ── VLM→BEV Projection Endpoints ─────────────────────────────────────────────

class VLMBevReq(BaseModel):
    prompt: str = "Describe the vehicle closest to the camera in detail."
    camera: str = 'CAM_FRONT'
    filename: str = ''

@server.post("/api/vlm-bev/run")
async def api_vlm_bev_run(req: VLMBevReq):
    """Generate text, compute attention, project to BEV, overlay with LSS."""
    from viz.bev_projection import project_heatmap_to_bev, render_vlm_bev
    from viz.camera import render_camera

    runner = _vlm_st.get('runner')
    if not runner or not runner.loaded:
        if _vlm_ok:
            runner = VLMRunner()
            device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
            runner.load(device=device)
            _vlm_st['runner'] = runner
        else:
            return JSONResponse({'error': 'VLM not available'})

    if not req.filename:
        return JSONResponse({'error': 'No image selected. Pick a camera and image first.'})

    img_path = os.path.join(NUSCENES_SAMPLES_DIR, req.camera, req.filename)
    if not os.path.isfile(img_path):
        return JSONResponse({'error': f'Image not found: {req.camera}/{req.filename}'})

    # Auto-load scene + LSS matching the selected image
    sample = _st.get('sample')
    if _pipeline_ok:
        try:
            from pipeline.data import load_sample_by_filename
            new_sample = load_sample_by_filename(req.camera, req.filename)
            _st['sample'] = new_sample
            model = _ensure_model(backend_name='lss')
            if model and _st['sample']:
                backend = _st.get('backend')
                if backend:
                    _st['raw_output'] = backend.get_raw_output(model, _st['sample'])
                    _st['bev_grid'] = backend.get_bev_grid(_st['raw_output'])
                else:
                    _st['bev_grid'] = infer(model, _st['sample'])
                logger.info("Loaded scene for %s/%s, BEV grid: %s",
                            req.camera, req.filename, _st['bev_grid'].shape)
        except Exception as e:
            logger.warning("Scene load failed: %s", e)

    sample = _st.get('sample')
    has_calib = sample and 'ego_to_cameras' in sample and 'intrinsics' in sample

    try:
        t0 = time.time()
        image = Image.open(img_path).convert('RGB')

        # 1. Generate text + compute word attentions
        runner.generate(image, req.prompt, max_new_tokens=80)
        words = runner.compute_word_attentions(method="avg")

        # 2. Build aggregate attention heatmap — top-5 strongest content words
        scored = [(i, w['strength']) for i, w in enumerate(words)
                  if len(w['text']) >= 3 and w['strength'] > 0.1]
        scored.sort(key=lambda x: x[1], reverse=True)
        top_indices = [i for i, _ in scored[:5]]
        if not top_indices:
            top_indices = list(range(min(5, len(words))))
        content_heatmaps = [runner.get_word_heatmap(i) for i in top_indices]
        agg_heatmap = np.maximum.reduce(content_heatmaps) if content_heatmaps else np.zeros((image.size[1], image.size[0]))

        # 3. Render camera image with attention overlay
        cam_img = render_camera(image, heatmap=agg_heatmap, camera_name=req.camera)

        # 4. Project to BEV (only if calibration available)
        bev_img = None
        if has_calib:
            ci = CAMERA_NAMES.index(req.camera) if req.camera in CAMERA_NAMES else 0
            K = np.array(sample['intrinsics'][ci], dtype=np.float64)
            if K.shape == (9,):
                K = K.reshape(3, 3)
            E = np.array(sample['ego_to_cameras'][ci], dtype=np.float64)
            if E.shape == (16,):
                E = E.reshape(4, 4)

            bev_attn = project_heatmap_to_bev(
                agg_heatmap, K, E,
                grid_range=GRID_RANGE, resolution=RESOLUTION, grid_cells=GRID_CELLS
            )

            # LSS vehicle mask overlay
            lss_mask = None
            bev_grid = _st.get('bev_grid')
            if bev_grid is not None and bev_grid.ndim == 3:
                vehicle_logits = bev_grid[0]
                lss_mask = vehicle_logits > 0.3

            bev_img = render_vlm_bev(bev_attn, lss_bev_mask=lss_mask,
                                      gt_boxes=_st.get('bev_gt_boxes'),
                                      grid_range=GRID_RANGE, resolution=RESOLUTION)

        elapsed = time.time() - t0
        word_list = [{'text': w['text'], 'strength': round(w['strength'], 3)} for w in words]
        gen_text = runner.processor.tokenizer.decode(runner._generated_ids, skip_special_tokens=True)

        result = {
            'camera_image': _pil_uri(cam_img, fmt='JPEG', q=80),
            'text': gen_text,
            'words': word_list,
            'status': f'{len(words)} words, {elapsed:.1f}s',
        }
        if bev_img:
            result['bev_image'] = _pil_uri(bev_img, fmt='PNG')
        else:
            result['bev_warn'] = 'Load a scene in BEV tab first for calibration data'

        return JSONResponse(result)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({'error': str(e)})

class VLMBevWordReq(BaseModel):
    word_index: int
    camera: str = 'CAM_FRONT'

@server.post("/api/vlm-bev/word")
async def api_vlm_bev_word(req: VLMBevWordReq):
    """Project a single word's attention to BEV and camera."""
    from viz.bev_projection import project_heatmap_to_bev, render_vlm_bev
    from viz.camera import render_camera

    runner = _vlm_st.get('runner')
    if not runner or runner._word_heatmaps is None:
        return JSONResponse({'error': 'Run a prompt first'})
    if req.word_index >= len(runner._word_heatmaps):
        return JSONResponse({'error': 'Word index out of range'})

    sample = _st.get('sample')
    heatmap = runner.get_word_heatmap(req.word_index)
    word_text = runner._words[req.word_index]['text']

    # Camera image with this word's attention
    cam_img = render_camera(runner._image, heatmap=heatmap,
                            camera_name=f'"{word_text}"')

    # BEV projection
    bev_img = None
    if sample and 'ego_to_cameras' in sample:
        ci = CAMERA_NAMES.index(req.camera) if req.camera in CAMERA_NAMES else 0
        K = np.array(sample['intrinsics'][ci], dtype=np.float64)
        if K.shape == (9,): K = K.reshape(3, 3)
        E = np.array(sample['ego_to_cameras'][ci], dtype=np.float64)
        if E.shape == (16,): E = E.reshape(4, 4)

        bev_attn = project_heatmap_to_bev(heatmap, K, E,
                                           grid_range=GRID_RANGE, resolution=RESOLUTION,
                                           grid_cells=GRID_CELLS)
        lss_mask = None
        bev_grid = _st.get('bev_grid')
        if bev_grid is not None and bev_grid.ndim == 3:
            lss_mask = bev_grid[0] > 0.3
        bev_img = render_vlm_bev(bev_attn, lss_bev_mask=lss_mask,
                                  gt_boxes=_st.get('bev_gt_boxes'),
                                  grid_range=GRID_RANGE, resolution=RESOLUTION)

    result = {
        'camera_image': _pil_uri(cam_img, fmt='JPEG', q=82),
        'word': word_text,
    }
    if bev_img:
        result['bev_image'] = _pil_uri(bev_img, fmt='PNG')
    return JSONResponse(result)

class VLMBevClickReq(BaseModel):
    x_frac: float  # click x as fraction of image width [0, 1]
    y_frac: float  # click y as fraction of image height [0, 1]
    camera: str = 'CAM_FRONT'

@server.post("/api/vlm-bev/click")
async def api_vlm_bev_click(req: VLMBevClickReq):
    """Project a BEV click to camera pixel and return camera image with crosshair."""
    from viz.camera import render_camera

    sample = _st.get('sample')
    if not sample or 'ego_to_cameras' not in sample:
        return JSONResponse({'error': 'No calibration data'})

    runner = _vlm_st.get('runner')
    if not runner or runner._image is None:
        return JSONResponse({'error': 'Run a VLM prompt first'})

    # Reconstruct BEV crop parameters (must match render_vlm_bev)
    grid_cells = GRID_CELLS  # 200
    half = grid_cells // 2   # 100
    lateral_half = min(70, half)
    col_start = half - lateral_half
    crop_h = half             # rows 0..99
    crop_w = lateral_half * 2 # 140

    # Click fraction → cropped grid cell
    crop_i = req.y_frac * crop_h
    crop_j = req.x_frac * crop_w

    # Cropped cell → full grid cell
    full_i = crop_i
    full_j = crop_j + col_start

    # Full grid cell → BEV world coords
    wx = -GRID_RANGE + (full_j + 0.5) * RESOLUTION
    wz = GRID_RANGE - (full_i + 0.5) * RESOLUTION

    # BEV world → ego frame
    ego_x, ego_y, ego_z = wz, -wx, 0.0

    # Ego → camera pixel
    ci = CAMERA_NAMES.index(req.camera) if req.camera in CAMERA_NAMES else 0
    K = np.array(sample['intrinsics'][ci], dtype=np.float64)
    if K.shape == (9,):
        K = K.reshape(3, 3)
    E = np.array(sample['ego_to_cameras'][ci], dtype=np.float64)
    if E.shape == (16,):
        E = E.reshape(4, 4)

    ego_pt = np.array([ego_x, ego_y, ego_z, 1.0])
    cam_pt = E @ ego_pt
    if cam_pt[2] <= 0:
        return JSONResponse({'error': 'Point behind camera', 'u': -1, 'v': -1})

    px = K @ cam_pt[:3]
    u = px[0] / px[2]
    v = px[1] / px[2]

    # Render camera image with crosshair
    image = runner._image
    cam_img = render_camera(image, heatmap=None, camera_name=req.camera,
                            projection_point=(u, v))

    dist = float(np.sqrt(ego_x**2 + ego_y**2))
    return JSONResponse({
        'camera_image': _pil_uri(cam_img, fmt='JPEG', q=82),
        'u': round(float(u)),
        'v': round(float(v)),
        'ego_x': round(float(ego_x), 1),
        'ego_y': round(float(ego_y), 1),
        'dist_m': round(dist, 1),
    })

# ══════════════════════════════════════════════════════════════════════════════
# FRONTEND
# ══════════════════════════════════════════════════════════════════════════════

FRONTEND_HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BEV Attribution Debug</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{background:#0a0a0a;color:#e0e0e0;font-family:'Consolas','Monaco','Menlo',monospace;overflow:hidden;height:100vh}
#app{display:flex;flex-direction:column;height:100vh;padding:8px;gap:6px}

/* Header */
#hdr{display:flex;align-items:center;justify-content:space-between;padding:6px 14px;background:#111;border:1px solid #1a3a3a;border-radius:6px;flex-shrink:0}
#hdr h1{font-size:15px;color:#5cf;letter-spacing:1px}

/* Controls */
#ctrl{display:flex;gap:14px;align-items:center;flex-wrap:wrap;padding:5px 14px;background:#111;border:1px solid #1a3a3a;border-radius:6px;flex-shrink:0;font-size:11px}
.cg{display:flex;align-items:center;gap:5px}
.cg label{color:#888;white-space:nowrap;font-size:10px}
.cg select{background:#1a1a1a;border:1px solid #333;color:#ccc;padding:3px 6px;border-radius:3px;font:inherit;font-size:11px;cursor:pointer}
.cg select option:disabled{color:#555}
.btn{background:#0a2a2a;border:1px solid #5cf;color:#5cf;padding:4px 14px;border-radius:4px;cursor:pointer;font:inherit;font-size:11px;transition:all .15s;white-space:nowrap}
.btn:hover{background:#1a3a3a}
.btn:disabled{opacity:.4;cursor:default}
.btn.loading{animation:pulse .8s infinite alternate}
@keyframes pulse{from{opacity:.4}to{opacity:1}}

/* Main area */
#main{display:flex;gap:8px;flex:1;min-height:0}

/* BEV panel */
#bev-panel{flex:0 0 42%;display:flex;flex-direction:column;gap:4px;background:#111;border:1px solid #1a3a3a;border-radius:6px;padding:8px;min-width:0}
.ph{display:flex;align-items:center;gap:8px;font-size:12px;flex-shrink:0}
.ph-t{color:#5cf;font-weight:bold;font-size:13px}
.tabs{display:flex;gap:3px;margin-left:auto;align-items:center}
.tab-label{color:#666;font-size:9px;font-weight:bold;text-transform:uppercase;letter-spacing:.5px;margin-right:2px}
.tab{background:#1a1a1a;border:1px solid #333;color:#888;padding:2px 9px;border-radius:3px;cursor:pointer;font:inherit;font-size:10px;transition:all .15s;position:relative}
.tab:hover{border-color:#5cf;color:#5cf}
.tab.on{border-color:#5cf;color:#5cf;background:#0a2a2a}
.tab:disabled{opacity:.3;cursor:not-allowed;border-color:#222;color:#555}
.tab:disabled:hover{border-color:#222;color:#555}
.tab[data-tip]:hover::after{content:attr(data-tip);position:absolute;bottom:calc(100% + 6px);left:50%;transform:translateX(-50%);background:rgba(0,0,0,.95);color:#aaa;padding:4px 8px;border-radius:3px;font-size:9px;white-space:nowrap;border:1px solid #333;z-index:100;pointer-events:none}
#bev-wrap{position:relative;flex:1;min-height:0;overflow:hidden;border-radius:4px}
#bev-c{width:100%;height:100%;object-fit:contain;cursor:crosshair;display:block}
#bev-tip{position:fixed;pointer-events:none;background:rgba(0,0,0,.92);color:#5cf;padding:4px 8px;border-radius:3px;font-size:10px;border:1px solid #2a4a4a;display:none;white-space:nowrap;z-index:9999}
#bev-info{font-size:10px;color:#888;padding:2px 4px;flex-shrink:0;min-height:14px}

/* Camera panel */
#cam-panel{flex:1;display:flex;flex-direction:column;gap:4px;min-width:0}
.cfg-btn{background:none;border:1px solid #333;color:#888;padding:2px 7px;border-radius:3px;cursor:pointer;font-size:13px;transition:all .15s}
.cfg-btn:hover{color:#5cf;border-color:#5cf}
#cam-grid{flex:1;display:grid;grid-template-columns:repeat(3,1fr);grid-template-rows:repeat(2,1fr);gap:5px;min-height:0}
.cc{background:#111;border:1px solid #1a3a3a;border-radius:5px;padding:3px;display:flex;flex-direction:column;overflow:hidden;transition:border-color .15s,opacity .15s;min-height:0}
.cc.vis{border-color:#5cf}
.cc.out{opacity:.35;border-color:#222}
.cc.hid{display:none}
.cc-h{display:flex;justify-content:space-between;font-size:10px;padding:0 3px 2px;flex-shrink:0}
.cc-n{font-weight:bold;text-transform:uppercase}
.cc-p{color:#666;font-size:9px}
.cc-w{flex:1;position:relative;min-height:0;overflow:hidden}
.cc-w canvas{width:100%;height:100%;display:block;border-radius:3px;object-fit:contain}

/* Status */
#status{font-size:11px;color:#5cf;padding:4px 12px;background:#111;border:1px solid #1a3a3a;border-radius:4px;flex-shrink:0;min-height:22px}

/* Config modal */
#cfg-modal{display:none;position:fixed;inset:0;background:rgba(0,0,0,.75);z-index:100;justify-content:center;align-items:center}
#cfg-modal.open{display:flex}
#cfg-inner{background:#111;border:1px solid #1a3a3a;border-radius:8px;padding:16px;width:540px;max-height:80vh;overflow-y:auto;display:flex;flex-direction:column;gap:10px}
.cfg-hdr{display:flex;justify-content:space-between;color:#5cf;font-size:14px;font-weight:bold}
#cfg-tabs{display:flex;gap:3px;flex-wrap:wrap}
.cfg-s{display:flex;flex-direction:column;gap:4px}
.cfg-s label{color:#888;font-size:10px}
.cfg-r{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.cfg-l{color:#5cf;font-size:10px;min-width:18px}
.cfg-r input{width:90px;background:#0a0a0a;border:1px solid #333;color:#ccc;padding:3px 6px;border-radius:3px;font:inherit;font-size:11px}
#cfg-ext{background:#0a0a0a;border:1px solid #333;color:#ccc;padding:6px;border-radius:3px;font:inherit;font-size:10px;resize:vertical;width:100%}
/* Info tooltip */
.info-icon{display:inline-flex;align-items:center;justify-content:center;width:18px;height:18px;border-radius:50%;border:1px solid #555;color:#888;font-size:10px;cursor:pointer;transition:all .15s;font-style:normal;flex-shrink:0}
.info-icon:hover{color:#5cf;border-color:#5cf}
#info-tip{display:none;position:fixed;background:#141414;border:1px solid #2a4a4a;border-radius:6px;padding:10px 14px;color:#ccc;font-size:11px;max-width:380px;z-index:200;line-height:1.5;box-shadow:0 4px 20px rgba(0,0,0,.6)}
#info-tip .it-title{color:#5cf;font-weight:bold;font-size:12px;margin-bottom:4px}
#info-tip .it-body{color:#aaa}

/* Log viewer */
.log-btn{background:none;border:1px solid #333;color:#888;padding:2px 8px;border-radius:3px;cursor:pointer;font-size:11px;transition:all .15s;font-family:inherit}
.log-btn:hover{color:#5cf;border-color:#5cf}
.log-btn.active{color:#5cf;border-color:#5cf}
#log-panel{display:none;position:fixed;bottom:0;left:0;right:0;height:240px;background:#0c0c0c;border-top:1px solid #1a3a3a;z-index:150;flex-direction:column;font-size:11px}
#log-panel.open{display:flex}
#log-hdr{display:flex;align-items:center;justify-content:space-between;padding:4px 12px;background:#111;border-bottom:1px solid #1a3a3a;flex-shrink:0}
#log-hdr span{color:#5cf;font-weight:bold;font-size:12px}
#log-actions{display:flex;gap:6px}
#log-body{flex:1;overflow-y:auto;padding:6px 12px;font-family:'Consolas','Monaco',monospace;font-size:10px;line-height:1.6;color:#8a8a8a}
#log-body .log-line{white-space:pre-wrap;word-break:break-all}
#log-body .log-line.INFO{color:#8a8a8a}
#log-body .log-line.WARNING{color:#e8a838}
#log-body .log-line.ERROR{color:#e85050}
#log-body .log-line.DEBUG{color:#666}
#log-body::-webkit-scrollbar{width:5px}
#log-body::-webkit-scrollbar-track{background:#0c0c0c}
#log-body::-webkit-scrollbar-thumb{background:#333;border-radius:3px}

.cfg-act{display:flex;gap:8px;justify-content:flex-end}

/* Mode tabs */
#mode-tabs{display:flex;gap:4px}
#mode-tabs .tab.on{background:#0a2a2a}

/* VLM panel */
#vlm-panel{display:none;flex:1;gap:8px;min-height:0}
#vlm-panel.active{display:flex}
#vlm-img-wrap{flex:2;position:relative;min-height:0;background:#050505;border-radius:6px;overflow:hidden}
#vlm-img-wrap img{width:100%;height:100%;object-fit:contain;display:block}
#vlm-sidebar{width:360px;display:flex;flex-direction:column;gap:8px;background:#111;border:1px solid #1a3a3a;border-radius:6px;padding:10px;overflow-y:auto;min-height:0}
#vlm-controls{display:flex;gap:6px;align-items:center;flex-wrap:wrap;flex-shrink:0}
#vlm-controls select,#vlm-controls input{background:#1a1a1a;border:1px solid #333;color:#ccc;padding:3px 6px;border-radius:3px;font:inherit;font-size:11px}
#vlm-controls input[type="text"]{flex:1;min-width:120px}
#vlm-prompt-row{display:flex;gap:6px;align-items:center;flex-shrink:0;width:100%}
#vlm-prompt{flex:1;background:#1a1a1a;border:1px solid #333;color:#ccc;padding:4px 8px;border-radius:3px;font:inherit;font-size:11px}
#vlm-text-output{flex:1;overflow-y:auto;line-height:1.8;font-size:13px;padding:4px;min-height:0}
#vlm-text-output .vlm-word{display:inline;padding:2px 4px;cursor:pointer;border-radius:3px;transition:all .15s;user-select:none;margin:1px}
#vlm-text-output .vlm-word:hover{outline:1px solid #5cf}
#vlm-text-output .vlm-word.active{outline:2px solid #5cf;color:#fff}
#vlm-loading{color:#888;font-size:11px;padding:8px 0}
#vlm-status{font-size:10px;color:#888;flex-shrink:0}

/* VLM→BEV panel */
#vb-panel{display:none;flex:1;gap:8px;min-height:0}
#vb-panel.active{display:flex}
#vb-left{flex:1;display:flex;flex-direction:column;gap:4px;min-height:0;background:#111;border:1px solid #1a3a3a;border-radius:6px;padding:8px}
#vb-left .ph-t{color:#5cf;font-weight:bold;font-size:13px}
#vb-bev-wrap{flex:1;position:relative;min-height:0;overflow:hidden;border-radius:4px}
#vb-bev-wrap img{width:100%;height:100%;object-fit:contain;display:block}
#vb-legend{font-size:10px;color:#888;padding:2px 4px;flex-shrink:0}
#vb-right{flex:1;display:flex;flex-direction:column;gap:6px;min-height:0}
#vb-cam-wrap{flex:1;position:relative;min-height:0;background:#050505;border-radius:6px;overflow:hidden}
#vb-cam-wrap img{width:100%;height:100%;object-fit:contain;display:block}
#vb-controls{display:flex;flex-direction:column;gap:6px;background:#111;border:1px solid #1a3a3a;border-radius:6px;padding:10px;flex-shrink:0}
#vb-presets{display:flex;gap:6px}
#vb-prompt-row{display:flex;gap:6px;align-items:center}
#vb-prompt{flex:1;background:#1a1a1a;border:1px solid #333;color:#ccc;padding:4px 8px;border-radius:3px;font:inherit;font-size:11px}
#vb-text-output{max-height:120px;overflow-y:auto;font-size:12px;line-height:1.6;padding:4px}
#vb-text-output .vb-word{display:inline;padding:1px 3px;border-radius:2px}
#vb-status{font-size:10px;color:#888}
</style>
</head><body>
<div id="app">
  <div id="hdr">
    <h1>BEV & VLM ATTRIBUTION DEBUG</h1>
    <div id="mode-tabs">
      <button class="tab on" data-mode="bev">BEV Attribution</button>
      <button class="tab" data-mode="vlm">VLM Reasoning</button>
      <button class="tab" data-mode="vb">VLM→BEV</button>
    </div>
  </div>

  <div id="ctrl">
    <div class="cg"><label>Scene</label><select id="sel-scene"></select></div>
    <div class="cg"><label>Sample</label><select id="sel-sample"></select></div>
    <div class="cg"><label>Class</label><select id="sel-class"><option value="vehicle">vehicle</option></select></div>
    <div class="cg"><label>Method</label><select id="sel-method"></select><i class="info-icon" id="method-info">i</i></div>
    <button class="btn" id="btn-load">Load Scene</button>
    <button class="btn" id="btn-attr">Run Attribution</button>
    <button class="log-btn" id="log-toggle" title="Toggle Log Viewer">⌸ Log</button>
  </div>

  <div id="main">
    <div id="bev-panel">
      <div class="ph">
        <div class="tabs" id="bev-tabs">
          <button class="tab on" data-m="argmax" data-tip="Per-cell class with highest logit, color-coded">Argmax</button>
          <button class="tab" data-m="class_heatmap" data-tip="Heatmap of vehicle confidence across the grid">Heatmap</button>
        </div>
        <button class="tab" id="bev-coverage-toggle" title="Show/hide camera coverage on BEV">Coverage</button>
        <button class="tab" id="bev-gt-toggle" title="Show/hide ground truth 3D bounding boxes">GT Boxes</button>
      </div>
      <div id="bev-wrap"><canvas id="bev-c" width="800" height="800"></canvas><div id="bev-tip"></div></div>
      <div id="bev-info">Hover to inspect · Click to select</div>
    </div>

    <div id="cam-panel">
      <div class="ph">
        <span class="ph-t">Cameras</span>
        <div class="tabs" id="cam-tabs">
          <button class="tab on" data-m="6">6</button>
          <button class="tab" data-m="auto">Auto</button>
          <button class="tab" data-m="1">1</button>
        </div>
        <button class="cfg-btn" id="cfg-open" title="Edit Calibration">⚙</button>
      </div>
      <div id="cam-grid"></div>
    </div>

    <div id="vlm-panel">
      <div id="vlm-img-wrap">
        <img id="vlm-img" src="" alt="Select a camera and generate">
      </div>
      <div id="vlm-sidebar">
        <div id="vlm-controls">
          <label style="color:#888;font-size:10px">Camera</label>
          <select id="vlm-cam-select"></select>
          <label style="color:#888;font-size:10px">Image</label>
          <select id="vlm-file-select" style="max-width:180px" onchange="vlmPreviewImage()"></select>
          <label style="color:#888;font-size:10px">Aggregation</label>
          <select id="vlm-attn-method">
            <option value="avg" selected>All-Layers Avg (recommended)</option>
            <option value="rollout">Attention Rollout (demo: attention sink)</option>
          </select>
        </div>
        <div id="vlm-prompt-row">
          <input id="vlm-prompt" type="text" value="Describe this driving scene." placeholder="Prompt...">
          <button class="btn" id="vlm-generate-btn">Generate</button>
        </div>
        <div id="vlm-loading" style="display:none">Loading model...</div>
        <div id="vlm-text-output"></div>
        <div id="vlm-status"></div>
      </div>
    </div>

    <div id="vb-panel">
      <div id="vb-left">
        <div class="ph"><span class="ph-t">BEV Projection</span></div>
        <div id="vb-bev-wrap"><img id="vb-bev-img" src="" alt="Run a prompt to project attention to BEV"></div>
        <div id="vb-legend">■ VLM attention (turbo) · ○ LSS vehicle detections (cyan) · ▲ ego</div>
      </div>
      <div id="vb-right">
        <div id="vb-cam-wrap"><img id="vb-cam-img" src="" alt="Front camera"></div>
        <div id="vb-controls">
          <div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap">
            <label style="color:#888;font-size:10px">Camera</label>
            <select id="vb-cam-select" style="background:#1a1a1a;border:1px solid #333;color:#ccc;padding:3px 6px;border-radius:3px;font:inherit;font-size:11px"></select>
            <label style="color:#888;font-size:10px">Image</label>
            <select id="vb-file-select" style="max-width:180px;background:#1a1a1a;border:1px solid #333;color:#ccc;padding:3px 6px;border-radius:3px;font:inherit;font-size:11px"></select>
          </div>
          <div id="vb-presets">
            <select id="vb-preset-select" style="background:#1a1a1a;border:1px solid #333;color:#ccc;padding:3px 6px;border-radius:3px;font:inherit;font-size:11px;flex:1">
              <option value="">-- Preset Prompts --</option>
              <option value="Describe the color of each vehicle in this image, from closest to most distant.">Vehicle Colors</option>
              <option value="Describe the vehicle closest to the camera in detail.">Closest Vehicle</option>
              <option value="Describe the most distant vehicle visible ahead.">Farthest Vehicle</option>
              <option value="Describe the ego vehicle's current driving environment.">Driving Environment</option>
              <option value="What are the key objects and events in the driver's field of view?">Key Objects</option>
              <option value="Describe traffic conditions, road structure, and agent behaviors.">Traffic & Agents</option>
            </select>
            <button class="btn" id="vb-btn-preset">Use Preset</button>
          </div>
          <div id="vb-prompt-row">
            <input id="vb-prompt" type="text" value="Describe the color of each vehicle in this image, from closest to most distant." placeholder="Custom prompt...">
            <button class="btn" id="vb-btn-run">Run</button>
          </div>
          <div id="vb-text-output"></div>
          <div id="vb-status"></div>
        </div>
      </div>
    </div>
  </div>

  <div id="status">Ready — load a scene</div>
</div>

<!-- Info tooltip (positioned by JS) -->
<div id="info-tip"></div>

<!-- Log panel (bottom drawer) -->
<div id="log-panel">
  <div id="log-hdr">
    <span>⌸ Server Log</span>
    <div id="log-actions">
      <button class="tab" id="log-clear" title="Clear">Clear</button>
      <button class="tab" id="log-close" title="Close">✕</button>
    </div>
  </div>
  <div id="log-body"></div>
</div>

<!-- Config modal -->
<div id="cfg-modal">
  <div id="cfg-inner">
    <div class="cfg-hdr"><span>Camera Calibration</span><button class="tab" id="cfg-close">✕</button></div>
    <div id="cfg-tabs"></div>
    <div class="cfg-s"><label>Intrinsic (fx, fy, cx, cy)</label>
      <div class="cfg-r">
        <span class="cfg-l">fx</span><input id="cfg-fx" type="number" step="any">
        <span class="cfg-l">fy</span><input id="cfg-fy" type="number" step="any">
        <span class="cfg-l">cx</span><input id="cfg-cx" type="number" step="any">
        <span class="cfg-l">cy</span><input id="cfg-cy" type="number" step="any">
      </div>
    </div>
    <div class="cfg-s"><label>Extrinsic 4×4 (row-major, world→camera)</label>
      <textarea id="cfg-ext" rows="4" spellcheck="false"></textarea>
    </div>
    <div class="cfg-act">
      <button class="btn" id="cfg-revert">↩ Revert Default</button>
      <button class="btn" id="cfg-apply">Apply</button>
    </div>
  </div>
</div>

<script>
"use strict";

// ─── Constants ──────────────────────────────────────────────────────────────
const CN=['CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT'];
const CC=['#00ccff','#66ff33','#ff6600','#ffcc00','#ff33cc','#9966ff'];
const GI=[2,0,1,4,3,5]; // grid order: FL,F,FR,BL,B,BR → data indices
const GN=['FRONT_LEFT','FRONT','FRONT_RIGHT','BACK_LEFT','BACK','BACK_RIGHT'];
let CLASSES=['vehicle'];
const METHODS=['GradCAM','Integrated Gradients','Attention','Occlusion'];

// ─── Populate selects ───────────────────────────────────────────────────────
const selScene=document.getElementById('sel-scene');
const selSample=document.getElementById('sel-sample');
const selClass=document.getElementById('sel-class');
const selMethod=document.getElementById('sel-method');
const selRepr=document.getElementById('sel-repr')||document.createElement('select'); // removed from UI
for(let i=0;i<10;i++){const o=document.createElement('option');o.value=i;o.textContent='Scene '+i;selScene.appendChild(o);}
for(let i=0;i<40;i++){const o=document.createElement('option');o.value=i;o.textContent='Sample '+i;selSample.appendChild(o);}
function populateClasses(names){
  CLASSES=names;selClass.innerHTML='';
  names.forEach(c=>{const o=document.createElement('option');o.value=c;o.textContent=c;selClass.appendChild(o);});
}
populateClasses(CLASSES);
METHODS.forEach(m=>{const o=document.createElement('option');o.value=m;o.textContent=m;selMethod.appendChild(o);});
// Update repr selector when backend sends availability
function updateReprTypes(types){
  if(!types)return;
  const sel=selRepr;
  const curVal=sel.value;
  sel.innerHTML='';
  types.forEach(t=>{
    const o=document.createElement('option');
    o.value=t.id;
    o.textContent=t.label;
    o.disabled=!t.available;
    sel.appendChild(o);
  });
  // Restore previous selection if still available
  if(curVal){
    const opt=Array.from(sel.options).find(o=>o.value===curVal&&!o.disabled);
    if(opt)sel.value=curVal;
  }
}

// ─── State ──────────────────────────────────────────────────────────────────
let D=null, bevMode='argmax', camMode='6', sel=null, hover=null, actSingle=0;
let intr=[], extr=[], origIntr=[], origExtr=[], cfgCam=0;
let bevBg=null, camImgs=Array(6).fill(null), hmImgs=Array(6).fill(null);
const bevC=document.getElementById('bev-c'), ctx=bevC.getContext('2d');
const tip=document.getElementById('bev-tip'), info=document.getElementById('bev-info');
const grid=document.getElementById('cam-grid'), statusEl=document.getElementById('status');

function setStatus(s){statusEl.textContent=s;}

// ─── Matrix math ────────────────────────────────────────────────────────────
function m4v4(M,v){return[M[0]*v[0]+M[1]*v[1]+M[2]*v[2]+M[3]*v[3],M[4]*v[0]+M[5]*v[1]+M[6]*v[2]+M[7]*v[3],M[8]*v[0]+M[9]*v[1]+M[10]*v[2]+M[11]*v[3],M[12]*v[0]+M[13]*v[1]+M[14]*v[2]+M[15]*v[3]];}
function m3v3(M,v){return[M[0]*v[0]+M[1]*v[1]+M[2]*v[2],M[3]*v[0]+M[4]*v[1]+M[5]*v[2],M[6]*v[0]+M[7]*v[1]+M[8]*v[2]];}

// 3x3 matrix inversion (row-major 9-elem)
function inv3x3(M){
  const [a,b,c,d,e,f,g,h,k]=M;
  const det=a*(e*k-f*h)-b*(d*k-f*g)+c*(d*h-e*g);
  if(Math.abs(det)<1e-12)return null;
  const id=1/det;
  return[(e*k-f*h)*id,(c*h-b*k)*id,(b*f-c*e)*id,
         (f*g-d*k)*id,(a*k-c*g)*id,(c*d-a*f)*id,
         (d*h-e*g)*id,(b*g-a*h)*id,(a*e-b*d)*id];
}

// 4x4 matrix inversion (row-major 16-elem)
function inv4x4(m){
  const [a00,a01,a02,a03,a10,a11,a12,a13,a20,a21,a22,a23,a30,a31,a32,a33]=m;
  const b00=a00*a11-a01*a10, b01=a00*a12-a02*a10, b02=a00*a13-a03*a10;
  const b03=a01*a12-a02*a11, b04=a01*a13-a03*a11, b05=a02*a13-a03*a12;
  const b06=a20*a31-a21*a30, b07=a20*a32-a22*a30, b08=a20*a33-a23*a30;
  const b09=a21*a32-a22*a31, b10=a21*a33-a23*a31, b11=a22*a33-a23*a32;
  const det=b00*b11-b01*b10+b02*b09+b03*b08-b04*b07+b05*b06;
  if(Math.abs(det)<1e-12)return null;
  const id=1/det;
  return[
    (a11*b11-a12*b10+a13*b09)*id,(-a01*b11+a02*b10-a03*b09)*id,(a31*b05-a32*b04+a33*b03)*id,(-a21*b05+a22*b04-a23*b03)*id,
    (-a10*b11+a12*b08-a13*b07)*id,(a00*b11-a02*b08+a03*b07)*id,(-a30*b05+a32*b02-a33*b01)*id,(a20*b05-a22*b02+a23*b01)*id,
    (a10*b10-a11*b08+a13*b06)*id,(-a00*b10+a01*b08-a03*b06)*id,(a30*b04-a31*b02+a33*b00)*id,(-a20*b04+a21*b02-a23*b00)*id,
    (-a10*b09+a11*b07-a12*b06)*id,(a00*b09-a01*b07+a02*b06)*id,(-a30*b03+a31*b01-a32*b00)*id,(a20*b03-a21*b01+a22*b00)*id
  ];
}

function proj(ci,wx,wy,wz){
  const E=extr[ci],K=intr[ci];
  if(!E||!K)return{u:0,v:0,d:-1,ok:false};
  const p=m4v4(E,[wx,wy,wz,1]);
  if(p[2]<=0)return{u:0,v:0,d:p[2],ok:false};
  const px=m3v3(K,[p[0]/p[2],p[1]/p[2],1]);
  const sz=D?D.image_sizes[ci]:[1600,900];
  return{u:px[0],v:px[1],d:p[2],ok:px[0]>=0&&px[0]<sz[0]&&px[1]>=0&&px[1]<sz[1]};
}
// nuScenes ego frame: X=forward, Y=left, Z=up
// BEV (wx=right, wz=forward) → ego (X=wz, Y=-wx, Z=0)
function bev2ego(wx,wz){return[wz, -wx, 0];}
// ego → BEV (inverse of bev2ego): ego(X,Y,Z) → bev(wx=-Y, wz=X)
function ego2bev(ex,ey,ez){return[-ey, ex];}

function visCams(wx,wz){const e=bev2ego(wx,wz);const r=[];for(let c=0;c<6;c++){const p=proj(c,e[0],e[1],e[2]);if(p.ok&&p.d>0)r.push(c);}return r;}

// ─── Camera pixel → BEV cell (forward map) ──────────────────────────────────
// Unproject a camera pixel to the ego ground plane (Z=0), return BEV cell
function camPixelToBEV(ci, u, v) {
  if(!D || !extr[ci] || !intr[ci]) return null;
  const Ki = inv3x3(intr[ci]);       // inv(K) to unproject pixel → camera ray
  const Ei = inv4x4(extr[ci]);       // inv(ego_to_camera) = camera_to_ego
  if(!Ki || !Ei) return null;

  // Ray in camera frame: direction = K_inv × [u, v, 1]
  const ray_cam = m3v3(Ki, [u, v, 1]);

  // Transform ray to ego frame using camera_to_ego rotation (3x3 upper-left)
  const ray_ego = [
    Ei[0]*ray_cam[0] + Ei[1]*ray_cam[1] + Ei[2]*ray_cam[2],
    Ei[4]*ray_cam[0] + Ei[5]*ray_cam[1] + Ei[6]*ray_cam[2],
    Ei[8]*ray_cam[0] + Ei[9]*ray_cam[1] + Ei[10]*ray_cam[2]
  ];
  // Camera origin in ego frame (translation column of camera_to_ego)
  const org = [Ei[3], Ei[7], Ei[11]];

  // Intersect with ego Z=0 plane: org.z + t * ray_ego.z = 0
  if(Math.abs(ray_ego[2]) < 1e-8) return null;  // ray parallel to ground
  const t = -org[2] / ray_ego[2];
  if(t < 0) return null;  // intersection behind camera

  // Ego hit point
  const ego_x = org[0] + t * ray_ego[0];
  const ego_y = org[1] + t * ray_ego[1];

  // Ego → BEV world coords
  const [wx, wz] = ego2bev(ego_x, ego_y, 0);

  // BEV world → cell indices
  const g = D.bev_info.grid_range, r = D.bev_info.resolution, n = D.bev_info.grid_cells;
  const j = Math.floor((wx + g) / r);
  const i = Math.floor((g - wz) / r);
  if(i < 0 || i >= n || j < 0 || j >= n) return null;  // outside grid

  return {i, j, wx, wz, ego_x, ego_y, depth: t};
}

// Unproject camera pixel to ego-frame ray (for 3D voxel intersection)
function camPixelToRay(ci, u, v) {
  if(!extr[ci] || !intr[ci]) return null;
  const Ki = inv3x3(intr[ci]);
  const Ei = inv4x4(extr[ci]);
  if(!Ki || !Ei) return null;
  const ray_cam = m3v3(Ki, [u, v, 1]);
  const ray_ego = [
    Ei[0]*ray_cam[0] + Ei[1]*ray_cam[1] + Ei[2]*ray_cam[2],
    Ei[4]*ray_cam[0] + Ei[5]*ray_cam[1] + Ei[6]*ray_cam[2],
    Ei[8]*ray_cam[0] + Ei[9]*ray_cam[1] + Ei[10]*ray_cam[2]
  ];
  // Normalize
  const len = Math.sqrt(ray_ego[0]**2 + ray_ego[1]**2 + ray_ego[2]**2);
  const dir = [ray_ego[0]/len, ray_ego[1]/len, ray_ego[2]/len];
  const org = [Ei[3], Ei[7], Ei[11]];
  return {org, dir};
}

// ─── BEV helpers ────────────────────────────────────────────────────────────
function c2w(i,j){if(!D)return[0,0];const g=D.bev_info.grid_range,r=D.bev_info.resolution;return[-g+(j+.5)*r,g-(i+.5)*r];}
function px2cell(cx,cy){
  if(!D)return null;
  const n=D.bev_info.grid_cells,rect=bevC.getBoundingClientRect();
  const sx=800/rect.width,sy=800/rect.height;
  const j=Math.floor(cx*sx/800*n),i=Math.floor(cy*sy/800*n);
  return(i>=0&&i<n&&j>=0&&j<n)?{i,j}:null;
}

// ─── Draw BEV ───────────────────────────────────────────────────────────────
let showCoverage = false;
let showGtBoxes = false;

function drawBev(){
  ctx.clearRect(0,0,800,800);
  if(bevBg&&bevBg.complete)ctx.drawImage(bevBg,0,0,800,800);
  else{ctx.fillStyle='#0a0a0a';ctx.fillRect(0,0,800,800);}
  if(!D)return;
  const n=D.bev_info.grid_cells,c=800/n;
  if(showCoverage) drawCamCoverage(ctx, n, c);
  if(hover&&(!sel||hover.i!==sel.i||hover.j!==sel.j)){ctx.strokeStyle='rgba(80,200,220,.45)';ctx.lineWidth=1.5;ctx.strokeRect(hover.j*c,hover.i*c,c,c);}
  if(sel){ctx.fillStyle='rgba(80,220,255,.25)';ctx.fillRect(sel.j*c,sel.i*c,c,c);ctx.strokeStyle='#5cf';ctx.lineWidth=2;ctx.strokeRect(sel.j*c,sel.i*c,c,c);}
}

// Camera coverage toggle
document.getElementById('bev-coverage-toggle').addEventListener('click', function(){
  showCoverage = !showCoverage;
  this.classList.toggle('on', showCoverage);
  drawBev();
});

// GT boxes toggle
document.getElementById('bev-gt-toggle').addEventListener('click', function(){
  showGtBoxes = !showGtBoxes;
  this.classList.toggle('on', showGtBoxes);
  if(D) {
    const src = showGtBoxes && D.bev_images_gt ? D.bev_images_gt[bevMode] || D.bev_images_gt.argmax : D.bev_images[bevMode] || D.bev_images.argmax;
    if(src) { bevBg = new Image(); bevBg.onload = () => drawBev(); bevBg.src = src; }
  }
});

// Draw camera FOV divider lines from ego center to BEV edges
function drawCamCoverage(ctx, gridCells, cellPx) {
  if(!D || !extr.length) return;
  const g = D.bev_info.grid_range;
  const size = gridCells * cellPx;
  const cx = size / 2, cy = size / 2; // ego = center of BEV

  for(let ci = 0; ci < 6; ci++) {
    if(!extr[ci] || !intr[ci]) continue;
    const sz = D.image_sizes[ci] || [1600, 900];
    const Ki = inv3x3(intr[ci]);
    const Ei = inv4x4(extr[ci]);
    if(!Ki || !Ei) continue;

    // Unproject left and right image edges at mid-height to get FOV boundaries
    const midV = sz[1] / 2;
    const edges = [[0, midV], [sz[0], midV]]; // left edge, right edge

    for(const [pu, pv] of edges) {
      const ray_cam = m3v3(Ki, [pu, pv, 1]);
      const ray_ego = [
        Ei[0]*ray_cam[0] + Ei[1]*ray_cam[1] + Ei[2]*ray_cam[2],
        Ei[4]*ray_cam[0] + Ei[5]*ray_cam[1] + Ei[6]*ray_cam[2],
        Ei[8]*ray_cam[0] + Ei[9]*ray_cam[1] + Ei[10]*ray_cam[2]
      ];

      // Extend ray to BEV edge (use large t)
      const len = Math.sqrt(ray_ego[0]**2 + ray_ego[1]**2) || 1;
      const dx = ray_ego[0] / len, dy = ray_ego[1] / len;
      const far = g * 1.5; // extend past grid edge
      const ego_x = dx * far, ego_y = dy * far;

      // Ego → BEV pixel
      const wx = -ego_y, wz = ego_x;
      const px_x = (wx + g) / (2 * g) * size;
      const px_y = (g - wz) / (2 * g) * size;

      ctx.save();
      ctx.strokeStyle = CC[ci];
      ctx.globalAlpha = 0.45;
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(px_x, px_y);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();
    }

    // Label: draw camera name near ego center along the center ray direction
    const ray_cam_c = m3v3(Ki, [sz[0]/2, sz[1]/2, 1]);
    const ray_ego_c = [
      Ei[0]*ray_cam_c[0] + Ei[1]*ray_cam_c[1] + Ei[2]*ray_cam_c[2],
      Ei[4]*ray_cam_c[0] + Ei[5]*ray_cam_c[1] + Ei[6]*ray_cam_c[2],
      Ei[8]*ray_cam_c[0] + Ei[9]*ray_cam_c[1] + Ei[10]*ray_cam_c[2]
    ];
    const clen = Math.sqrt(ray_ego_c[0]**2 + ray_ego_c[1]**2) || 1;
    const labelDist = 55; // pixels from center
    const lx = cx + (-ray_ego_c[1] / clen) * labelDist;
    const ly = cy + (-ray_ego_c[0] / clen) * labelDist;
    ctx.save();
    ctx.fillStyle = CC[ci];
    ctx.globalAlpha = 0.7;
    ctx.font = '9px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(CN[ci].replace('CAM_',''), lx, ly);
    ctx.restore();
  }

  // Ego dot
  ctx.save();
  ctx.fillStyle = '#fff';
  ctx.beginPath();
  ctx.arc(cx, cy, 3, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

// ─── Build camera cards ─────────────────────────────────────────────────────
let camClickMark = null; // {gi, u, v} — last clicked pixel on a camera
function buildCards(){
  grid.innerHTML='';
  for(let gi=0;gi<6;gi++){
    const ci=GI[gi],col=CC[ci];
    const d=document.createElement('div');d.className='cc';d.id='cc-'+gi;d.dataset.ci=ci;
    d.innerHTML=`<div class="cc-h"><span class="cc-n" style="color:${col}">${GN[gi]}</span><span class="cc-p" id="cp-${gi}"></span></div><div class="cc-w"><canvas id="cv-${gi}" width="1600" height="900"></canvas></div>`;
    // Single-cam mode toggle (double-click)
    d.addEventListener('dblclick',()=>{if(camMode==='1'){actSingle=gi;layoutCams();drawCams(sel?c2w(sel.i,sel.j):null);}});
    grid.appendChild(d);

    // Forward map: click camera pixel → highlight BEV cell or 3D voxel
    const cv = d.querySelector('canvas');
    cv.style.cursor = 'crosshair';
    cv.addEventListener('click', (e) => {
      e.stopPropagation();
      if(!D) return;
      const rect = cv.getBoundingClientRect();
      const u = (e.clientX - rect.left) / rect.width * cv.width;
      const v = (e.clientY - rect.top) / rect.height * cv.height;
      camClickMark = {gi, u, v};

      // BEV ground intersection
      const hit = camPixelToBEV(ci, u, v);
      if(hit) {
        sel = {i: hit.i, j: hit.j};
        info.textContent = `${CN[ci]} (${u.toFixed(0)},${v.toFixed(0)}) → BEV [${hit.i},${hit.j}] ego(${hit.ego_x.toFixed(1)},${hit.ego_y.toFixed(1)}) d=${hit.depth.toFixed(1)}m`;
        layoutCams(); drawBev(); drawCams([hit.wx, hit.wz]);
      } else {
        info.textContent = `${CN[ci]} (${u.toFixed(0)},${v.toFixed(0)}) → no ground hit (sky/behind)`;
        drawCams(sel ? c2w(sel.i, sel.j) : null);
      }
    });
  }
}

function layoutCams(){
  let vis=[0,1,2,3,4,5];
  if(camMode==='auto'&&sel){
    const[wx,wz]=c2w(sel.i,sel.j);
    const vc=visCams(wx,wz);vis=[];
    for(let gi=0;gi<6;gi++)if(vc.includes(GI[gi]))vis.push(gi);
    if(!vis.length)vis=[0,1,2,3,4,5];
  }else if(camMode==='1')vis=[actSingle];
  for(let gi=0;gi<6;gi++){const e=document.getElementById('cc-'+gi);if(e)e.classList.toggle('hid',!vis.includes(gi));}
  const n=vis.length,g=grid;
  if(n<=1){g.style.gridTemplateColumns='1fr';g.style.gridTemplateRows='1fr';}
  else if(n<=2){g.style.gridTemplateColumns='repeat(2,1fr)';g.style.gridTemplateRows='1fr';}
  else if(n<=3){g.style.gridTemplateColumns='repeat(3,1fr)';g.style.gridTemplateRows='1fr';}
  else if(n<=4){g.style.gridTemplateColumns='repeat(2,1fr)';g.style.gridTemplateRows='repeat(2,1fr)';}
  else{g.style.gridTemplateColumns='repeat(3,1fr)';g.style.gridTemplateRows='repeat(2,1fr)';}
}

// ─── Draw cameras ───────────────────────────────────────────────────────────
// drawCams: wp=[wx,wz] for BEV 2D, or ego=[x,y,z] for 3D voxel click
function drawCams(wp, ego3d){
  for(let gi=0;gi<6;gi++){
    const ci=GI[gi],cv=document.getElementById('cv-'+gi),card=document.getElementById('cc-'+gi),px=document.getElementById('cp-'+gi);
    if(!cv||!card||card.classList.contains('hid'))continue;
    const c=cv.getContext('2d'),sz=D?D.image_sizes[ci]:[1600,900];
    cv.width=sz[0];cv.height=sz[1];const w=sz[0],h=sz[1];
    c.fillStyle='#0d0d0d';c.fillRect(0,0,w,h);
    if(camImgs[ci]&&camImgs[ci].complete)c.drawImage(camImgs[ci],0,0,w,h);
    if(hmImgs[ci]&&hmImgs[ci].complete){c.globalAlpha=.55;c.drawImage(hmImgs[ci],0,0,w,h);c.globalAlpha=1;}
    c.strokeStyle=CC[ci]+'44';c.lineWidth=2;c.strokeRect(0,0,w,h);
    const hasTarget = wp || ego3d;
    if(hasTarget){
      const e = ego3d ? ego3d : bev2ego(wp[0],wp[1]); // ego frame: X=fwd, Y=left, Z=up
      const p=proj(ci,e[0],e[1],e[2]);
      if(p.ok){
        card.className='cc vis';if(px)px.textContent=`(${p.u.toFixed(0)},${p.v.toFixed(0)})`;
        c.strokeStyle=CC[ci];c.lineWidth=2;c.globalAlpha=.8;
        const a=22;c.beginPath();c.moveTo(p.u-a,p.v);c.lineTo(p.u+a,p.v);c.stroke();
        c.beginPath();c.moveTo(p.u,p.v-a);c.lineTo(p.u,p.v+a);c.stroke();c.globalAlpha=1;
        c.fillStyle=CC[ci];c.beginPath();c.arc(p.u,p.v,4,0,Math.PI*2);c.fill();
        c.strokeStyle=CC[ci];c.lineWidth=1.5;c.beginPath();c.arc(p.u,p.v,14,0,Math.PI*2);c.stroke();
      }else{
        card.className='cc out';if(px)px.textContent=p.d<=0?'behind':'outside';
      }
    }else{card.className='cc';if(px)px.textContent=`${w}×${h}`;}
    // Camera click origin marker (shows where user clicked)
    if(camClickMark && camClickMark.gi === gi) {
      const mu = camClickMark.u, mv = camClickMark.v;
      c.strokeStyle='#fff';c.lineWidth=2;c.beginPath();c.arc(mu,mv,18,0,Math.PI*2);c.stroke();
      c.fillStyle='#fff';c.beginPath();c.arc(mu,mv,4,0,Math.PI*2);c.fill();
      c.strokeStyle='rgba(255,255,255,0.3)';c.lineWidth=1;
      c.setLineDash([6,4]);c.beginPath();c.moveTo(mu,0);c.lineTo(mu,h);c.stroke();
      c.beginPath();c.moveTo(0,mv);c.lineTo(w,mv);c.stroke();c.setLineDash([]);
    }
    // Name label
    c.fillStyle='rgba(0,0,0,.55)';c.fillRect(0,0,Math.min(150,w),22);
    c.fillStyle=CC[ci];c.font='bold 12px monospace';c.fillText(CN[ci].replace('CAM_',''),6,15);
  }
}

// ─── BEV events ─────────────────────────────────────────────────────────────
// Point-in-polygon (ray casting) for GT box hit testing
function pointInPoly(px,py,corners){
  let inside=false;
  for(let i=0,j=corners.length-1;i<corners.length;j=i++){
    const xi=corners[i][0],yi=corners[i][1],xj=corners[j][0],yj=corners[j][1];
    if(((yi>py)!==(yj>py))&&(px<(xj-xi)*(py-yi)/(yj-yi)+xi))inside=!inside;
  }
  return inside;
}
function findGtBox(wx,wz){
  if(!D||!D.gt_boxes||!showGtBoxes)return null;
  for(const b of D.gt_boxes){
    if(pointInPoly(wx,wz,b.corners))return b;
  }
  return null;
}

bevC.addEventListener('mousemove',e=>{
  if(!D)return;
  const r=bevC.getBoundingClientRect(),cx=e.clientX-r.left,cy=e.clientY-r.top;
  const cell=px2cell(cx,cy);hover=cell;
  if(cell){
    const[wx,wz]=c2w(cell.i,cell.j);
    tip.style.display='block';tip.style.left=(e.clientX+14)+'px';tip.style.top=(e.clientY-20)+'px';
    let t=`[${cell.i},${cell.j}] X:${wx.toFixed(1)} Z:${wz.toFixed(1)}`;
    if(D.bev_info.cell_classes.length){
      const idx=cell.i*D.bev_info.grid_cells+cell.j;
      t+=` | ${D.bev_info.class_names[D.bev_info.cell_classes[idx]]} ${D.bev_info.cell_confs[idx].toFixed(2)}`;
    }
    const gtBox=findGtBox(wx,wz);
    if(gtBox) t+=`\nGT: ${gtBox.class_name}`;
    tip.style.whiteSpace='pre-wrap';
    tip.textContent=t;drawBev();drawCams([wx,wz]);
  }else{tip.style.display='none';drawBev();drawCams(sel?c2w(sel.i,sel.j):null);}
});
bevC.addEventListener('mouseleave',()=>{hover=null;tip.style.display='none';drawBev();drawCams(sel?c2w(sel.i,sel.j):null);});
bevC.addEventListener('click',e=>{
  if(!D)return;
  const r=bevC.getBoundingClientRect(),cell=px2cell(e.clientX-r.left,e.clientY-r.top);
  if(!cell)return;sel=cell;camClickMark=null; // clear camera click marker
  const[wx,wz]=c2w(cell.i,cell.j);
  info.textContent=`Selected [${cell.i},${cell.j}] X=${wx.toFixed(2)}m Z=${wz.toFixed(2)}m`;
  layoutCams();drawBev();drawCams([wx,wz]);
});
document.addEventListener('keydown',e=>{
  if(!sel||!D)return;const n=D.bev_info.grid_cells;let{i,j}=sel;
  if(e.key==='ArrowUp'){i=Math.max(0,i-1);e.preventDefault();}
  else if(e.key==='ArrowDown'){i=Math.min(n-1,i+1);e.preventDefault();}
  else if(e.key==='ArrowLeft'){j=Math.max(0,j-1);e.preventDefault();}
  else if(e.key==='ArrowRight'){j=Math.min(n-1,j+1);e.preventDefault();}
  else return;
  sel={i,j};const[wx,wz]=c2w(i,j);
  info.textContent=`Selected [${i},${j}] X=${wx.toFixed(2)}m Z=${wz.toFixed(2)}m`;
  layoutCams();drawBev();drawCams([wx,wz]);
});

// ─── Tab switching ──────────────────────────────────────────────────────────
document.querySelectorAll('#bev-tabs .tab').forEach(b=>b.addEventListener('click',()=>{
  document.querySelectorAll('#bev-tabs .tab').forEach(x=>x.classList.remove('on'));
  b.classList.add('on');bevMode=b.dataset.m;
  const bevWrap = document.getElementById('bev-wrap');
  bevWrap.style.display = '';
  if(D&&D.bev_images&&D.bev_images[bevMode]){
    bevBg=new Image();bevBg.onload=()=>drawBev();bevBg.src=D.bev_images[bevMode];
  }else{fetchBev();}
}));
document.querySelectorAll('#cam-tabs .tab').forEach(b=>b.addEventListener('click',()=>{
  document.querySelectorAll('#cam-tabs .tab').forEach(x=>x.classList.remove('on'));
  b.classList.add('on');camMode=b.dataset.m;
  layoutCams();drawCams(sel?c2w(sel.i,sel.j):null);
}));

// ─── Config modal ───────────────────────────────────────────────────────────
document.getElementById('cfg-open').addEventListener('click',()=>{document.getElementById('cfg-modal').classList.add('open');buildCfgTabs();loadCfg(cfgCam);});
document.getElementById('cfg-close').addEventListener('click',()=>document.getElementById('cfg-modal').classList.remove('open'));
document.getElementById('cfg-revert').addEventListener('click',()=>{intr=origIntr.map(a=>[...a]);extr=origExtr.map(a=>[...a]);loadCfg(cfgCam);drawCams(sel?c2w(sel.i,sel.j):null);});
document.getElementById('cfg-apply').addEventListener('click',()=>{saveCfg(cfgCam);drawCams(sel?c2w(sel.i,sel.j):null);});

function buildCfgTabs(){
  const el=document.getElementById('cfg-tabs');el.innerHTML='';
  for(let ci=0;ci<6;ci++){
    const b=document.createElement('button');b.className='tab'+(ci===cfgCam?' on':'');
    b.style.color=CC[ci];if(ci===cfgCam)b.style.borderColor=CC[ci];
    b.textContent=CN[ci].replace('CAM_','');
    b.addEventListener('click',()=>{saveCfg(cfgCam);cfgCam=ci;buildCfgTabs();loadCfg(ci);});
    el.appendChild(b);
  }
}
function loadCfg(ci){
  const K=intr[ci]||[800,0,800,0,800,450,0,0,1];
  document.getElementById('cfg-fx').value=K[0].toFixed(2);
  document.getElementById('cfg-fy').value=K[4].toFixed(2);
  document.getElementById('cfg-cx').value=K[2].toFixed(2);
  document.getElementById('cfg-cy').value=K[5].toFixed(2);
  const E=extr[ci]||Array(16).fill(0);
  document.getElementById('cfg-ext').value=Array.from({length:4},(_,r)=>E.slice(r*4,r*4+4).map(v=>v.toFixed(6)).join(', ')).join('\n');
}
function saveCfg(ci){
  const K=intr[ci]||[800,0,800,0,800,450,0,0,1];
  K[0]=parseFloat(document.getElementById('cfg-fx').value)||K[0];
  K[4]=parseFloat(document.getElementById('cfg-fy').value)||K[4];
  K[2]=parseFloat(document.getElementById('cfg-cx').value)||K[2];
  K[5]=parseFloat(document.getElementById('cfg-cy').value)||K[5];
  intr[ci]=K;
  try{const v=document.getElementById('cfg-ext').value.replace(/\n/g,',').split(',').map(s=>parseFloat(s.trim()));if(v.length===16&&v.every(x=>!isNaN(x)))extr[ci]=v;}catch(e){}
}

// ─── API calls ──────────────────────────────────────────────────────────────
async function fetchScene(){
  const btn=document.getElementById('btn-load');btn.disabled=true;btn.classList.add('loading');btn.textContent='Loading…';
  setStatus('Loading scene…');
  try{
    const res=await fetch('/api/load-scene',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({scene_idx:parseInt(selScene.value),sample_idx:parseInt(selSample.value),backend:selRepr.value||'lss'})});
    D=await res.json();
    sel=null;hover=null;hmImgs=Array(6).fill(null);
    intr=D.intrinsics.map(a=>[...a]);extr=D.extrinsics.map(a=>[...a]);
    origIntr=D.intrinsics.map(a=>[...a]);origExtr=D.extrinsics.map(a=>[...a]);
    // Update dynamic class list from model
    if(D.bev_info&&D.bev_info.class_names)populateClasses(D.bev_info.class_names);
    // Update repr type availability
    if(D.repr_types)updateReprTypes(D.repr_types);
    // 3D tab removed (LSS-only)
    if(bevMode==='3d'){
      bevMode='argmax';
      document.querySelectorAll('#bev-tabs .tab').forEach(x=>x.classList.remove('on'));
      document.querySelector('#bev-tabs .tab[data-m="argmax"]').classList.add('on');
      document.getElementById('bev-wrap').style.display='';
      // 3D viewer removed (LSS-only)
    }
    // Load BEV bg
    bevBg=new Image();bevBg.onload=()=>drawBev();bevBg.src=D.bev_images[bevMode]||D.bev_images.argmax;
    // Load cam images
    for(let ci=0;ci<6;ci++){
      if(D.camera_images[ci]){camImgs[ci]=new Image();camImgs[ci].onload=((_ci)=>()=>drawCams(null))(ci);camImgs[ci].src=D.camera_images[ci];}
      else camImgs[ci]=null;
    }
    layoutCams();drawBev();drawCams(null);
    setStatus(D.status);info.textContent='Click BEV grid to select cell';
  }catch(e){setStatus('Error: '+e.message);console.error(e);}
  btn.disabled=false;btn.classList.remove('loading');btn.textContent='Load Scene';
}

async function fetchBev(){
  try{
    const res=await fetch('/api/render-bev',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({mode:bevMode,class_name:selClass.value})});
    const r=await res.json();
    if(r.bev_image){bevBg=new Image();bevBg.onload=()=>drawBev();bevBg.src=r.bev_image;
      // Cache it
      if(D&&D.bev_images)D.bev_images[bevMode]=r.bev_image;
    }
  }catch(e){console.error(e);}
}

async function fetchAttr(){
  if(!sel){setStatus('Select a BEV cell first');return;}
  const btn=document.getElementById('btn-attr');btn.disabled=true;btn.classList.add('loading');btn.textContent='Running…';
  setStatus('Computing attribution…');
  try{
    const res=await fetch('/api/attribute',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({cell_i:sel.i,cell_j:sel.j,method:selMethod.value,class_name:selClass.value})});
    const r=await res.json();
    if(r.error){setStatus('Error: '+r.error);}
    else if(r.heatmaps&&r.heatmaps.length===6){
      // Clear old heatmaps before loading new ones
      hmImgs = Array(6).fill(null);
      drawCams(sel?c2w(sel.i,sel.j):null);
      for(let ci=0;ci<6;ci++){
        if(r.heatmaps[ci]){hmImgs[ci]=new Image();hmImgs[ci].onload=((_ci)=>()=>drawCams(sel?c2w(sel.i,sel.j):null))(ci);hmImgs[ci].src=r.heatmaps[ci];}
      }
      setStatus(r.status||'Attribution complete');
    }
  }catch(e){setStatus('Error: '+e.message);console.error(e);}
  btn.disabled=false;btn.classList.remove('loading');btn.textContent='Run Attribution';
}

// Class change → re-render BEV if in class_heatmap mode
selClass.addEventListener('change',()=>{if(bevMode==='class_heatmap')fetchBev();});

document.getElementById('btn-load').addEventListener('click',fetchScene);
document.getElementById('btn-attr').addEventListener('click',fetchAttr);

// ─── Attribution Method Info Tooltip ─────────────────────────────────────
const METHOD_INFO={
  'GradCAM':{title:'GradCAM (Gradient-weighted Class Activation Mapping)',body:'Computes gradients of the target class logit w.r.t. the last convolutional feature map. Weights each channel by its mean gradient, producing a coarse heatmap highlighting which spatial regions the model "looks at". Fast, but limited to the resolution of the feature map.'},
  'Integrated Gradients':{title:'Integrated Gradients',body:'Accumulates gradients along a straight path from a baseline (e.g. black image) to the actual input. Satisfies axioms of sensitivity and implementation invariance. More precise than GradCAM but slower — requires multiple forward passes (default: 50 interpolation steps).'},
  'Attention':{title:'Attention Rollout / Cross-Attention',body:'Extracts cross-attention weights from transformer layers (e.g. BEVFormer). Maps BEV query positions back to camera image regions via attention scores. Only available for attention-based architectures. Falls back to gradient×input if no attention layers are found.'},
  'Occlusion':{title:'Occlusion Sensitivity',body:'Slides a grey patch across the input image and measures how much the target logit drops. High sensitivity regions are important for the prediction. Model-agnostic but very slow — requires one forward pass per patch position. Use a larger patch size for faster (but coarser) results.'}
};
const infoIcon=document.getElementById('method-info');
const infoTip=document.getElementById('info-tip');
let infoVisible=false;

infoIcon.addEventListener('click',e=>{
  e.stopPropagation();
  if(infoVisible){infoTip.style.display='none';infoVisible=false;return;}
  const method=selMethod.value;
  const mi=METHOD_INFO[method]||{title:method,body:'No description available.'};
  infoTip.innerHTML='<div class="it-title">'+mi.title+'</div><div class="it-body">'+mi.body+'</div>';
  const r=infoIcon.getBoundingClientRect();
  infoTip.style.display='block';
  infoTip.style.left=Math.min(r.left,window.innerWidth-400)+'px';
  infoTip.style.top=(r.bottom+6)+'px';
  infoVisible=true;
});
document.addEventListener('click',()=>{if(infoVisible){infoTip.style.display='none';infoVisible=false;}});
selMethod.addEventListener('change',()=>{if(infoVisible){infoTip.style.display='none';infoVisible=false;}});

// ─── Log Viewer (terminal-style, SSE-streamed) ─────────────────────────
const logPanel=document.getElementById('log-panel');
const logBody=document.getElementById('log-body');
const logToggle=document.getElementById('log-toggle');
let logOpen=false, logSSE=null;

function openLog(){
  logOpen=true;logPanel.classList.add('open');logToggle.classList.add('active');
  document.getElementById('main').style.height='calc(100vh - 170px - 240px)';
  if(!logSSE){
    logSSE=new EventSource('/api/logs/stream');
    logSSE.onmessage=e=>{
      try{
        const d=JSON.parse(e.data);
        if(Array.isArray(d)){logBody.innerHTML='';d.forEach(addLogLine);}
        else addLogLine(d);
      }catch(ex){}
    };
    logSSE.onerror=()=>{if(logSSE){logSSE.close();logSSE=null;}if(logOpen)setTimeout(()=>{if(logOpen)openLog();},2000);};
  }
}
function closeLog(){
  logOpen=false;logPanel.classList.remove('open');logToggle.classList.remove('active');
  document.getElementById('main').style.height='';
  if(logSSE){logSSE.close();logSSE=null;}
}
function addLogLine(text){
  const div=document.createElement('div');div.className='log-line';
  if(text.includes('ERROR'))div.classList.add('ERROR');
  else if(text.includes('WARN'))div.classList.add('WARNING');
  else if(text.includes('DEBUG'))div.classList.add('DEBUG');
  else div.classList.add('INFO');
  div.textContent=text;logBody.appendChild(div);
  logBody.scrollTop=logBody.scrollHeight;
  while(logBody.children.length>500)logBody.removeChild(logBody.firstChild);
}

logToggle.addEventListener('click',()=>{logOpen?closeLog():openLog();});
document.getElementById('log-close').addEventListener('click',closeLog);
document.getElementById('log-clear').addEventListener('click',()=>{logBody.innerHTML='';});

// ─── Init ───────────────────────────────────────────────────────────────────
// Fetch available backends on init
fetch('/api/backends').then(r=>r.json()).then(types=>{
  updateReprTypes(types);
}).catch(()=>{});

buildCards();drawBev();drawCams(null);

// ─── VLM Panel ───────────────────────────────────────────────────────────────
let vlmLoaded = false, vlmTokens = [], vlmWords = [], vlmActiveWord = -1;
const vlmAttnCache = new Map();
let vlmHoverTimer = null;
let vlmCurrentMode = 'bev';

// Mode switching
document.querySelectorAll('#mode-tabs .tab').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('#mode-tabs .tab').forEach(b => b.classList.remove('on'));
    btn.classList.add('on');
    vlmCurrentMode = btn.dataset.mode;
    const bevPanel = document.getElementById('bev-panel');
    const camPanel = document.getElementById('cam-panel');
    const vlmPanel = document.getElementById('vlm-panel');
    const vbPanel = document.getElementById('vb-panel');
    const ctrl = document.getElementById('ctrl');
    // Hide all panels
    bevPanel.style.display = 'none';
    camPanel.style.display = 'none';
    vlmPanel.style.display = 'none';
    vlmPanel.classList.remove('active');
    vbPanel.style.display = 'none';
    vbPanel.classList.remove('active');
    ctrl.style.display = 'none';
    if (vlmCurrentMode === 'bev') {
      bevPanel.style.display = '';
      camPanel.style.display = '';
      ctrl.style.display = '';
    } else if (vlmCurrentMode === 'vlm') {
      vlmPanel.style.display = 'flex';
      vlmPanel.classList.add('active');
      vlmPopulateCameras();
    } else if (vlmCurrentMode === 'vb') {
      vbPanel.style.display = 'flex';
      vbPanel.classList.add('active');
      vbPopulateSelectors();
      // Show existing BEV image with GT boxes if scene is loaded
      if (D && D.bev_images_gt && D.bev_images_gt.argmax) {
        document.getElementById('vb-bev-img').src = D.bev_images_gt.argmax;
      } else if (D && D.bev_images && D.bev_images.argmax) {
        document.getElementById('vb-bev-img').src = D.bev_images.argmax;
      }
    }
  });
});

let vlmImageIndex = {};  // {CAM_FRONT: [file1.jpg, ...], ...}

async function vlmPopulateCameras() {
  const sel = document.getElementById('vlm-cam-select');
  if (sel.children.length > 0) return;
  try {
    const r = await fetch('/api/vlm/images');
    vlmImageIndex = await r.json();
    Object.keys(vlmImageIndex).forEach(cam => {
      const opt = document.createElement('option');
      opt.value = cam; opt.textContent = cam + ' (' + vlmImageIndex[cam].length + ')';
      sel.appendChild(opt);
    });
    // Add image file selector
    vlmUpdateFileList();
    sel.addEventListener('change', vlmUpdateFileList);
  } catch(e) { console.error('vlmPopulateCameras:', e); }
}

function vlmUpdateFileList() {
  const cam = document.getElementById('vlm-cam-select').value;
  let fsel = document.getElementById('vlm-file-select');
  if (!fsel) return;
  fsel.innerHTML = '';
  const files = vlmImageIndex[cam] || [];
  files.forEach((f, i) => {
    const opt = document.createElement('option');
    opt.value = f;
    // Show a short label: just the timestamp part
    const parts = f.split('__');
    opt.textContent = parts.length >= 3 ? '#' + i + ' t=' + parts[2].replace('.jpg','') : f;
    fsel.appendChild(opt);
  });
  // Preview selected image
  vlmPreviewImage();
}

function vlmPreviewImage() {
  const cam = document.getElementById('vlm-cam-select').value;
  const fsel = document.getElementById('vlm-file-select');
  if (!fsel || !fsel.value) return;
  document.getElementById('vlm-img').src = '/api/vlm/image/' + cam + '/' + fsel.value;
}

async function vlmEnsureLoaded() {
  if (vlmLoaded) return true;
  const ld = document.getElementById('vlm-loading');
  ld.style.display = ''; ld.textContent = 'Loading SmolVLM-256M... (first time downloads ~500MB)';
  try {
    const r = await fetch('/api/vlm/load', {method:'POST'});
    const d = await r.json();
    ld.style.display = 'none';
    if (d.ok) { vlmLoaded = true; return true; }
    ld.style.display = ''; ld.textContent = 'Load failed: ' + (d.error || '');
    return false;
  } catch(e) {
    ld.style.display = ''; ld.textContent = 'Load error: ' + e.message;
    return false;
  }
}

document.getElementById('vlm-generate-btn').addEventListener('click', vlmGenerate);

async function vlmGenerate() {
  const btn = document.getElementById('vlm-generate-btn');
  btn.disabled = true; btn.classList.add('loading');
  if (!await vlmEnsureLoaded()) { btn.disabled = false; btn.classList.remove('loading'); return; }
  const cam = document.getElementById('vlm-cam-select').value || 'CAM_FRONT';
  const fsel = document.getElementById('vlm-file-select');
  const filename = fsel ? fsel.value : '';
  if (!filename) { btn.disabled = false; btn.classList.remove('loading'); return; }
  const prompt = document.getElementById('vlm-prompt').value;
  const statusEl = document.getElementById('vlm-status');
  statusEl.textContent = 'Generating...';
  try {
    const r = await fetch('/api/vlm/generate', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({camera: cam, filename: filename, prompt: prompt,
        attn_method: document.getElementById('vlm-attn-method').value})
    });
    const d = await r.json();
    if (d.error) { statusEl.textContent = 'Error: ' + d.error; return; }
    // Show camera image
    document.getElementById('vlm-img').src = d.image_uri;
    // Method selector removed — aggregation dropdown handles this now
    // Render tokens
    vlmTokens = d.tokens;
    vlmWords = d.words || [];
    vlmActiveWord = -1;
    vlmAttnCache.clear();
    vlmRenderWords(vlmWords);
    statusEl.textContent = d.status || '';
  } catch(e) {
    statusEl.textContent = 'Error: ' + e.message;
  } finally {
    btn.disabled = false; btn.classList.remove('loading');
  }
}

let vlmCamRunning = false;  // concurrency guard — only one CAM at a time

function vlmRenderWords(words) {
  const container = document.getElementById('vlm-text-output');
  container.innerHTML = '';
  words.forEach((word, idx) => {
    const span = document.createElement('span');
    span.className = 'vlm-word';
    span.textContent = word.text + ' ';
    span.dataset.idx = idx;
    // Color-code by strength: 0=transparent dark, 1=bright cyan bg
    const s = word.strength || 0;
    const r = Math.round(10 + s * 20), g = Math.round(30 + s * 50), b = Math.round(30 + s * 60);
    span.style.background = `rgb(${r},${g},${b})`;
    span.style.color = s > 0.5 ? '#fff' : '#aaa';
    span.addEventListener('click', () => {
      if (vlmCamRunning) return;
      if (vlmActiveWord === idx) {
        vlmActiveWord = -1;
        document.querySelectorAll('.vlm-word').forEach(el => el.classList.remove('active'));
        vlmRestoreBaseImage();
      } else {
        vlmActiveWord = idx;
        document.querySelectorAll('.vlm-word').forEach(el => el.classList.remove('active'));
        span.classList.add('active');
        vlmShowWordAttention(idx);
      }
    });
    container.appendChild(span);
  });
}

function vlmRestoreBaseImage() {
  const cam = document.getElementById('vlm-cam-select').value;
  const fsel = document.getElementById('vlm-file-select');
  if (fsel && fsel.value) {
    document.getElementById('vlm-img').src = '/api/vlm/image/' + cam + '/' + fsel.value;
  }
  document.getElementById('vlm-status').textContent = '';
}

async function vlmShowWordAttention(wordIdx) {
  if (vlmCamRunning) return;
  const cacheKey = `word-${wordIdx}`;
  if (vlmAttnCache.has(cacheKey)) {
    document.getElementById('vlm-img').src = vlmAttnCache.get(cacheKey);
    document.getElementById('vlm-status').textContent = `"${vlmWords[wordIdx]?.text}" (cached)`;
    return;
  }
  vlmCamRunning = true;
  const statusEl = document.getElementById('vlm-status');
  statusEl.textContent = `Loading "${vlmWords[wordIdx]?.text}"...`;
  try {
    const r = await fetch('/api/vlm/word-attention', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({word_index: wordIdx})
    });
    const d = await r.json();
    if (d.error) { statusEl.textContent = 'Error: ' + d.error; return; }
    vlmAttnCache.set(cacheKey, d.heatmap);
    document.getElementById('vlm-img').src = d.heatmap;
    statusEl.textContent = `"${d.word}" (strength: ${(d.strength * 100).toFixed(0)}%)`;
  } catch(e) {
    statusEl.textContent = 'Error: ' + e.message;
  } finally {
    vlmCamRunning = false;
  }
}

// ─── VLM→BEV Panel ──────────────────────────────────────────────────────────
let vbRunning = false;

// Populate VB camera/file selectors
async function vbPopulateSelectors() {
  const camSel = document.getElementById('vb-cam-select');
  if (camSel.children.length > 0) return;
  // Ensure image index is loaded
  if (Object.keys(vlmImageIndex).length === 0) {
    try {
      const r = await fetch('/api/vlm/images');
      vlmImageIndex = await r.json();
    } catch(e) { console.error('vbPopulateSelectors fetch:', e); }
  }
  const cams = Object.keys(vlmImageIndex);
  if (cams.length === 0) {
    ['CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT'].forEach(c => {
      const opt = document.createElement('option');
      opt.value = c; opt.textContent = c;
      camSel.appendChild(opt);
    });
  } else {
    cams.forEach(c => {
      const opt = document.createElement('option');
      opt.value = c; opt.textContent = c;
      camSel.appendChild(opt);
    });
  }
  camSel.addEventListener('change', vbUpdateFileList);
  vbUpdateFileList();
}

function vbUpdateFileList() {
  const cam = document.getElementById('vb-cam-select').value;
  const fsel = document.getElementById('vb-file-select');
  fsel.innerHTML = '';
  const files = vlmImageIndex[cam] || [];
  files.forEach((f, i) => {
    const opt = document.createElement('option');
    opt.value = f;
    const parts = f.split('__');
    opt.textContent = parts.length >= 3 ? '#' + i + ' t=' + parts[2].replace('.jpg','') : f;
    fsel.appendChild(opt);
  });
  // Preview on front camera panel
  if (fsel.value) {
    document.getElementById('vb-cam-img').src = '/api/vlm/image/' + cam + '/' + fsel.value;
  }
}
document.getElementById('vb-file-select').addEventListener('change', () => {
  const cam = document.getElementById('vb-cam-select').value;
  const f = document.getElementById('vb-file-select').value;
  if (f) document.getElementById('vb-cam-img').src = '/api/vlm/image/' + cam + '/' + f;
});

// Word click → project that word's attention to BEV + camera
async function vbWordClick(wordIdx, spanEl) {
  // Highlight selected word
  document.querySelectorAll('#vb-text-output .vb-word').forEach(s => s.style.borderBottom = '');
  spanEl.style.borderBottom = '2px solid #5cf';
  const cam = document.getElementById('vb-cam-select').value || 'CAM_FRONT';
  const statusEl = document.getElementById('vb-status');
  statusEl.textContent = 'Projecting word to BEV...';
  try {
    const r = await fetch('/api/vlm-bev/word', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({word_index: wordIdx, camera: cam})
    });
    const d = await r.json();
    if (d.error) { statusEl.textContent = 'Error: ' + d.error; return; }
    if (d.bev_image) document.getElementById('vb-bev-img').src = d.bev_image;
    if (d.camera_image) document.getElementById('vb-cam-img').src = d.camera_image;
    statusEl.textContent = `Showing attention for "${d.word}"`;
  } catch(e) { statusEl.textContent = 'Error: ' + e.message; }
}

// BEV click → project to camera
document.getElementById('vb-bev-img').addEventListener('click', async (e) => {
  const img = e.target;
  const rect = img.getBoundingClientRect();
  const x_frac = (e.clientX - rect.left) / rect.width;
  const y_frac = (e.clientY - rect.top) / rect.height;
  const cam = document.getElementById('vb-cam-select').value || 'CAM_FRONT';
  const statusEl = document.getElementById('vb-status');
  try {
    const r = await fetch('/api/vlm-bev/click', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({x_frac, y_frac, camera: cam})
    });
    const d = await r.json();
    if (d.error) { statusEl.textContent = d.error; return; }
    document.getElementById('vb-cam-img').src = d.camera_image;
    statusEl.textContent = `BEV → pixel (${d.u}, ${d.v}) | ${d.dist_m}m from ego | ego (${d.ego_x}, ${d.ego_y})`;
  } catch(e) { statusEl.textContent = 'Click error: ' + e.message; }
});
document.getElementById('vb-bev-img').style.cursor = 'crosshair';

document.getElementById('vb-btn-preset').addEventListener('click', () => {
  const sel = document.getElementById('vb-preset-select');
  if (sel.value) {
    document.getElementById('vb-prompt').value = sel.value;
    vbRun();
  }
});
document.getElementById('vb-preset-select').addEventListener('dblclick', () => {
  const sel = document.getElementById('vb-preset-select');
  if (sel.value) {
    document.getElementById('vb-prompt').value = sel.value;
    vbRun();
  }
});
document.getElementById('vb-btn-run').addEventListener('click', vbRun);

async function vbRun() {
  if (vbRunning) return;
  vbRunning = true;
  const btn = document.getElementById('vb-btn-run');
  btn.disabled = true; btn.classList.add('loading');
  const statusEl = document.getElementById('vb-status');
  statusEl.textContent = 'Running VLM + BEV projection...';

  const cam = document.getElementById('vb-cam-select').value || 'CAM_FRONT';
  const filename = document.getElementById('vb-file-select').value;
  const prompt = document.getElementById('vb-prompt').value;

  if (!filename) {
    statusEl.textContent = 'Select an image first';
    btn.disabled = false; btn.classList.remove('loading'); vbRunning = false;
    return;
  }

  try {
    const r = await fetch('/api/vlm-bev/run', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({prompt, camera: cam, filename})
    });
    const d = await r.json();
    if (d.error) { statusEl.textContent = 'Error: ' + d.error; return; }

    // Update images
    if (d.bev_image) document.getElementById('vb-bev-img').src = d.bev_image;
    if (d.camera_image) document.getElementById('vb-cam-img').src = d.camera_image;
    if (d.bev_warn) statusEl.textContent = d.bev_warn;

    // Render words with strength coloring + click to project individual word
    const container = document.getElementById('vb-text-output');
    container.innerHTML = '';
    if (d.words) {
      d.words.forEach((w, idx) => {
        const span = document.createElement('span');
        span.className = 'vb-word';
        span.textContent = w.text + ' ';
        span.style.cursor = 'pointer';
        const s = w.strength;
        span.style.background = `rgba(${Math.floor(s*90)},${Math.floor(s*200)},${Math.floor(s*255)},${s*0.5})`;
        span.addEventListener('click', () => vbWordClick(idx, span));
        container.appendChild(span);
      });
    }

    statusEl.textContent = d.status || '';
  } catch(e) {
    statusEl.textContent = 'Error: ' + e.message;
  } finally {
    btn.disabled = false; btn.classList.remove('loading');
    vbRunning = false;
  }
}

</script>
</body></html>
"""

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=== BEV Attribution Tool — imports complete ===", flush=True)
    print(f"  pipeline_ok={_pipeline_ok}  backends_ok={_backends_ok}  vlm_ok={_vlm_ok}", flush=True)
    print("Starting uvicorn on http://0.0.0.0:7860 …", flush=True)
    uvicorn.run(server, host='0.0.0.0', port=7860)
