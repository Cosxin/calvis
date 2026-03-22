"""BEV Attribution Debug Tool — FastAPI + custom HTML frontend."""

import os, base64, io, json, logging, time, traceback, collections
import numpy as np
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
    from pipeline.data import load_sample
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
    """Build representation type list from available backends."""
    types = []
    # Map backend repr_type to UI labels
    seen = set()
    repr_labels = {
        'bev_seg': '2D BEV Segmentation',
        '3d_occ': '3D Occupancy',
        'gaussian': 'Gaussian 3D',
    }
    if _backends_ok:
        for name, cls in BACKENDS.items():
            b = cls()
            rt = b.repr_type
            types.append({
                'id': name,
                'label': f"{repr_labels.get(rt, rt)} ({name})",
                'available': True,
                'repr_type': rt,
            })
    else:
        types = [
            {'id': 'lss', 'label': '2D BEV Seg (LSS)', 'available': True, 'repr_type': 'bev_seg'},
        ]
    return types

_st = dict(model=None, sample=None, bev_grid=None, heatmaps=None,
           backend=None, backend_name='lss', raw_output=None)
_attr_cache = {}

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
server = FastAPI()

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
            _st['sample'] = load_sample(req.scene_idx, req.sample_idx)
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
    if backend is not None and hasattr(backend, 'name') and backend.name in ('tpvformer', 'gaussianformer', 'sparseocc'):
        from pipeline.backends.tpvformer_backend import NUSCENES_LIDARSEG_COLORS
        bev_class_colors = NUSCENES_LIDARSEG_COLORS

    bev_imgs = {}
    for m in ('argmax','class_heatmap','composite'):
        bev_imgs[m] = _pil_uri(render_bev(_st['bev_grid'], mode=m, target_class=0,
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

    return JSONResponse({
        'camera_images': cam_uris, 'intrinsics': intrinsics, 'extrinsics': extrinsics,
        'image_sizes': img_sizes, 'camera_names': list(CAMERA_NAMES),
        'bev_images': bev_imgs,
        'bev_info': {'grid_range':GRID_RANGE,'resolution':actual_resolution,'grid_cells':actual_grid_cells,
                     'class_names':model_classes,'cell_classes':cell_classes,'cell_confs':cell_confs},
        'repr_types': _get_repr_types(),
        'repr_type': backend.repr_type if backend else 'bev_seg',
        'has_3d': _st.get('raw_output') is not None and _st['raw_output'].ndim == 4,
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
    if backend is not None and hasattr(backend, 'name') and backend.name in ('tpvformer', 'gaussianformer', 'sparseocc'):
        from pipeline.backends.tpvformer_backend import NUSCENES_LIDARSEG_COLORS
        bev_class_colors = NUSCENES_LIDARSEG_COLORS
    img = render_bev(_st['bev_grid'], mode=req.mode, target_class=ci,
                     grid_range=GRID_RANGE, resolution=RESOLUTION,
                     class_names=model_classes, class_colors=bev_class_colors)
    return JSONResponse({'bev_image': _pil_uri(img, fmt='PNG')})

@server.post("/api/occupancy-3d")
async def api_occupancy_3d():
    """Return sparse voxel data for 3D viewer."""
    backend = _st.get('backend')
    raw = _st.get('raw_output')
    if backend is None or raw is None:
        return JSONResponse({'error': 'No model loaded or no 3D data'}, status_code=400)
    if raw.ndim != 4:
        return JSONResponse({'error': 'Model output is 2D, no 3D data available'}, status_code=400)

    logger.info("occupancy-3d: raw shape=%s, ndim=%d", raw.shape, raw.ndim)
    sparse = backend.get_sparse_voxels(raw)
    logger.info("occupancy-3d: %d voxels, %d classes, %d positions",
                sparse['num_voxels'], len(sparse.get('classes',[])), len(sparse.get('positions',[])))

    # Add camera frustum data for 3D visualization
    sample = _st.get('sample')
    frustums = []
    if sample:
        for ci in range(len(sample.get('images', []))):
            try:
                E = np.array(sample['ego_to_cameras'][ci])
                cam_from_ego = np.linalg.inv(E) if np.linalg.det(E) != 0 else E
                cam_pos = cam_from_ego[:3, 3].tolist()
                frustums.append({'cam_pos': cam_pos, 'name': CAMERA_NAMES[ci] if ci < len(CAMERA_NAMES) else f'CAM_{ci}'})
            except Exception:
                pass
    sparse['camera_frustums'] = frustums
    return JSONResponse(sparse)

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

# ══════════════════════════════════════════════════════════════════════════════
# FRONTEND
# ══════════════════════════════════════════════════════════════════════════════

FRONTEND_HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BEV Attribution Debug</title>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
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

/* 3D Occupancy viewer */
#occ3d-wrap{position:relative;overflow:hidden;border-radius:4px;display:none;background:#050505}
#occ3d-wrap canvas{display:block}
#occ3d-bar{position:absolute;bottom:6px;left:6px;right:6px;display:flex;gap:6px;align-items:center;z-index:10;font-size:9px;flex-wrap:wrap}
.occ3d-chip{padding:2px 7px;border-radius:3px;cursor:pointer;border:1px solid #333;font-size:9px;transition:all .15s;user-select:none}
.occ3d-chip.off{opacity:.3}
.occ3d-chip:hover{border-color:#fff}
#occ3d-stats{position:absolute;top:6px;right:8px;font-size:9px;color:#888;z-index:10}

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
</style>
</head><body>
<div id="app">
  <div id="hdr"><h1>BEV GRID — CAMERA ATTRIBUTION DEBUG</h1></div>

  <div id="ctrl">
    <div class="cg"><label>Scene</label><select id="sel-scene"></select></div>
    <div class="cg"><label>Sample</label><select id="sel-sample"></select></div>
    <div class="cg"><label>Model</label><select id="sel-repr"></select></div>
    <div class="cg"><label title="Target class for Class heatmap and GradCAM attribution">Class</label><select id="sel-class"></select></div>
    <div class="cg"><label>Method</label><select id="sel-method"></select><i class="info-icon" id="method-info">i</i></div>
    <button class="btn" id="btn-load">Load Scene</button>
    <button class="btn" id="btn-attr">Run Attribution</button>
    <button class="log-btn" id="log-toggle" title="Toggle Log Viewer">⌸ Log</button>
  </div>

  <div id="main">
    <div id="bev-panel">
      <div class="ph">
        <div class="tabs" id="bev-tabs">
          <span class="tab-label">2D:</span>
          <button class="tab on" data-m="argmax" data-tip="Per-cell class with highest logit, color-coded">Argmax</button>
          <button class="tab" data-m="class_heatmap" data-tip="Heatmap of selected class confidence across the grid">Class</button>
          <button class="tab" data-m="composite" data-tip="Alpha-blended overlay of all class confidences">Blend</button>
          <span class="tab-label" style="margin-left:6px">3D:</span>
          <button class="tab" data-m="3d" id="tab-3d" disabled data-tip="Interactive 3D occupancy voxel grid (sparse, top-k by confidence)">Occupancy</button>
        </div>
      </div>
      <div id="bev-wrap"><canvas id="bev-c" width="800" height="800"></canvas><div id="bev-tip"></div></div>
      <div id="occ3d-wrap"><div id="occ3d-stats"></div><div id="occ3d-bar"></div></div>
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
let CLASSES=['car','truck','bus','trailer','construction_vehicle','pedestrian','motorcycle','bicycle','traffic_cone','barrier'];
const METHODS=['GradCAM','Integrated Gradients','Attention','Occlusion'];

// ─── Populate selects ───────────────────────────────────────────────────────
const selScene=document.getElementById('sel-scene');
const selSample=document.getElementById('sel-sample');
const selClass=document.getElementById('sel-class');
const selMethod=document.getElementById('sel-method');
const selRepr=document.getElementById('sel-repr');
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

// Find nearest voxel along a camera ray in 3D mode
function raycastVoxel(ci, u, v) {
  if(!occ3dData || occ3dData.num_voxels === 0) return null;
  const ray = camPixelToRay(ci, u, v);
  if(!ray) return null;

  const pos = occ3dData.positions;
  const vs = occ3dData.voxel_size; // [dz, dy, dx]
  const halfX = vs[2] / 2, halfY = vs[1] / 2, halfZ = vs[0] / 2;
  let bestIdx = -1, bestT = Infinity;

  // Sample along ray, check each voxel for intersection
  // For efficiency, check all voxels against the ray (30K is manageable)
  for(let k = 0; k < occ3dData.num_voxels; k++) {
    const vx = pos[k*3], vy = pos[k*3+1], vz = pos[k*3+2]; // ego coords

    // AABB ray intersection: voxel centered at (vx, vy, vz) with half-extents
    const tMinX = ((vx - halfX) - ray.org[0]) / (ray.dir[0] || 1e-20);
    const tMaxX = ((vx + halfX) - ray.org[0]) / (ray.dir[0] || 1e-20);
    const tMinY = ((vy - halfY) - ray.org[1]) / (ray.dir[1] || 1e-20);
    const tMaxY = ((vy + halfY) - ray.org[1]) / (ray.dir[1] || 1e-20);
    const tMinZ = ((vz - halfZ) - ray.org[2]) / (ray.dir[2] || 1e-20);
    const tMaxZ = ((vz + halfZ) - ray.org[2]) / (ray.dir[2] || 1e-20);

    const tEnter = Math.max(Math.min(tMinX,tMaxX), Math.min(tMinY,tMaxY), Math.min(tMinZ,tMaxZ));
    const tExit  = Math.min(Math.max(tMinX,tMaxX), Math.max(tMinY,tMaxY), Math.max(tMinZ,tMaxZ));

    if(tEnter <= tExit && tExit > 0 && tEnter < bestT) {
      bestT = tEnter > 0 ? tEnter : 0;
      bestIdx = k;
    }
  }
  return bestIdx >= 0 ? {idx: bestIdx, depth: bestT} : null;
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
function drawBev(){
  ctx.clearRect(0,0,800,800);
  if(bevBg&&bevBg.complete)ctx.drawImage(bevBg,0,0,800,800);
  else{ctx.fillStyle='#0a0a0a';ctx.fillRect(0,0,800,800);}
  if(!D)return;
  const n=D.bev_info.grid_cells,c=800/n;
  if(hover&&(!sel||hover.i!==sel.i||hover.j!==sel.j)){ctx.strokeStyle='rgba(80,200,220,.45)';ctx.lineWidth=1.5;ctx.strokeRect(hover.j*c,hover.i*c,c,c);}
  if(sel){ctx.fillStyle='rgba(80,220,255,.25)';ctx.fillRect(sel.j*c,sel.i*c,c,c);ctx.strokeStyle='#5cf';ctx.lineWidth=2;ctx.strokeRect(sel.j*c,sel.i*c,c,c);}
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

      // 3D mode: raycast against voxels
      if(bevMode === '3d' && occ3dData && occ3d) {
        const vhit = raycastVoxel(ci, u, v);
        if(vhit) {
          const pos = occ3dData.positions;
          const eX = pos[vhit.idx*3], eY = pos[vhit.idx*3+1], eZ = pos[vhit.idx*3+2];
          const cls = occ3dData.classes[vhit.idx];
          const clsName = occ3dData.class_names[cls] || cls;
          info.textContent = `${CN[ci]} → voxel #${vhit.idx}: ${clsName} @ ego(${eX.toFixed(1)},${eY.toFixed(1)},${eZ.toFixed(1)}) d=${vhit.depth.toFixed(1)}m`;
          // Highlight voxel in 3D viewer (reuse the existing click handler logic)
          drawCams(null, [eX, eY, eZ]);
          // Add highlight sphere
          if(occ3d) {
            if(occ3d._highlightSphere) { occ3d.scene.remove(occ3d._highlightSphere); }
            const sGeo = new THREE.SphereGeometry(1.2, 12, 12);
            const sMat = new THREE.MeshBasicMaterial({color:0xffffff, wireframe:true, transparent:true, opacity:0.8});
            occ3d._highlightSphere = new THREE.Mesh(sGeo, sMat);
            occ3d._highlightSphere.position.set(-eY, eZ, -eX);
            occ3d.scene.add(occ3d._highlightSphere);
            document.getElementById('occ3d-stats').textContent = `Voxel #${vhit.idx}: ${clsName} @ ego(${eX.toFixed(1)},${eY.toFixed(1)},${eZ.toFixed(1)})`;
          }
        } else {
          info.textContent = `${CN[ci]} (${u.toFixed(0)},${v.toFixed(0)}) → no voxel hit`;
          drawCams(sel ? c2w(sel.i, sel.j) : null);
        }
        return;
      }

      // 2D mode: BEV ground intersection
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
  const is3d = bevMode === '3d';
  const bevWrap = document.getElementById('bev-wrap');
  const occ3dWrap = document.getElementById('occ3d-wrap');
  if(is3d){
    // Copy bev-wrap dimensions to occ3d-wrap before hiding bev-wrap
    const bRect = bevWrap.getBoundingClientRect();
    occ3dWrap.style.width = bRect.width + 'px';
    occ3dWrap.style.height = bRect.height + 'px';
  }
  bevWrap.style.display = is3d ? 'none' : '';
  occ3dWrap.style.display = is3d ? 'block' : 'none';
  if(is3d){
    load3DView();
  } else if(D&&D.bev_images&&D.bev_images[bevMode]){
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
    // Show/hide 3D tab based on model capability
    document.getElementById('tab-3d').disabled = !D.has_3d;
    // Reset to BEV view if switching away from 3D model
    if(!D.has_3d && bevMode==='3d'){
      bevMode='argmax';
      document.querySelectorAll('#bev-tabs .tab').forEach(x=>x.classList.remove('on'));
      document.querySelector('#bev-tabs .tab[data-m="argmax"]').classList.add('on');
      document.getElementById('bev-wrap').style.display='';
      document.getElementById('occ3d-wrap').style.display='none';
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

// ─── 3D Occupancy Viewer (three.js) ────────────────────────────────────────
let occ3d = null; // {scene, camera, renderer, controls, mesh, animId}
let occ3dData = null;
let occ3dClassVis = {}; // class index -> visible bool

function init3D() {
  if (occ3d) return occ3d;
  console.log('[3D] init3D called, THREE:', typeof THREE, 'OrbitControls:', typeof THREE.OrbitControls);
  const container = document.getElementById('occ3d-wrap');
  const w = container.clientWidth, h = container.clientHeight;
  console.log('[3D] init3D container:', w, 'x', h);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x050505);

  const camera = new THREE.PerspectiveCamera(50, w / h, 0.5, 200);
  camera.position.set(40, 30, 40);
  camera.lookAt(0, 0, 0);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setSize(w, h);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.insertBefore(renderer.domElement, container.firstChild);

  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.12;
  controls.target.set(0, 0, -1);
  controls.maxDistance = 150;

  // Lights
  scene.add(new THREE.AmbientLight(0x404040, 1.5));
  const dir = new THREE.DirectionalLight(0xffffff, 1.2);
  dir.position.set(30, 50, 20);
  scene.add(dir);

  // Ground grid — GridHelper lies on XZ plane by default (Y=up), which matches our mapping
  const grid = new THREE.GridHelper(102.4, 20, 0x1a3a3a, 0x111111);
  grid.position.set(0, -5, 0);  // Y=-5 = ego Z=-5m (ground level)
  scene.add(grid);

  // Ego marker (small arrow)
  const egoGeo = new THREE.ConeGeometry(0.8, 2, 4);
  const egoMat = new THREE.MeshLambertMaterial({ color: 0x00ccff });
  const ego = new THREE.Mesh(egoGeo, egoMat);
  ego.rotation.x = -Math.PI / 2;
  ego.position.set(0, 0, 0);
  scene.add(ego);

  // Axes: X=red(forward), Y=green(left), Z=blue(up)
  const axes = new THREE.AxesHelper(8);
  scene.add(axes);

  occ3d = { scene, camera, renderer, controls, mesh: null, animId: null, camLines: [] };

  // Click→camera backtracking
  setup3DClickHandler(occ3d);

  // Animate
  function animate() {
    occ3d.animId = requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  // Resize observer
  const ro = new ResizeObserver(() => {
    const cw = container.clientWidth, ch = container.clientHeight;
    if (cw > 0 && ch > 0) {
      camera.aspect = cw / ch;
      camera.updateProjectionMatrix();
      renderer.setSize(cw, ch);
    }
  });
  ro.observe(container);

  return occ3d;
}

function populate3DVoxels(data) {
  const o = init3D();
  occ3dData = data;

  // Remove old mesh
  if (o.mesh) { o.scene.remove(o.mesh); o.mesh.geometry.dispose(); o.mesh = null; }
  // Remove old camera lines
  o.camLines.forEach(l => o.scene.remove(l));
  o.camLines = [];

  if (!data || data.num_voxels === 0) {
    document.getElementById('occ3d-stats').textContent = '0 voxels';
    console.warn('[3D] No voxels to display');
    return;
  }

  const N = data.num_voxels;
  const pos = data.positions;
  const cls = data.classes;
  const colors = data.class_colors;
  const voxSize = data.voxel_size; // [dz, dy, dx]
  console.log('[3D] populate:', N, 'voxels, voxel_size:', voxSize, 'positions length:', pos.length);
  console.log('[3D] first 3 positions:', pos.slice(0, 9));
  console.log('[3D] class_colors type:', typeof colors, Array.isArray(colors) ? 'array len=' + colors.length : '');

  // Build class visibility toggles
  occ3dClassVis = {};
  const uniqueClasses = [...new Set(cls)].sort((a, b) => a - b);
  uniqueClasses.forEach(c => { occ3dClassVis[c] = true; });
  buildClassChips(data);

  // Create InstancedMesh
  const geo = new THREE.BoxGeometry(voxSize[2] * 0.92, voxSize[1] * 0.92, voxSize[0] * 0.92);
  const mat = new THREE.MeshLambertMaterial();
  const mesh = new THREE.InstancedMesh(geo, mat, N);

  const dummy = new THREE.Object3D();
  const color = new THREE.Color();

  for (let i = 0; i < N; i++) {
    const x = pos[i * 3], y = pos[i * 3 + 1], z = pos[i * 3 + 2];
    // Map: ego X(fwd) -> three.js Z(-), ego Y(left) -> three.js X(-), ego Z(up) -> three.js Y
    dummy.position.set(-y, z, -x);
    dummy.updateMatrix();
    mesh.setMatrixAt(i, dummy.matrix);

    const c = cls[i];
    const rgb = colors[c] || [128, 128, 128];
    color.setRGB(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255);
    mesh.setColorAt(i, color);
  }

  mesh.instanceMatrix.needsUpdate = true;
  mesh.instanceColor.needsUpdate = true;
  o.scene.add(mesh);
  o.mesh = mesh;
  console.log('[3D] mesh added to scene, children:', o.scene.children.length);
  console.log('[3D] camera pos:', o.camera.position.toArray(), 'target:', o.controls.target.toArray());

  // Camera frustums
  if (data.camera_frustums) {
    data.camera_frustums.forEach((cf, ci) => {
      const cp = cf.cam_pos;
      const pts = [new THREE.Vector3(-cp[1], cp[2], -cp[0]), new THREE.Vector3(0, 0, 0)];
      const lineGeo = new THREE.BufferGeometry().setFromPoints(pts);
      const lineColor = ci < CC.length ? new THREE.Color(CC[ci]) : new THREE.Color(0x888888);
      const lineMat = new THREE.LineBasicMaterial({ color: lineColor, opacity: 0.5, transparent: true });
      const line = new THREE.Line(lineGeo, lineMat);
      o.scene.add(line);
      o.camLines.push(line);
    });
  }

  document.getElementById('occ3d-stats').textContent = N.toLocaleString() + ' voxels';
}

// ─── 3D Click→Camera backtracking ───────────────────────────────────────────
function setup3DClickHandler(o) {
  const raycaster = new THREE.Raycaster();
  const mouse = new THREE.Vector2();
  let highlightSphere = null;

  o.renderer.domElement.addEventListener('click', (e) => {
    if (!o.mesh || !occ3dData) return;
    const rect = o.renderer.domElement.getBoundingClientRect();
    mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, o.camera);
    const hits = raycaster.intersectObject(o.mesh);
    if (hits.length === 0) {
      // Click on empty space — clear selection
      if (highlightSphere) { o.scene.remove(highlightSphere); highlightSphere = null; }
      document.getElementById('occ3d-stats').textContent = occ3dData.num_voxels.toLocaleString() + ' voxels';
      drawCams(null);
      return;
    }

    const idx = hits[0].instanceId;
    const pos = occ3dData.positions;
    const egoX = pos[idx * 3], egoY = pos[idx * 3 + 1], egoZ = pos[idx * 3 + 2];
    const cls = occ3dData.classes[idx];
    const conf = occ3dData.confidences[idx];
    const clsName = occ3dData.class_names[cls] || cls;

    console.log('[3D] Click voxel #' + idx + ': ego(' + egoX.toFixed(1) + ',' + egoY.toFixed(1) + ',' + egoZ.toFixed(1) + ') class=' + clsName);

    // Show highlight sphere at clicked voxel
    if (highlightSphere) o.scene.remove(highlightSphere);
    const sGeo = new THREE.SphereGeometry(1.2, 12, 12);
    const sMat = new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true, transparent: true, opacity: 0.8 });
    highlightSphere = new THREE.Mesh(sGeo, sMat);
    highlightSphere.position.set(-egoY, egoZ, -egoX); // ego→three.js mapping
    o.scene.add(highlightSphere);

    // Update stats
    document.getElementById('occ3d-stats').textContent =
      `Voxel #${idx}: ${clsName} (${conf.toFixed(2)}) @ ego(${egoX.toFixed(1)}, ${egoY.toFixed(1)}, ${egoZ.toFixed(1)})`;

    // Project to cameras — ego coords go directly to proj()
    drawCams(null, [egoX, egoY, egoZ]);
    layoutCams();
  });
}

function buildClassChips(data) {
  const bar = document.getElementById('occ3d-bar');
  bar.innerHTML = '';
  const counts = {};
  data.classes.forEach(c => { counts[c] = (counts[c] || 0) + 1; });
  const sorted = Object.keys(counts).map(Number).sort((a, b) => counts[b] - counts[a]);

  sorted.forEach(ci => {
    const chip = document.createElement('span');
    chip.className = 'occ3d-chip';
    const rgb = data.class_colors[ci] || [128, 128, 128];
    chip.style.background = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},0.3)`;
    chip.style.color = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
    chip.style.borderColor = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
    chip.textContent = (data.class_names[ci] || ci) + ' ' + counts[ci];
    chip.dataset.cls = ci;
    if (!occ3dClassVis[ci]) chip.classList.add('off');
    chip.addEventListener('click', () => {
      occ3dClassVis[ci] = !occ3dClassVis[ci];
      chip.classList.toggle('off');
      updateVoxelVisibility();
    });
    bar.appendChild(chip);
  });
}

function updateVoxelVisibility() {
  if (!occ3d || !occ3d.mesh || !occ3dData) return;
  const mesh = occ3d.mesh;
  const cls = occ3dData.classes;
  const N = occ3dData.num_voxels;
  const pos = occ3dData.positions;
  const dummy = new THREE.Object3D();

  let visCount = 0;
  for (let i = 0; i < N; i++) {
    const c = cls[i];
    if (occ3dClassVis[c]) {
      const x = pos[i * 3], y = pos[i * 3 + 1], z = pos[i * 3 + 2];
      dummy.position.set(-y, z, -x);
      dummy.scale.set(1, 1, 1);
    } else {
      dummy.position.set(0, -100, 0); // hide offscreen
      dummy.scale.set(0, 0, 0);
    }
    dummy.updateMatrix();
    mesh.setMatrixAt(i, dummy.matrix);
  }
  mesh.instanceMatrix.needsUpdate = true;
}

async function load3DView() {
  const container = document.getElementById('occ3d-wrap');
  const cw = container.clientWidth, ch = container.clientHeight;
  console.log('[3D] container size:', cw, 'x', ch);
  if (cw === 0 || ch === 0) {
    document.getElementById('occ3d-stats').textContent = 'Layout error (0x0)';
    console.error('[3D] Container has 0 dimensions');
    return;
  }
  const o = init3D();
  // Force resize to match container
  o.camera.aspect = cw / ch;
  o.camera.updateProjectionMatrix();
  o.renderer.setSize(cw, ch);

  document.getElementById('occ3d-stats').textContent = 'Loading...';
  try {
    const r = await fetch('/api/occupancy-3d', { method: 'POST' });
    console.log('[3D] API status:', r.status);
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      document.getElementById('occ3d-stats').textContent = err.error || 'No 3D data';
      return;
    }
    const data = await r.json();
    console.log('[3D] Received:', data.num_voxels, 'voxels,', [...new Set(data.classes)].length, 'classes');
    populate3DVoxels(data);
  } catch (e) {
    document.getElementById('occ3d-stats').textContent = 'Error: ' + e.message;
    console.error('[3D] Error:', e);
  }
}

</script>
</body></html>
"""

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Starting BEV Attribution Debug Tool on http://0.0.0.0:7860 …", flush=True)
    uvicorn.run(server, host='0.0.0.0', port=7860)
