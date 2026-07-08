# Implementation Plan: Neuronpedia-Grade VLM Attribution Dashboard

**Status:** approved plan, not yet started
**Branch:** `claude/git-lfs-skip-smudge-pqsim2` (develop and push here; do NOT open a PR unless asked)
**Benchmark:** match the *experience quality* of https://www.neuronpedia.org/qwen3.6-27b —
hover-preview/click-lock interaction, per-layer/head drill-down, multiple lenses,
quantitative evals, interventions, permalinks — on this repo's SmolVLM-256M stack.
We cannot match model scale (27B vs 256M); we match interaction grammar and polish.

---

## 1. Context: what this repo is

FastAPI app (HF Space, Docker, CPU-only) for attribution/interpretability of
autonomous-driving perception models. Three existing modes, switched by tabs
(`#mode-tabs`, `app.py` ~line 940, `data-mode="bev"|"vlm"|"vb"`):

1. **BEV Attribution** — LSS (Lift-Splat-Shoot) model, 6 nuScenes cameras → BEV grid;
   click a BEV cell → GradCAM/IG/Attention/Occlusion heatmaps on cameras
   (methods in `attribution/`, model in `pipeline/lss_model.py`, backend in
   `pipeline/backends/lss_backend.py`).
2. **VLM Reasoning** — SmolVLM-256M-Instruct describes a camera image; generated
   words are color-coded by attention strength and clickable; click word →
   image heatmap of where the model attended. **This is the mode being rebuilt.**
3. **VLM→BEV** — projects VLM front-camera attention into the BEV grid.

### Key files and anchors (verified against current HEAD)

| File | Lines | Role |
|---|---|---|
| `app.py` | 2123 | EVERYTHING frontend + API. Inline HTML from ~line 790, CSS inside, JS from ~line 1100. FastAPI `server`, endpoints listed below. |
| `pipeline/vlm/model.py` | 550 | `VLMRunner` class (line 57): `load()` (81), `generate()` (119), `compute_word_attentions()` (296), `get_word_heatmap()` (407), `_get_attention_heatmap()` (413), `_find_image_positions()` (238), `_attn_to_heatmap()` (251), `_group_tokens_into_words()` (265), `get_cam_heatmap()` (218, GradCAM path). |
| `pipeline/vlm/targets.py` | 28 | `VLMTokenTarget` — pytorch-grad-cam target extracting one token's logit (teacher-forced). |
| `pipeline/vlm/wrapper.py` | 45 | grad-cam model wrapper. |
| `attribution/` | 1222 total | BEV-side methods (Captum IG/GradCAM/Occlusion + attention hooks). Reuse patterns here for VLM-side IG/occlusion. |
| `viz/camera.py` | — | `render_camera(image, heatmap=..., camera_name=...)` → PIL overlay. Reuse for all new heatmaps. |
| `data/samples/CAM_*/` | 12 jpgs | 2 images × 6 cameras (nuScenes mini). LFS-tracked. |
| `checkpoints/lss_model.pt` | LFS | LSS weights. SmolVLM weights are downloaded at Docker build (see `Dockerfile`) to `/app/checkpoints/smolvlm-256m`; locally `VLMRunner.load()` falls back to the HF hub id. |

### Existing VLM API endpoints (app.py line numbers)

- `POST /api/vlm/load` (427) — lazy-load SmolVLM.
- `POST /api/vlm/generate` (443) — generate + `compute_word_attentions()`; returns `words[{text,strength}]`, `image_uri`.
- `POST /api/vlm/attention` (481) — per-token heatmap (legacy).
- `POST /api/vlm/word-attention` (515) — per-word heatmap overlay (the current click handler target).
- `GET /api/vlm/images` (352), `GET /api/vlm/image/{camera}/{filename}` (364), `GET /api/vlm/debug` (386).
- VLM→BEV: `POST /api/vlm-bev/run|word|click` (548/667/720).

### Module-level state in app.py

`_vlm_st = dict(runner=None)`, `_vlm_attr_cache = {}` (line ~78), `VLM_METHODS`
dict (line ~81) currently exposes only `'Attention'` — GradCAM intentionally
hidden. `_pil_uri()` (line ~88) converts PIL → base64 data URI (all images ship
as data URIs; there is no static file serving today).

### Model facts (SmolVLM-256M-Instruct)

- Idefics3 architecture; loaded via `AutoModelForVision2Seq`, `_attn_implementation="eager"`
  (**required** — attentions must be materialized; do not change).
- Vision: SigLIP, 512×512, `do_image_splitting=False` → exactly **64 image tokens (8×8 grid)**.
  `_find_image_positions()` finds them by token-id frequency (count ≥ 60).
- LM: ~30 layers (config: `model.config.text_config.num_hidden_layers`),
  ~9 heads (`num_attention_heads`), hidden 576. **Read L and H from config at
  runtime; never hardcode.**
- CPU inference runs in float32 (`compute_word_attentions` moves model to CPU
  float32 then back — see `model.py:323` and the `finally` at 367). Keep this pattern.

### Environment constraints

- HF Space: CPU-only, 2 vCPU / 16 GB. A full teacher-forced forward with
  `output_attentions=True` takes a few seconds; generation ~10–30 s. Design all
  UX around "compute once, cache, then instant".
- `requirements.txt` pins `transformers>=4.45.0,<5.0.0` (5.x breaks Idefics3
  processor auto-detect). **Do not bump to 5.x.**
- Git LFS is configured on GitHub for this repo (`.gitattributes` covers
  `*.pt`, `*.safetensors`, sample jpgs, nuScenes json). New binary artifacts →
  add LFS patterns + `git lfs push` happens automatically once tracked.
- Frontend must remain dependency-light: vendor libraries into the repo
  (no npm build step). The Docker CMD is `python3 app.py`; keep that working.

---

## 2. Target architecture (after Phase 0)

```
app.py                    # FastAPI app + endpoints ONLY (~600 lines)
static/
  index.html              # extracted from app.py HTML_PAGE
  css/app.css             # extracted styles
  js/core.js              # tabs, status bar, log drawer, shared helpers (fetchJSON, etc.)
  js/bev.js               # existing BEV mode JS (extracted, unchanged behavior)
  js/vlm.js               # VLM mode (rebuilt in Phase 1-2)
  js/vlmbev.js            # VLM→BEV mode JS (extracted, unchanged)
  js/jspace.js            # Phase 3
  vendor/d3.v7.min.js     # vendored (no CDN dependency)
pipeline/vlm/model.py     # VLMRunner (extended: full attention tensor, logprobs)
pipeline/vlm/methods.py   # NEW: gradcam/grad×input/IG/occlusion for VLM (Phase 2)
pipeline/vlm/evals.py     # NEW: deletion/insertion faithfulness (Phase 2)
pipeline/jlens_runner.py  # NEW: Phase 3
server/cache.py           # NEW: disk artifact cache (Phase 0)
scripts/fit_jlens.py      # NEW: offline lens fitting (Phase 3)
scripts/precompute_gallery.py  # NEW: Phase 5
data/cache/               # runtime artifact cache (gitignored)
data/gallery/             # precomputed curated results (committed, small JSON + jpg)
checkpoints/jlens/lens.pt # fitted Jacobian lens (LFS)
```

Serving: `server.mount("/static", StaticFiles(directory="static"))`; `GET /`
returns `static/index.html` contents. Keep every existing endpoint path and
response shape unchanged — BEV and VLM→BEV modes must keep working untouched.

---

## 3. Phases

Work strictly in order. Each phase ends with: manual verification (see §5),
a commit (small, descriptive), and a push. The app must be runnable after every
phase — never leave `main` broken across a push.

---

### Phase 0 — Foundation refactor + artifact cache (~1 day)

**P0.1 Extract frontend.** Move the HTML string in `app.py` (from `HTML_PAGE`/
`@server.get("/")` at line 162 through end of the JS) into `static/index.html`,
`static/css/app.css`, and the four `static/js/*.js` modules. Mechanical split —
byte-identical behavior. The JS is plain ES5/6 with global functions wired by
`onclick`/`addEventListener`; keep globals for now (no bundler).

**P0.2 Static serving.** `from fastapi.staticfiles import StaticFiles`;
mount `/static`; `GET /` reads `static/index.html` (read at request time in dev,
fine to keep simple). Update `Dockerfile` `COPY . .` already covers it — verify.

**P0.3 Disk cache (`server/cache.py`).**
```python
def cache_key(**kwargs) -> str      # sha256 of sorted-json kwargs, first 24 hex
def cache_get(namespace, key)       # returns deserialized JSON/npz or None
def cache_put(namespace, key, obj)  # JSON for dicts, .npz for numpy arrays
```
Root `data/cache/{namespace}/{key}.{json|npz}`. Add `data/cache/` to `.gitignore`.
Wire into `/api/vlm/word-attention` first (namespace `vlm_word`, key from
image path + prompt + method + word_index) to prove it works. All later phases
must use this instead of the in-memory `_vlm_attr_cache` (keep in-memory as L1).

**P0.4 Vendor d3** into `static/vendor/d3.v7.min.js` (download once, commit;
it's ~280 KB, plain text, no LFS needed).

**Done when:** app boots (`python3 app.py`), all three modes behave exactly as
before, `data/cache/` fills up on word clicks, second click is instant, Docker
build still passes (or at minimum `python3 -c "import app"` succeeds and a local
uvicorn run serves `/` with working static assets).

---

### Phase 1 — Token Lens: hover-preview / click-lock + layer/head drill-down (~2 days)

**The core UX upgrade.** Neuronpedia's grammar: *hover = instant preview,
click = lock + detail panel*. All data for this exists in one forward pass and
is currently thrown away by the `.mean()` calls in `compute_word_attentions()`
(`model.py:337` averages heads, `:357` averages layers).

**P1.1 Backend: keep the full tensor.** In `compute_word_attentions()`:
- After stacking, keep `full = torch.stack([a[0] for a in outputs.attentions])`
  → `[L, H, S, S]` (do NOT mean over heads yet). Memory check: S ≈ 64 img + ~40
  prompt + ~128 gen ≈ 230; L=30, H=9 → 30·9·230·230 ≈ 14.3M floats ≈ 57 MB fp32.
  Fine on 16 GB, but **slice immediately**: for each generated token position,
  extract `full[:, :, target_pos, image_positions]` → `[L, H, 64]` and discard
  the rest. Store per-word (max over subword tokens, matching existing logic)
  as `self._word_lh_attn: list[np.ndarray [L,H,64]]` (~50 words × 69 KB ≈ 3.5 MB).
- Existing avg/rollout outputs stay unchanged (computed from the same stack).

**P1.2 Backend: token alternatives.** The same teacher-forced forward already
returns logits when asked. Add `output_hidden_states=False`, capture `outputs.logits`
`[1, S, V]`, and for each generated token store: its logprob, its rank, and
top-5 alternative tokens `{text, prob}` at that position. Store as
`self._token_stats: list[dict]` aligned with `self._tokens`. (Logits for S=230,
V≈49k → 45 MB transient; slice to generated positions immediately.)

**P1.3 New endpoint** `POST /api/vlm/word-detail` `{word_index}` →
```json
{
  "word": "car", "strength": 0.83,
  "layer_profile": [0.1, ...],              // L floats: mean over heads+patches per layer
  "head_matrix": [[0.2, ...], ...],         // L×H: mean over patches
  "token_stats": [{"text":" car","logprob":-0.4,"rank":1,
                   "alts":[{"t":" truck","p":0.12}, ...]}],
  "heatmap_lh": null                        // omitted here; see /word-heatmap-lh
}
```
`POST /api/vlm/word-heatmap-lh` `{word_index, layer, head}` → overlay data-URI for
that (layer, head) slice (`layer=-1` or `head=-1` = mean). Cache both (namespace
`vlm_detail`, `vlm_lh`).

**P1.4 Frontend rebuild of `static/js/vlm.js` word strip + detail panel.**
- After generate, prefetch all word heatmaps (mean view) in the background
  (sequential fetch loop, they're cached server-side) so **hover** swaps the
  image instantly with zero latency. Hover off → restore locked selection
  (or base image if nothing locked).
- **Click word → lock** (highlight with outline; second click unlocks). Locked
  word opens a detail panel below the image (or right side, reuse `#vlm-sidebar`
  layout) containing:
  1. **Layer sparkline** (d3): x = layer 0..L-1, y = layer_profile. Click a
     layer → filters heatmap to that layer (calls `/word-heatmap-lh`).
  2. **L×H head grid** (d3, each cell ~10 px, viridis scale): click cell →
     heatmap for that exact layer/head. Hovered cell shows tooltip `L12 H3 0.42`.
     A "mean" reset button clears the filter.
  3. **Token stats row**: for each subword — logprob bar, rank badge, expandable
     top-5 alternatives.
- Keyboard: `←`/`→` moves locked word, `Esc` unlocks. (Foundations for Phase 5.)

**Done when:** hover over any word previews its heatmap with no visible delay
(after prefetch completes); clicking locks and shows sparkline + head grid;
clicking a head-grid cell visibly changes the heatmap (verify heads differ —
if all heads look identical something is wrong with slicing); token alternatives
render with sane probabilities.

---

### Phase 2 — Multi-lens + faithfulness evals (~2 days)

**P2.1 `pipeline/vlm/methods.py`** — unified interface:
```python
def compute_heatmap(runner, word_index, method, **params) -> np.ndarray  # [H,W] in [0,1]
```
Methods (each returns image-resolution heatmap; reuse `runner._word_lh_attn`
where possible):
- `attention_mean` — existing (delegate).
- `attention_layer_head` — existing Phase 1 path.
- `gradcam` — already implemented in `model.py:_get_gradcam_heatmap` via
  pytorch-grad-cam + `VLMTokenTarget`; un-hide it. Aggregate subwords by max.
- `gradient_x_input` — teacher-forced forward, `loss = sum(logit of word's
  token ids at their positions)`, backward to `pixel_values.grad`,
  `|grad × input|` summed over channels → resize. Mirror the pattern in
  `attribution/attention.py:_gradient_input_fallback`.
- `integrated_gradients` — Captum on `pixel_values` with black baseline,
  n_steps=8 default (CPU budget), mirror `attribution/integrated_gradients.py`.
  Mark "slow (~1 min)" in UI.
- `occlusion_patches` — operate at the **vision-token level, not pixels**: for
  each of the 64 image patches, mask its 64×64 px region (grey), single forward,
  measure drop in the word's summed logprob → 8×8 importance grid → upsample.
  64 forwards ≈ 1.5–3 min CPU. Mark "slow". Cache aggressively.

Update `VLM_METHODS` (app.py:81) to expose all; add a method `<select>` in the
VLM controls. Extend `/api/vlm/word-attention` with a `method` and `params` field
(default `attention_mean` — backward compatible).

**P2.2 Comparison view.** "Compare" toggle in VLM mode: renders a 2×3 grid of
the locked word's heatmap across all methods (each tile labeled, lazy-loaded,
slow methods show a run button instead of auto-computing). Endpoint reuse —
frontend just fans out N requests.

**P2.3 `pipeline/vlm/evals.py` — deletion/insertion faithfulness.**
```python
def faithfulness_curve(runner, word_index, heatmap, mode) -> dict
# mode: 'deletion' | 'insertion'
# steps: mask top-{0,10,...,100}% patches (11 forwards),
# metric: sum of word's token logprobs (teacher-forced)
# returns {'fractions': [...], 'scores': [...], 'auc': float}
```
Patch ranking from the heatmap downsampled to the 8×8 grid. Deletion masks the
most-important patches first (score should FALL fast for a faithful method);
insertion starts fully-masked and reveals (score should RISE fast).
`POST /api/vlm/faithfulness` `{word_index, method}` → both curves + AUCs. ~22
forwards ≈ 30–60 s CPU; run behind an explicit "Evaluate" button with progress
via the existing log drawer; cache (namespace `vlm_eval`).
- UI: in comparison view, an "Evaluate methods" button that runs curves for the
  cheap methods and renders a d3 line chart (deletion + insertion per method)
  with an AUC leaderboard table. **This is the honest-ranking feature — it tells
  the user which lens to trust.**

**Done when:** all 6 methods produce visually distinct, plausible heatmaps on
`CAM_FRONT` sample images; comparison grid renders; faithfulness chart shows
attention/gradcam ordering with deletion-AUC clearly separated from a random
baseline (add a `random` method as sanity control — it must score worst; if it
doesn't, the eval is buggy, not the methods).

---

### Phase 3 — J-lens page (~2 days)

Reproduce the Jacobian-lens interaction from the paper
(https://transformer-circuits.pub/2026/workspace/index.html) using the official
Apache-2.0 companion repo **github.com/anthropics/jacobian-lens** (`jlens`
package). Key API (verified from repo README):
```python
model = jlens.from_hf(hf_model, tokenizer)
lens  = jlens.fit(model, prompts=prompts, checkpoint_path=...)   # backward-pass heavy
lens.save("lens.pt"); lens = jlens.JacobianLens.from_pretrained(...)
lens_logits, model_logits, _ = lens.apply(model, text, positions=[...])
# lens_logits: {layer: logits}; readout = unembed(J_l @ h_l)
```

**P3.1 Offline fitting (`scripts/fit_jlens.py`).** Target: **SmolVLM's language
model** (`runner.model.model.text_model` — verify exact attribute path at
runtime; it's the Idefics3/SmolVLM text backbone) with the model's `lm_head` as
unembed, wrapped via `jlens.from_hf`. If the VLM backbone proves incompatible
with `jlens.from_hf` (it expects a plain `AutoModelForCausalLM`), fall back to
**`HuggingFaceTB/SmolLM2-135M-Instruct`** — the same LM family SmolVLM was built
from — and note the substitution in the UI. Prompts: use the repo's
`data/experiments` sets, or 100–200 sequences of generic web text; per the repo,
~100 prompts is usable. Fit on CPU is slow but feasible for a 135M-scale LM
(hours, run once); GPU if available. Output → `checkpoints/jlens/lens.pt`,
tracked by LFS (`.gitattributes` already covers `*.pt`), committed.

**P3.2 `pipeline/jlens_runner.py`.** Loads `lens.pt` once; exposes:
```python
def run(text) -> {"tokens": [...], "grid": [[{"w": "car", "r": 3}, ...], ...]}
    # grid[layer][position] = top-1 word + its rank of the model's actual next token
def pin(text, position, layer) -> {"topk": [...], "rank_track": [...]}
    # rank_track: rank of the pinned token across all layers at that position
```
Applying the lens is matmul-cheap; the forward pass to get `h_l` dominates
(seconds). Cache by text hash.

**P3.3 Endpoints + page.** `POST /api/jspace/run`, `POST /api/jspace/pin`.
New tab `data-mode="jspace"` + `static/js/jspace.js`:
- Text input (plus a "use VLM output" button that pulls the current VLM
  generation — the multimodal twist: *watch the workspace while the model
  describes a driving scene*).
- **Layer × position grid** (d3 table): each cell = lens top-1 word with rank
  superscript; bottom row = the model's actual output tokens. Color cells by
  agreement-with-final-output. Hover → highlight row/col; **click → pin**:
  right panel shows rank-vs-layer line chart for the pinned token + top-k list
  for that cell. Mirrors the paper's/Neuronpedia's slice-view interaction.

**Done when:** entering "Fact: the capital of the state containing Dallas is"
shows intermediate-layer cells resolving toward " Austin" before the final
layer does (the paper's canonical multi-hop demo); pin interaction works; VLM
handoff button populates the grid from a generated description.

---

### Phase 4 — Interventions (~2 days)

Convert from observational to causal.

**P4.1 Region knockout (pixel-space MVP).** UI: drag-select a rectangle on the
VLM image (or click patches on an 8×8 grid overlay). Backend
`POST /api/vlm/knockout` `{camera, filename, prompt, patches:[int]}`:
grey/blur the selected patch regions of the input image, re-run `generate()`
with identical decoding, return both texts. Frontend renders a **word-level
diff** (simple LCS diff, highlight added/removed words) — "what would it have
said without this region". ~10–30 s; show progress.

**P4.2 Vision-token knockout (embedding-space, stronger).** Same endpoint with
`mode: "embed"`: hook `inputs_embeds` (build embeds manually: run vision tower,
splice into text embeds — or simpler: forward with `pixel_values` but register a
forward hook on the connector/`model.model.connector` output zeroing selected
token rows). Compare with pixel-space result. If the embedding surgery gets
brittle, ship pixel-space only and log a TODO — do not sink >½ day here.

**P4.3 Attention steering (stretch, skip if behind schedule).** Multiply
attention scores toward selected patches by α∈[0,4] via attention-module
forward hooks during generation; show text change. Eager attention makes this
possible; still fiddly across 30 layers — timebox to ½ day.

**Done when:** knocking out the patch region over a described vehicle changes
the generated description (e.g. the vehicle disappears from the text), and the
diff view makes the change obvious in <3 s of reading.

---

### Phase 5 — Platform polish (~1–2 days)

**P5.1 Permalinks.** Serialize UI state to URL hash:
`#m=vlm&cam=CAM_FRONT&f=<filename>&p=<b64 prompt>&mtd=attention_mean&w=7&L=12&H=3`.
On load, restore state (auto-load model, auto-generate if `p` present — show a
"restoring shared view…" status). Add a "Copy link" button. Same for jspace mode.

**P5.2 Gallery.** `scripts/precompute_gallery.py`: for ~8 curated
(image, prompt) pairs, precompute generation + word attentions + mean heatmaps +
faithfulness AUCs into `data/gallery/*.json` (+ thumbnail jpgs). Landing panel in
VLM mode shows gallery cards; clicking one loads instantly from the precomputed
artifact (no model needed) — this is what makes the Space feel alive in the
first 5 seconds, before the model loads. Commit gallery artifacts (small JSON;
keep total <5 MB, no LFS needed for JSON).

**P5.3 Export + keyboard + design pass.**
- "Export PNG" (current composite view via canvas) and "Export JSON" (raw
  heatmap arrays + stats) buttons.
- Keyboard: `←/→` word nav, `1..6` method switch, `Esc` unlock, `?` help overlay.
- Design pass: consistent 8px spacing grid, one accent color, hover states on
  every interactive element, empty-states with hints ("Generate to begin"),
  loading skeletons instead of blank panels. Keep the existing dark palette
  (#111 bg, #1a1a1a panels, #333 borders) — tighten, don't redesign.
- Update `README.md` (HF Space card) with the new modes + screenshots; update
  `TODO.md`/`PROJECT_PLAN.md` to reflect shipped state.

---

## 4. API summary (new/changed)

| Endpoint | Phase | Req | Resp (essentials) |
|---|---|---|---|
| `POST /api/vlm/word-detail` | 1 | `{word_index}` | `layer_profile[L]`, `head_matrix[L][H]`, `token_stats[]` |
| `POST /api/vlm/word-heatmap-lh` | 1 | `{word_index, layer, head}` | `{heatmap: dataURI}` |
| `POST /api/vlm/word-attention` | 2 (ext) | `+{method, params}` | unchanged shape |
| `POST /api/vlm/faithfulness` | 2 | `{word_index, method}` | `{deletion:{fractions,scores,auc}, insertion:{...}}` |
| `POST /api/jspace/run` | 3 | `{text}` | `{tokens[], grid[L][P]{w,r}}` |
| `POST /api/jspace/pin` | 3 | `{text, position, layer}` | `{topk[], rank_track[L]}` |
| `POST /api/vlm/knockout` | 4 | `{camera, filename, prompt, patches[], mode}` | `{text_before, text_after, diff[]}` |

All responses include `{"error": str}` on failure (existing convention).
All heavy endpoints must go through `server/cache.py`.

---

## 5. Verification protocol (every phase)

1. `python3 -c "import app"` — no import errors.
2. Run `python3 app.py`, open `/`, exercise **all three legacy modes** (BEV scene
   load + cell click; VLM generate + word click; VLM→BEV run) — zero regressions.
3. Exercise the phase's new feature end-to-end on `CAM_FRONT` sample image with
   prompt "Describe the color of each vehicle in this image."
4. Restart the server; verify cached artifacts make repeat interactions instant.
5. If `Dockerfile` inputs changed, verify the build (or at minimum that
   requirements/paths referenced exist).
6. Commit with a message describing behavior (not implementation), push:
   `git push -u origin claude/git-lfs-skip-smudge-pqsim2` (retry ×4, backoff 2/4/8/16 s).

## 6. Known pitfalls (read before coding)

- **transformers must stay <5.0** (Idefics3 processor auto-detect breaks; see
  Dockerfile patch of `preprocessor_config.json`).
- **eager attention is load-bearing** for `output_attentions=True` and grad-cam.
- `compute_word_attentions` flips the model to CPU/float32 and back
  (`model.py:323, 367-372`) — new forward-pass code must follow the same pattern
  or MPS/bf16 states corrupt.
- Image-token positions come from `_find_image_positions()` frequency heuristic —
  reuse it, don't re-derive.
- `do_image_splitting=False` is what guarantees 64 image tokens; changing it
  breaks every 8×8 assumption.
- Attention tensor slicing (P1.1): slice per-target-position immediately; holding
  `[L,H,S,S]` for long prompts will OOM the 16 GB Space when S grows.
- Data URIs, not static files, for generated images (matches existing frontend).
- The frontend has no build step — plain JS, globals, `onclick` wiring. Keep it.
- LFS: GitHub rejects pushes referencing LFS objects it doesn't have (GH008).
  New LFS files must be added while `git-lfs` is installed (`apt-get install git-lfs`,
  `git lfs install --local`) so upload happens on push.
- Do not create a PR; do not push to `main`.

## 7. Suggested schedule / commit granularity

| Day | Work | Commits |
|---|---|---|
| 1 | Phase 0 | `refactor: extract frontend to static/`, `feat: disk artifact cache` |
| 2–3 | Phase 1 | `feat: full L×H attention capture`, `feat: word detail endpoint`, `feat: hover-preview + click-lock UI`, `feat: layer/head drill-down` |
| 4–5 | Phase 2 | `feat: VLM method suite`, `feat: comparison grid`, `feat: faithfulness evals` |
| 6–7 | Phase 3 | `feat: jlens fitting script + lens.pt`, `feat: j-space grid page` |
| 8–9 | Phase 4 | `feat: region knockout + text diff`, (`feat: embedding knockout`) |
| 10 | Phase 5 | `feat: permalinks`, `feat: gallery`, `polish: keyboard/export/design` |

Phases 0–2 are the mandatory core. If time is cut, ship 0–2 fully rather than
0–5 partially.
