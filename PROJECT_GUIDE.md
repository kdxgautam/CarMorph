# PPP Car Paint project guide

This guide explains how the repository fits together. Use [README.md](README.md)
for installation, environment variables, and API examples; use this document
when reading, debugging, or extending the code.

## What the project does

The application accepts one car photograph, determines its visible view,
segments the vehicle and its parts, identifies which pixels represent editable
paint, and renders a new body colour without changing protected parts or the
background.

The core guarantee is:

```text
changed pixels ⊆ requested editable mask
editable mask ∩ protected mask = ∅
background and protected pixels remain identical to the original
```

There are two entry points:

- `app/main.py`: FastAPI service.
- `streamlit_app.py`: interactive editor and evaluation gallery.

Both use the same processing and rendering code.

## Recommended reading order

Read these files in order for the shortest path through the system:

1. `app/schemas.py` — persisted asset metadata and view types.
2. `app/config.py` — environment settings, part aliases, and view requirements.
3. `app/pipeline.py` — the complete asset-preparation workflow.
4. `app/detection.py` — local car/part detection and automatic view selection.
5. `app/roboflow.py` — SAM2/SAM3 requests and response validation.
6. `app/paint_analysis/paint_group_classifier.py` — material and paint-group
   decisions.
7. `app/paint_analysis/surface_completion.py` — region growth across coherent
   painted panels.
8. `app/paint_analysis/mask_builder.py` — final editable/protected/uncertain
   mask construction.
9. `app/image_ops.py` — mask utilities and LAB recolouring.
10. `app/renderers/deterministic.py` — deterministic render orchestration and
    quality enforcement.
11. `app/quality/checks.py` — mask and pixel invariants.
12. `app/main.py` or `streamlit_app.py` — public interface behavior.

## End-to-end processing flow

```text
Upload image
    │
    ├─ Validate format, size, orientation, and dimensions
    │
    ├─ Detect primary car with YOLO-World
    │
    ├─ Detect detailed parts with the local car-parts model
    │     └─ Resolve auto view from raw labels and confidence
    │
    ├─ Refine geometry
    │     ├─ SAM2: full car and detector-box masks
    │     └─ SAM3: handles, mirrors, pillars, lights, and trim
    │
    ├─ Build raw semantic part masks
    │
    ├─ Estimate the dominant body-paint LAB profile
    │
    ├─ Classify semantic/material paint groups
    │     ├─ Hard protection first
    │     ├─ Painted handles, mirror caps, roof, and bumper
    │     └─ Unknown regions remain conservative
    │
    ├─ Complete the main painted surface
    │     ├─ Strict paint seeds
    │     ├─ Constrained local region growth
    │     └─ Connected-region voting and bounded gap filling
    │
    ├─ Persist masks, diagnostics, reports, and metadata
    │
    └─ Render requested colour
          ├─ Preserve source luminance and reflections in LAB
          ├─ Blend only inside the editable mask
          └─ Verify protected/background pixels are unchanged
```

## Detection and automatic view selection

`app/detection.py` performs one local detection pass and preserves raw labels
and confidences. `infer_view()` scores evidence for four cardinal views:

- Front: grille/hood, windshield, front bumper, headlights.
- Rear: trunk, rear windshield, rear bumper, tail lights.
- Left/right: corresponding doors, windows, wheels, mirrors, and panels.

Three-quarter images resolve to their dominant front or rear face when that
evidence is sufficiently close to the side evidence. Weak or conflicting
evidence raises `ambiguous_view`; the caller can then select a manual view.

Door detections also create clipped upper-door prompts for side glass. This is
done even when a three-quarter photograph resolves to `front` or `rear`.

## Hybrid segmentation

The default `ROBOFLOW_SEGMENTER=hybrid` splits responsibilities:

- SAM2 provides stable full-car and box-prompt geometry.
- SAM3 adds zero-shot semantic details that the local model may miss.
- Local detector polygons are reused directly when available.

Every part mask is clipped to the selected full-car mask. Empty optional
results are allowed; missing required view-specific regions fail with a stable
`PipelineError` code.

`sam2` and `sam3` remain available as A/B or rollback modes.

## Paint analysis mental model

Paint analysis follows precedence, not colour alone:

```text
semantic protection
    > confident material protection
    > separately editable contrast paint
    > main-body paint recovery
    > uncertainty
```

This matters because the same RGB colour can appear in glass reflections,
chrome, painted metal, and plastic. Conversely, one painted panel can contain
very different RGB values because of sunlight, shadow, curvature, and glare.

### 1. Body-paint profile

`body_colour_estimator.py` erodes a conservative body candidate away from part
boundaries, removes extreme lightness samples, clusters LAB chroma, and records
a robust median profile. LAB chroma relates paint across lighting changes while
the L channel represents photographed luminance.

### 2. Material and semantic classification

`material_classifier.py` gives reliable semantic parts such as glass, lights,
wheels, grille, and pillars priority over appearance. Other regions use simple
appearance statistics to distinguish probable paint, chrome, and dark plastic.

`paint_group_classifier.py` then assigns disjoint paint groups. Its internal
`claimed` mask makes the first, strongest decision win.

### 3. Surface completion

Strict paint-compatible pixels become seeds, not the final mask.
`surface_completion.py` grows them only through:

- The safe body candidate.
- Globally plausible body chroma.
- Locally coherent LAB neighborhoods.
- Acceptable image gradients.
- Pixels outside hard protection.

Connected-region voting recovers panel shadows, highlights, thin edge strips,
and reflection-shaped gaps without globally expanding into protected parts.

### 4. Final masks

`mask_builder.py` creates three disjoint masks with fixed precedence:

1. `protected-mask.png`
2. `uncertain-mask.png`
3. `editable-mask.png`

`paintable-body.png` is a backward-compatible alias of the editable mask for
new assets.

## Deterministic rendering

`image_ops.recolour()` performs LAB paint transfer:

- The source L channel retains panel geometry, seams, shadows, and glare.
- The requested colour supplies the new base chroma.
- Local chroma residuals retain environmental reflections.
- Shadow pixels keep a minimum target-colour strength.
- Specular highlights progressively lose saturation.
- Finish settings adjust reflection and highlight strength.
- Feathering occurs only inside the editable mask.

`DeterministicSurfaceRenderer` loads the request-specific masks, applies body
and optional roof colours, checks invariants, then writes the result atomically.

## Generative rendering

Generative rendering is optional. Structured requests become constrained
prompts in `app/modifications/prompts.py`; raw user text is never passed through
unchecked. Even after generation, the original image is restored outside the
editable mask before quality checks run.

Use deterministic rendering for plain body colour and finish changes. Use the
generative path only for supported surface designs that require it.

## Stored asset layout

Each prepared image is immutable and stored under:

```text
data/processed/<asset_id>/
├── metadata.json
├── original.webp
├── source.jpg
├── luminance-map.png
├── body-paint-profile.json
├── paint-groups.json
├── paintability-report.json
├── surface-completion.json
├── masks/
│   ├── full-car.png
│   ├── editable-mask.png
│   ├── protected-mask.png
│   ├── uncertain-mask.png
│   ├── main-body-paint-mask.png
│   ├── hard-protected-mask.png
│   ├── main-body-seed-mask.png
│   └── diagnostic overlays and part masks
└── customisations/<request_hash>/
    ├── request.json
    ├── quality.json
    └── result.png
```

The asset ID hashes the input bytes, requested view, and pipeline version.
Changing mask behavior requires incrementing `PIPELINE_VERSION` in
`app/pipeline.py`, which prevents stale assets from being reused.

Render cache keys also include normalized request data and the asset pipeline
version.

## Module map

| Path | Responsibility |
|---|---|
| `app/config.py` | Settings, part aliases, required/expected view groups |
| `app/detection.py` | YOLO inference, part normalization, view inference |
| `app/roboflow.py` | SAM2/SAM3 HTTP client and response validation |
| `app/pipeline.py` | Processing orchestration, caching, persistence |
| `app/image_ops.py` | Input validation, mask cleanup, LAB renderer |
| `app/paint_analysis/` | Paint profile, materials, groups, surface completion |
| `app/modifications/` | Request schemas, restricted parsing, renderer selection |
| `app/renderers/` | Deterministic and generative render implementations |
| `app/quality/` | Mask and rendered-pixel checks |
| `app/main.py` | FastAPI routes and error envelopes |
| `streamlit_app.py` | Interactive editor and evaluation gallery |
| `tests/` | Synthetic regressions and interface/API compatibility |
| `evaluation/input/` | Real evaluation photographs |
| `evaluation/output/` | Latest evaluation renders |

## Important configuration groups

Most tuning values live in `.env` and are validated by `Settings.from_env()`:

- Provider/model selection: `ROBOFLOW_SEGMENTER`, SAM model IDs, YOLO model IDs.
- Detector sensitivity: YOLO and car-parts confidence thresholds.
- Mask cleanup: kernel size and inside-only feather radius.
- Paint profile: strict and relaxed LAB chroma thresholds.
- Surface growth: local LAB distance, gradient, neighbors, and iterations.
- Completion: morphology kernel, hole area, and region adjacency.
- Storage: `STORAGE_ROOT`.

Change defaults only after checking the complete evaluation set; these values
interact and are intended to remain image-independent.

## Running and checking the project

Load the project environment first:

```bash
source .venv/bin/activate
set -a
source .env
set +a
```

Run the interfaces:

```bash
uvicorn app.main:app --reload
streamlit run streamlit_app.py
```

Run the local verification suite:

```bash
python -m compileall -q app streamlit_app.py tests
python -m unittest discover -s tests -v
```

At the time this guide was written, the suite contains 60 passing tests.

## Debugging checklist

### Original paint remains on a body panel

Inspect, in order:

1. `masks/full-car.png` — is the panel inside the car silhouette?
2. Raw semantic part masks — was the panel incorrectly detected as glass,
   bumper, trim, or another part?
3. `masks/hard-protected-mask.png` — did protection claim it?
4. `masks/main-body-seed-mask.png` — are reliable seeds nearby?
5. `masks/growth-candidate-mask.png` — is its chroma eligible for growth?
6. `masks/surface-completion-overlay.png` and `surface-completion.json` — why
   was the connected region accepted or rejected?
7. `masks/editable-mask.png` — did final precedence remove it?

### A protected part receives paint

Inspect the raw part mask first. If it is incomplete, fix detection or the
SAM prompt/clip path rather than globally expanding protection. Then verify:

- The corresponding paint group is protected.
- `editable-mask.png` and `protected-mask.png` do not overlap.
- `quality.json` reports no changed protected pixels.

### Three-quarter side glass receives paint

Check whether door detections exist and whether their generated window prompts
were clipped to the upper door. The resolved cardinal view may correctly be
`front` or `rear`; door-driven prompts must still be present.

### The result looks flat or two-tone

First compare `editable-mask.png` with the result:

- If the area is absent from the mask, debug paint classification.
- If it is present but retains the old colour, debug LAB `paint_strength`,
  shadow floor, or highlight rolloff in `image_ops.recolour()`.

### Streamlit appears to use old behavior

The process may still have imported an older pipeline module. Restart
Streamlit after confirming `PIPELINE_VERSION` was bumped. Load `.env` before
starting it so Roboflow settings remain available.

### Auto view is ambiguous

Review raw car-parts labels/confidences and `VIEW_FAMILIES` in
`app/detection.py`. Do not silently guess when opposing evidence is weak;
manual view selection is the intended fallback.

## Rules for safe changes

When changing detection, classification, or rendering:

1. Preserve semantic/material precedence.
2. Never recover paint outside the full-car mask.
3. Never allow editable/protected overlap.
4. Keep background and protected pixels exact.
5. Avoid filename-specific or image-coordinate-specific fixes.
6. Add a small synthetic regression for the general failure mode.
7. Bump `PIPELINE_VERSION` when stored masks or metadata behavior changes.
8. Rerun representative real images after unit tests pass.

## Current limitations

- Paint/material classification is heuristic and not trained end-to-end.
- Transparent glass, black-on-black trim, unusual wraps, decals, and aftermarket
  finishes can remain ambiguous.
- Local process locks are unsuitable for multi-worker production deployment.
- The current car-parts weights and optional FLUX provider have licensing and
  deployment constraints described in the README.
- Physical part replacement, 3D output, authentication, durable job queues, and
  cloud object storage are intentionally outside the current milestone.

When a difficult image fails, prefer improving the shared semantic or
region-level rule over widening every mask. The project is designed to fail
conservatively rather than paint protected material.
