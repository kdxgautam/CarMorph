# Car customisation pipeline

Backend-only car paint customisation API. It uploads one car photograph,
segments reusable masks once, previews deterministic body recolours, and can
render controlled body or roof colour edits while restoring original pixels
outside editable paint.

For a guided code tour, processing flow, stored-asset anatomy, and debugging
checklist, see [PROJECT_GUIDE.md](PROJECT_GUIDE.md).

This project is not production-ready. The current car-parts weights are
AGPL-3.0 and process-local locks are not suitable for multi-worker production
deployment.

## Architecture

1. YOLO-World detects the primary car and zero-shot regions such as plate,
   grille, and trim.
2. A pretrained car-parts YOLO segmentation model detects windows, lights,
   bumpers, mirrors, and wheels.
3. Roboflow SAM 2 supplies the full-car and door-window geometry while SAM 3
   supplies zero-shot handles, pillars, and black trim. Set
   `ROBOFLOW_SEGMENTER=sam2` or `sam3` for single-model A/B runs.
4. OpenCV cleans masks and stores full-car and reusable part masks.
5. Body-paint analysis erodes safe panel interiors, rejects extreme lighting,
   estimates a dominant LAB/chroma profile, classifies detected parts by
   semantics/material/colour context, and stores disjoint paint groups.
6. Request-specific mask building selects main-body groups and, only when
   requested, the contrast-roof group. It still stores the explicit
   compatibility masks:
   `editable-mask.png`, `protected-mask.png`, and `uncertain-mask.png`.
7. `paintable-body.png` is still written for backward compatibility and matches
   the editable mask for new assets.
8. Deterministic rendering uses the selected mask for body colour plus glossy,
   matte, or metallic approximations.

## Requirements

- Python 3.12
- `curl`
- `jq`
- A Roboflow API key

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Download the pinned car-parts weights:

```bash
mkdir -p models

curl -L --fail \
  https://huggingface.co/mitbersh/car-parts-segmentation-m/resolve/0b9acbc20393a9c8d19a81405e9ba682b5eb3a69/parts_segmentation.pt \
  -o models/carparts-v2.pt

echo 'df8fb1aef65a91c32796205d56104c92d00cf863ed650f65437608d477ff400b  models/carparts-v2.pt' \
  | shasum -a 256 -c -
```

Create local configuration from `.env.example`:

```bash
cp .env.example .env
```

Set `ROBOFLOW_API_KEY` in `.env`. `.env`, uploaded images, processed assets,
and model weights are ignored by Git.

## Start the API

```bash
source .venv/bin/activate

set -a
source .env
set +a

uvicorn app.main:app --reload
```

The API is available at `http://127.0.0.1:8000`; OpenAPI is at `/docs`.

## Start the Streamlit interface

With the same virtual environment and `.env` loaded:

```bash
streamlit run streamlit_app.py
```

## API routes

| Method | Route | Purpose |
|---|---|---|
| `POST` | `/cars` | Upload and segment one image; use `view=auto` for local view detection |
| `GET` | `/cars/{asset_id}` | Read stored metadata |
| `GET` | `/cars/{asset_id}/assets/{path}` | Download original image or masks |
| `GET` | `/cars/{asset_id}/preview?colour=2563eb` | Backward-compatible deterministic recolour |
| `POST` | `/cars/{asset_id}/customise` | Controlled surface customisation PNG render |

`/customise` returns:

```text
X-Render-Cached: true|false
X-Renderer-Used: deterministic
X-Quality-Status: passed|passed_with_warnings|failed
```

## Customise examples

Plain colour:

```bash
curl -sS --fail-with-body \
  -X POST "http://127.0.0.1:8000/cars/${ASSET_ID}/customise" \
  -H 'Content-Type: application/json' \
  -d '{"type":"surface_edit","body_colour":"#183A63","finish":"glossy"}' \
  -o /tmp/car-plain.png
```

Body plus a separately targeted contrast roof:

```bash
curl -sS --fail-with-body \
  -X POST "http://127.0.0.1:8000/cars/${ASSET_ID}/customise" \
  -H 'Content-Type: application/json' \
  -d '{"type":"surface_edit","body_colour":"#183A63","roof_colour":"#111111"}' \
  -o /tmp/car-dual-tone.png
```

Matte finish:

```bash
curl -sS --fail-with-body \
  -X POST "http://127.0.0.1:8000/cars/${ASSET_ID}/customise" \
  -H 'Content-Type: application/json' \
  -d '{"type":"surface_edit","body_colour":"#111111","finish":"matte"}' \
  -o /tmp/car-matte.png
```

The Streamlit interface defaults to automatic view detection. The API keeps
`front` as its backward-compatible default and accepts `auto`, `front`, `rear`,
`left`, or `right`. Low-confidence automatic results return `ambiguous_view`
and require an explicit view.

## Supported modifications

- Body colour: six-digit RGB hex colours
- Finish: `glossy`, `matte`, `metallic`
- Optional contrast roof colour when that paint group is available

Unsupported in this milestone:

- Racing stripes, natural-language edits, rim/wheel/tyre/bumper/spoiler,
  suspension, convertible, body-kit, 3D, GLB, and geometry-changing
  modifications
- Arbitrary geometry fields in JSON

## Paintability masks

### Body-paint analysis

The first version is deliberately heuristic. It does not classify by colour
alone. Detected part identity and strong material evidence override colour
similarity; LAB chroma relationship, local brightness, appearance statistics,
spatial context, and confidence are then combined.

The main colour profile starts with car pixels outside detected accessories and
protected parts. It erodes that candidate mask away from panel boundaries,
removes the lightest and darkest deciles, clusters remaining LAB chroma values,
and uses the dominant cluster. This lets a painted panel in shade remain related
to the same panel in sunlight: chroma can match even when LAB lightness differs.
`body-paint-profile.json` records the samples, robust median/variance, lighting
ranges, confidence, and warnings.

Strict profile matches now create `main-body-seed-mask.png`; they are not the
final editable mask. A two-stage OpenCV/NumPy surface-completion pass grows
those seeds through locally compatible, connected pixels inside
`safe-body-candidate-mask.png`. Strong semantic protection is represented by
`hard-protected-mask.png` and cannot be crossed. Region voting and bounded
morphological completion recover coherent highlights, shadows, narrow strips,
and small internal gaps while retaining large unrelated colour regions. A final
chroma-only boundary pass tolerates strong lightness changes at panel edges
without relaxing the global candidate or hard-protection constraints.
Residual painted regions join the main body only when both their LAB chroma
and spatial adjacency match; large chromatically different regions remain
secondary paint.

`surface-completion.json` records region decisions, seed/final pixel counts,
recovered pixels, connected-component counts, small-fragment counts, internal
gap pixels, and growth iterations. `surface-completion-overlay.png` shows safe
candidates in yellow, completed body in green, strict seeds in cyan, and hard
protection in red.

Detected handles and mirror caps are included with the body only when they look
painted and match its profile confidently. Contrasting painted variants remain
separate. Chrome or plastic evidence protects them. Pillar semantics strongly
favour glossy trim protection. A detected roof that differs from the main paint
becomes `contrast_roof_paint`; it changes only when `roof_colour` is present.
Bumper masks run through the same constrained surface completion so reflections
remain painted while unrelated dark cladding stays protected.

Black main paint is retained when it is the dominant eroded panel profile;
detected trim semantics plus dark neutral appearance identify probable plastic.
Black-on-black boundaries, reflections, missing part detections, and unusual
aftermarket finishes can still be ambiguous. Those pixels remain uncertain
rather than editable. A labelled paintability/material model is the likely
future replacement for difficult cases.

### Paint groups

New assets may contain only the non-empty masks among:

```text
main-body-paint-mask.png
secondary-body-paint-mask.png
contrast-roof-mask.png
body-coloured-handles-mask.png
contrasting-handles-mask.png
body-coloured-mirror-caps-mask.png
contrasting-mirror-caps-mask.png
painted-bumper-sections-mask.png
black-plastic-trim-mask.png
glossy-black-trim-mask.png
chrome-trim-mask.png
silver-garnish-mask.png
paint-group-uncertain-mask.png
```

`paint-groups.json` contains per-group pixel counts/confidence and structured
region decisions. With `PAINT_ANALYSIS_DIAGNOSTICS=true`,
`paint-groups-overlay.png` and `body-paint-anchor-overlay.png` show group and
anchor coverage.

`protected-mask.png` combines confidently non-paintable or non-default-target
groups: windows, wheels/tyres, number plate, lights, grille, chrome, black
plastic, glossy trim, silver garnish, and contrasting paint groups.

`uncertain-mask.png` holds low-confidence regions. Luminance-only dark regions
are never declared plastic; they remain uncertain unless part/material evidence
supports protection.

The safe default `editable-mask.png` is:

```text
main body paint
+ body-coloured handles
+ body-coloured mirror caps
+ painted bumper sections
+ body-coloured spoiler
- protected
- uncertain
```

It excludes contrast roof, secondary paint, contrasting handles/caps, and trim.
A request with `roof_colour` loads the contrast-roof mask separately and applies
its colour independently. Older assets fall back to their existing editable or
paintable-body mask.

## Storage and caching

```text
data/processed/<asset_id>/
├── metadata.json
├── paintability-report.json
├── body-paint-profile.json
├── paint-groups.json
├── surface-completion.json
├── source.<extension>
├── luminance-map.png
├── masks/
│   ├── full-car.png
│   ├── paintable-body.png
│   ├── editable-mask.png
│   ├── protected-mask.png
│   ├── uncertain-mask.png
│   ├── safe-body-candidate-mask.png
│   ├── hard-protected-mask.png
│   ├── growth-candidate-mask.png
│   ├── main-body-seed-mask.png
│   ├── main-body-paint-mask.png
│   ├── contrast-roof-mask.png
│   ├── paint-groups-overlay.png
│   ├── body-paint-anchor-overlay.png
│   ├── surface-completion-overlay.png
│   └── part masks...
└── customisations/
    └── <request_hash>/
        ├── request.json
        ├── result.png
        └── quality.json
```

Upload caching remains content-addressed by image bytes, view, and pipeline
version. Customisation cache keys include renderer version, normalized
structured modification JSON, and mask/pipeline version. Equivalent JSON
requests reuse the same cache entry.

## Quality checks

Initial deterministic checks verify:

- result dimensions match the original
- outside-editable pixels remain exact
- protected pixels remain exact
- editable region is non-empty
- paint groups do not overlap
- protected and uncertain groups never overlap the default editable mask
- body-coloured handles follow the default body request
- contrasting handles and a contrast roof remain outside the default request
- the main-body profile and mask meet configured confidence/non-empty checks
- every strict seed survives in the final editable body mask
- surface growth never crosses hard protection
- sizeable main-colour-compatible residual regions produce a completeness warning

These checks do not claim vehicle-identity AI validation.

## Test commands

Local checks do not call Roboflow:

```bash
python -m unittest discover -s tests -v
python -m compileall -q app tests
python -m pip check
```

External Roboflow calls are mocked or avoided in unit tests.

## Common errors

| Code | Meaning |
|---|---|
| `no_car_detected` | YOLO did not find a car |
| `multiple_competing_cars` | More than one similarly sized car was found |
| `view_mismatch` | A side photograph was submitted as front/rear |
| `missing_masks` | A required part or stored mask is missing |
| `sam_api_timeout` | Roboflow did not answer within the configured timeout |
| `sam_api_error` | Roboflow could not be reached or rejected the request |
| `invalid_sam_response` | SAM returned unusable mask data |
| `mask_dimension_mismatch` | An image and mask have incompatible dimensions |
| `invalid_modification` | Customisation JSON failed validation |
| `quality_check_failed` | Final composite failed deterministic quality checks |

Useful masks and metadata are retained for wheels, tyres grouped with wheels,
bumper, mirrors, lights, grille, number plate, trim, and windows. This
milestone does not implement rim replacement, spoilers, body kits, 3D models,
GLB assets, 360-degree rendering, authentication, PostgreSQL, Redis, Celery,
S3, or deployment infrastructure.
# Generative bumper previews

Front and rear assets with a detected bumper support a visual-only bumper replacement preview. Configure Vertex AI with Application Default Credentials:

```bash
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT="your-project"
export GOOGLE_CLOUD_LOCATION="global"
export GENERATIVE_IMAGE_PROVIDER="vertex-ai"
export GENERATIVE_IMAGE_MODEL_ID="gemini-3.1-flash-image"
export GENERATIVE_IMAGE_TIMEOUT_SECONDS="180"
```

Enable Vertex AI API and billing for the project. The preview uses Gemini 3.1 Flash Image and restores every pixel outside CarMorph's strict bumper mask locally; it does not establish physical bumper compatibility.
