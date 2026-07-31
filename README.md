# Car customisation pipeline

Backend-only car paint customisation API. It uploads one car photograph,
segments reusable masks once, previews deterministic body recolours, and can
render controlled surface edits such as finish changes and simple racing
stripes while restoring original pixels outside editable paint.

This project is not production-ready. The current car-parts weights are
AGPL-3.0, FLUX.1 Kontext dev is not a production-commercial model, the public
Hugging Face Space is development-only, and process-local locks are not
suitable for multi-worker production deployment.

## Architecture

1. YOLO-World detects the primary car and zero-shot regions such as plate,
   grille, and trim.
2. A pretrained car-parts YOLO segmentation model detects windows, lights,
   bumpers, mirrors, and wheels.
3. Roboflow SAM 2 refines full-car and box-prompt masks.
4. OpenCV cleans masks and stores full-car, reusable part masks, and explicit
   paintability masks:
   `editable-mask.png`, `protected-mask.png`, and `uncertain-mask.png`.
5. `paintable-body.png` is still written for backward compatibility and matches
   the editable mask for new assets.
6. Deterministic rendering uses the editable mask for body colour plus glossy,
   matte, or metallic approximations.
7. Generative rendering builds prompts only from validated structured requests,
   calls the FLUX provider, composites through the editable mask, and restores
   protected/background pixels.

Natural-language instructions are parsed by a restricted local parser. Raw user
instructions are not sent directly to FLUX.

## Requirements

- Python 3.12
- `curl`
- `jq`
- A Roboflow API key
- A Hugging Face read token with the FLUX.1 Kontext dev terms accepted

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
  https://huggingface.co/Majorburn/yolov11-carparts-seg/resolve/cff8060ca063b38dc03aa9aac596f1795e0e2db5/best.pt \
  -o models/carparts-seg.pt

echo '6759cf983e0bdefaa95d2d3fc6b37f89d3718a319c08617c3c7a339e18fdc3cd  models/carparts-seg.pt' \
  | shasum -a 256 -c -
```

Create local configuration from `.env.example`:

```bash
cp .env.example .env
```

Set `ROBOFLOW_API_KEY` and `HF_TOKEN` in `.env`. `.env`, uploaded images,
processed assets, and model weights are ignored by Git.

## Start the API

```bash
source .venv/bin/activate

set -a
source .env
set +a

uvicorn app.main:app --reload
```

The API is available at `http://127.0.0.1:8000`; OpenAPI is at `/docs`.

## API routes

| Method | Route | Purpose |
|---|---|---|
| `POST` | `/cars` | Upload and segment one image |
| `GET` | `/cars/{asset_id}` | Read stored metadata |
| `GET` | `/cars/{asset_id}/assets/{path}` | Download original image or masks |
| `GET` | `/cars/{asset_id}/preview?colour=2563eb` | Backward-compatible deterministic recolour |
| `POST` | `/cars/{asset_id}/render?colour=2563eb` | Backward-compatible cached FLUX colour render |
| `POST` | `/cars/{asset_id}/customise` | Controlled surface customisation PNG render |

`/customise` returns:

```text
X-Render-Cached: true|false
X-Renderer-Used: deterministic|generative
X-Quality-Status: passed|passed_with_warnings|failed
```

## Customise examples

Plain colour:

```bash
curl -sS --fail-with-body \
  -X POST "http://127.0.0.1:8000/cars/${ASSET_ID}/customise" \
  -H 'Content-Type: application/json' \
  -d '{"type":"surface_edit","body_colour":"#183A63","finish":"glossy","design_elements":[],"custom_instruction":null,"renderer":"auto"}' \
  -o /tmp/car-plain.png
```

Matte finish:

```bash
curl -sS --fail-with-body \
  -X POST "http://127.0.0.1:8000/cars/${ASSET_ID}/customise" \
  -H 'Content-Type: application/json' \
  -d '{"type":"surface_edit","body_colour":"#111111","finish":"matte","design_elements":[],"custom_instruction":null,"renderer":"auto"}' \
  -o /tmp/car-matte.png
```

Dual racing stripes:

```bash
curl -sS --fail-with-body \
  -X POST "http://127.0.0.1:8000/cars/${ASSET_ID}/customise" \
  -H 'Content-Type: application/json' \
  -d '{"type":"surface_edit","body_colour":"#183A63","finish":"metallic","design_elements":[{"type":"racing_stripes","count":2,"colour":"#D61F2C","width":"thin","placement":"bonnet_and_visible_roof","alignment":"centre"}],"custom_instruction":null,"renderer":"auto"}' \
  -o /tmp/car-stripes.png
```

Natural-language instruction:

```bash
curl -sS --fail-with-body \
  -X POST "http://127.0.0.1:8000/cars/${ASSET_ID}/customise" \
  -H 'Content-Type: application/json' \
  -d '{"type":"surface_edit","custom_instruction":"Use metallic blue with one white stripe.","renderer":"auto"}' \
  -o /tmp/car-natural-language.png
```

Rejected bumper instruction:

```bash
curl -sS --fail-with-body \
  -X POST "http://127.0.0.1:8000/cars/${ASSET_ID}/customise" \
  -H 'Content-Type: application/json' \
  -d '{"type":"surface_edit","custom_instruction":"Replace the bumper.","renderer":"auto"}'
```

The last command returns an error envelope with
`future_physical_modification`.

## Supported modifications

- Body colour: six-digit RGB hex colours
- Finish: `glossy`, `matte`, `metallic`
- Racing stripes: count `1` or `2`; width `thin`, `medium`, `thick`; placement
  `bonnet`, `visible_roof`, `bonnet_and_visible_roof`, or
  `visible_side_panels`
- Restricted instructions such as `Make the car matte black` or
  `Use metallic blue with one white stripe`

Unsupported in this milestone:

- Rim, wheel, tyre, bumper, spoiler, suspension, convertible, body-kit, 3D,
  GLB, and geometry-changing modifications
- Arbitrary geometry fields in JSON
- Unchecked free-form FLUX prompts

## Renderer selection

- Plain colour plus finish defaults to deterministic rendering.
- Matte and metallic are deterministic approximations unless
  `renderer:"generative"` is explicitly requested.
- Racing stripes and custom instructions use generative rendering.
- Physical modifications are rejected as future work.
- Complex requests are not silently downgraded to plain recolouring.

## Paintability masks

`protected-mask.png` combines confidently non-paintable regions: windows,
windscreen, wheels, tyres when grouped with wheels, number plate, lights,
grille, detector-produced chrome/black trim, and badges when detected.

`uncertain-mask.png` holds low-confidence dark or reflective regions such as
deep shadows, glossy black paint, ambiguous bumper inserts, and chrome-like
areas. Luminance-only dark regions are not blindly protected.

`editable-mask.png` is:

```text
full car - protected - uncertain
```

Protected pixels always override editable pixels. A `paintability-report.json`
and metadata `paintability_report` record editable/protected/uncertain ratios,
warnings, and the rules version.

## Storage and caching

```text
data/processed/<asset_id>/
├── metadata.json
├── paintability-report.json
├── source.<extension>
├── original.webp
├── luminance-map.png
├── masks/
│   ├── full-car.png
│   ├── paintable-body.png
│   ├── editable-mask.png
│   ├── protected-mask.png
│   ├── uncertain-mask.png
│   └── part masks...
├── renders/
│   └── flux-<colour>-<settings-hash>.png
└── customisations/
    └── <request_hash>/
        ├── request.json
        ├── result.png
        └── quality.json
```

Upload caching remains content-addressed by image bytes, view, and pipeline
version. Customisation cache keys include renderer version, provider/space,
provider settings, normalized structured modification JSON, and mask/pipeline
version. Equivalent JSON requests reuse the same cache entry.

## Quality checks

Initial deterministic checks verify:

- result dimensions match the original
- outside-editable pixels remain exact
- protected pixels remain exact
- editable region is non-empty
- generated images are readable before compositing

These checks do not claim vehicle-identity AI validation.

## Test commands

Local checks do not call Roboflow or Hugging Face:

```bash
python -m unittest discover -s tests -v
python -m compileall -q app tests
python -m pip check
```

External Roboflow and FLUX calls are mocked or avoided in unit tests.

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
| `future_physical_modification` | Natural language asked for a physical change |
| `future_not_supported` | Structured request targets a future renderer |
| `unsupported_instruction` | Natural language did not match safe surface edits |
| `renderer_not_supported` | Explicit renderer cannot handle the request |
| `flux_unavailable` | The FLUX Space is unavailable, queued, or out of quota |
| `invalid_flux_response` | FLUX returned an invalid file or response |
| `quality_check_failed` | Final composite failed deterministic quality checks |

## Future part replacement

Interfaces reserve a later physical-part pipeline:

```text
Original image
+ target-part mask
+ protected neighbouring masks
+ compatible reference asset
+ geometry/fitment constraints
-> part-replacement renderer
```

Useful masks and metadata are retained for wheels, tyres grouped with wheels,
bumper, mirrors, lights, grille, number plate, trim, and windows. This
milestone does not implement rim replacement, bumper replacement, spoilers,
body kits, 3D models, GLB assets, 360-degree rendering, frontend applications,
authentication, PostgreSQL, Redis, Celery, S3, or deployment infrastructure.
