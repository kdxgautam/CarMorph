# Car customisation pipeline

Backend-only MVP for uploading one car photograph, segmenting reusable car-part
masks once, and previewing paint colours without regenerating the background or
non-paintable parts.

## Pipeline

1. YOLO-World detects the primary car and zero-shot regions such as the plate.
2. A pretrained car-parts YOLO segmentation model detects windows, lights,
   bumpers, mirrors, and wheels.
3. Roboflow SAM 2 refines the full-car and box-prompt masks.
4. OpenCV cleans the masks and subtracts windows, wheels, tyres, lights, plates,
   grille, and trim from the full-car mask.
5. The original image, masks, and luminance map are stored under one
   image-and-view content hash. Uploading identical bytes with the same view
   reuses them without rerunning segmentation.
6. The deterministic preview recolours only the paintable-body mask.
7. The FLUX route asks the official FLUX.1 Kontext dev Space to edit the car,
   restores the original luminance pattern, normalizes the median body colour
   to the requested RGB, and composites only through the paintable-body mask.

No Roboflow dataset, manual annotation, or frontend is required.

## Requirements

- Python 3.12
- `curl`
- `jq`
- A Roboflow API key
- A Hugging Face read token with the FLUX.1 Kontext dev terms accepted

Create the environment and install dependencies:

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

Create the local configuration:

```bash
cp .env.example .env
```

Set these two secrets in `.env`:

```dotenv
ROBOFLOW_API_KEY=your_roboflow_key
HF_TOKEN=hf_your_read_token
```

Get the Roboflow key from
[Roboflow API settings](https://app.roboflow.com/settings/api), create the
Hugging Face token at
[Hugging Face access tokens](https://huggingface.co/settings/tokens), and
accept the gated
[FLUX.1 Kontext dev terms](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev).

`.env`, uploaded images, processed assets, and model weights are ignored by
Git.

## Start the API

In terminal 1:

```bash
source .venv/bin/activate

set -a
source .env
set +a

uvicorn app.main:app --reload
```

The API is available at `http://127.0.0.1:8000` and its interactive OpenAPI
page is at `http://127.0.0.1:8000/docs`.

## Test the complete pipeline

Run the following commands in terminal 2 while Uvicorn is running.

### 1. Select an image, view, and colour

Put the image in `data/uploads/`:

```bash
mkdir -p data/uploads
```

Set the test values:

```bash
source .venv/bin/activate

CAR_IMAGE_PATH="$PWD/data/uploads/tiagoside.png"
CAR_VIEW="right"
CAR_COLOUR="2563eb"

test -f "$CAR_IMAGE_PATH" && echo "Image found: $CAR_IMAGE_PATH"
```

`CAR_VIEW` must be `front`, `rear`, `left`, or `right` and must match the
photograph. The current left/right pipeline uses the same required parts.

### 2. Upload, detect, and segment

```bash
set -o pipefail

curl -sS --fail-with-body \
  -X POST http://127.0.0.1:8000/cars \
  -F "view=${CAR_VIEW}" \
  -F "image=@${CAR_IMAGE_PATH}" \
  | tee /tmp/car-response.json | jq
```

Extract and validate the returned asset ID:

```bash
ASSET_ID="$(jq -er '.asset_id' /tmp/car-response.json)"
echo "ASSET_ID=$ASSET_ID"
```

Read the stored metadata:

```bash
curl -sS --fail-with-body \
  "http://127.0.0.1:8000/cars/${ASSET_ID}" \
  | jq
```

Inspect all generated masks without downloading duplicates into the project
root:

```bash
open "data/processed/${ASSET_ID}/masks"
open "data/processed/${ASSET_ID}/masks/full-car.png"
open "data/processed/${ASSET_ID}/masks/paintable-body.png"
open "data/processed/${ASSET_ID}/masks/windows.png"
open "data/processed/${ASSET_ID}/masks/wheels.png"
open "data/processed/${ASSET_ID}/masks/lights.png"
```

Repeating the upload command with identical image bytes and view returns the
existing asset and does not rerun YOLO or SAM.

### 3. Test deterministic recolouring

```bash
curl -sS --fail-with-body \
  "http://127.0.0.1:8000/cars/${ASSET_ID}/preview?colour=${CAR_COLOUR}" \
  -o /tmp/car-preview.png

open /tmp/car-preview.png
```

### 4. Test FLUX generation

The first request for a new asset/colour may queue on Hugging Face ZeroGPU:

```bash
curl -sS --fail-with-body \
  -D /tmp/car-flux-headers.txt \
  -X POST \
  "http://127.0.0.1:8000/cars/${ASSET_ID}/render?colour=${CAR_COLOUR}" \
  -o /tmp/car-flux.png

cat /tmp/car-flux-headers.txt
open /tmp/car-flux.png
open "data/processed/${ASSET_ID}/renders"
```

The first successful request returns:

```text
X-Render-Cached: false
```

Repeat the same request to verify that no second FLUX inference runs:

```bash
curl -sS --fail-with-body \
  -D /tmp/car-flux-cached-headers.txt \
  -X POST \
  "http://127.0.0.1:8000/cars/${ASSET_ID}/render?colour=${CAR_COLOUR}" \
  -o /tmp/car-flux-cached.png

grep -i '^x-render-cached:' /tmp/car-flux-cached-headers.txt
```

The repeated request returns:

```text
X-Render-Cached: true
```

Test another colour by changing `CAR_COLOUR`. Each new asset/colour/settings
combination consumes one FLUX inference and then becomes cached:

```bash
CAR_COLOUR="1e3a8a"

curl -sS --fail-with-body \
  -X POST \
  "http://127.0.0.1:8000/cars/${ASSET_ID}/render?colour=${CAR_COLOUR}" \
  -o /tmp/car-flux-dark-blue.png

open /tmp/car-flux-dark-blue.png
```

### 5. Run local checks

These checks do not call Roboflow or Hugging Face:

```bash
python -m unittest discover -s tests -v
python -m compileall -q app tests
python -m pip check
```

## API routes

| Method | Route | Purpose |
|---|---|---|
| `POST` | `/cars` | Upload and segment one image |
| `GET` | `/cars/{asset_id}` | Read stored metadata and bounding box |
| `GET` | `/cars/{asset_id}/assets/{path}` | Download an original image or mask |
| `GET` | `/cars/{asset_id}/preview?colour=2563eb` | Deterministic body recolouring |
| `POST` | `/cars/{asset_id}/render?colour=2563eb` | Cached FLUX Kontext rendering |

## Storage

```text
data/
├── uploads/
│   └── local test images...
└── processed/
    └── <asset_id>/
        ├── metadata.json
        ├── source.<extension>
        ├── original.webp
        ├── luminance-map.png
        ├── renders/
        │   └── flux-<colour>-<settings-hash>.png
        └── masks/
            ├── full-car.png
            ├── paintable-body.png
            ├── bumper.png
            ├── dark_trim.png
            ├── lights.png
            ├── mirrors.png
            ├── plate.png
            ├── wheels.png
            ├── windows.png
            └── optional detected exclusions...
```

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
| `flux_unavailable` | The FLUX Space is unavailable, queued, or out of quota |
| `invalid_flux_response` | FLUX returned an invalid file or response |

For a SAM cold start, keep `ROBOFLOW_TIMEOUT_SECONDS=180` and fully restart
Uvicorn after changing `.env`. For FLUX quota exhaustion, wait for the Hugging
Face quota to reset; repeated cached colours do not consume additional quota.

For side views where the car-parts model does not detect glass directly, the
pipeline creates SAM prompts inside the expected glass area of detected doors
and clips the results to the upper-door bounds. It returns `missing_masks`
instead of using broad prompts when no usable door is detected.

The luminance-based `dark_trim` mask is stored for inspection but is not
subtracted from the paintable body because deep paint shadows can otherwise be
mistaken for plastic trim. Detector-produced grille and trim masks remain
excluded.

## Model and deployment notes

- The pinned third-party car-parts weights are AGPL-3.0.
- FLUX.1 Kontext dev is governed by Black Forest Labs'
  non-commercial/non-production development-model license.
- The public FLUX Space runs on Hugging Face ZeroGPU and is suitable for
  development, not production availability.
- Both segmentation and FLUX use process-local locks. Run one Uvicorn worker
  for this MVP; use shared job/lock storage before running multiple workers.
- Window-mask precision is the current quality ceiling on some side views:
  over-segmentation can exclude nearby painted roof or pillar pixels.
