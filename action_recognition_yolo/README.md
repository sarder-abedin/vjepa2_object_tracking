# V-JEPA2 SSV2 Action Recognition + (YOLO / OWL-ViT) Object Detection — Webcam Demo

This project runs a live webcam demo that combines:

- **V-JEPA2 video action classification (SSV2 fine-tuned)** to predict an action label for short clips (e.g., *"Showing [something] next to [something]"*).
- **Object detection** using **either**:
  - **YOLOv8 (Ultralytics)** — fast, fixed-category COCO-style detector, or
  - **OWL-ViT (open-vocabulary)** — detects only what you request via text prompts.

The demo can **replace** the action label's `[something]` placeholders with detected object names (best-effort heuristic) and optionally shows a **latent-space visualization** (heatmap + scatter) from the V-JEPA2 backbone.

---

## What it does

1. Captures webcam frames and buffers a short clip (`NUM_FRAMES`, default 7).
2. Runs **VJEPA2ForVideoClassification** on the clip and prints top action predictions.
3. Runs an object detector (YOLO or OWL-ViT) periodically on the live frame.
4. Fills placeholders in the action string, for example:  
   `Showing [something] next to [something]` → `Showing bottle next to laptop` (example).
5. Displays:
   - Live video with detections
   - Optional latent heatmap overlay
   - Optional latent scatter panel (right side)

## Demo Preview
<img width="1023" height="532" alt="Screenshot 2026-03-02 at 09 28 46" src="https://github.com/user-attachments/assets/2936a968-7445-4429-8c55-2fdcf2d2a4b2" />
<img width="721" height="246" alt="Screenshot 2026-03-02 at 09 41 53" src="https://github.com/user-attachments/assets/f9bc2709-567f-4df9-b3ea-4b5dc1d645e6" />

---

## Important note about "something" labels

SSV2-style models are trained to recognize actions in an object-agnostic way (hence "something").  
This demo **does not guarantee correct grounding** of "something #1" and "something #2". The replacement is heuristic and can be wrong in cluttered scenes or when the detector misses objects.

---

## Requirements

- Python 3.9+ recommended
- A working webcam
- macOS / Linux / Windows

### Python packages

Core:
- `torch`
- `transformers`
- `opencv-python`
- `numpy`
- `Pillow`

Optional (for YOLO):
- `ultralytics`

(All pinned in [`requirements.txt`](requirements.txt).)

Install:

```bash
pip install -r requirements.txt
```

> Hugging Face token warning: you can ignore it, but for higher rate limits:
> `huggingface-cli login` (or set `HF_TOKEN`).

---

## Run (on host)

```bash
python action_prediction_webcam.py
```

The script automatically selects **CUDA / MPS / CPU** depending on availability.

---

## Running with Docker

This project has its own [`Dockerfile`](Dockerfile) (Python 3.10-slim + the system libraries OpenCV needs for its GUI window and webcam capture, plus `yolov8n.pt` baked into the image). It builds a Linux-based image that runs identically through Docker on **Linux, macOS, and Windows** hosts — what differs between hosts is how you grant the container access to your webcam and display (see below).

### 1) Build the image

```bash
cd action_recognition_yolo
docker build -t vjepa2-action-yolo .
```

Or from the repo root with Docker Compose:

```bash
docker compose build action-yolo
```

### 2) Run it — Linux

The script opens a live OpenCV window (`cv2.imshow`) and reads from `/dev/video0`, so the container needs access to your X11 display and the camera device:

```bash
xhost +local:docker

docker run --rm -it \
  --device=/dev/video0:/dev/video0 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  vjepa2-action-yolo
```

Or simply:

```bash
docker compose run --rm action-yolo
```

### 3) Run it — macOS

Docker Desktop on macOS runs containers in a Linux VM, so there is no `/dev/video0` to pass through and no native X11. The practical options are:

- **Run the script directly on the host** (see *Run* above) — simplest for webcam access.
- **Forward the GUI only**: install [XQuartz](https://www.xquartz.org/), run `xhost +localhost`, and set `DISPLAY=host.docker.internal:0` when starting the container. You'll still need a way to forward the camera (e.g. run the script on the host and only containerize non-camera workloads).

### 4) Run it — Windows

Use **WSL2** with Docker Desktop:

- Install [usbipd-win](https://github.com/dorssel/usbipd-win) to attach your USB webcam to the WSL2 instance, exposing it as `/dev/videoX`.
- Run an X server on Windows (e.g. [VcXsrv](https://sourceforge.net/projects/vcxsrv/) or [X410](https://x410.dev/)), then set `DISPLAY=<host-ip>:0` and mount `/tmp/.X11-unix` from within WSL2.
- Then run the same `docker run` / `docker compose run` command shown for Linux from inside your WSL2 shell.

### 5) GPU acceleration (NVIDIA)

Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host, then add `--gpus all` to the `docker run` command (or `runtime: nvidia` / `deploy.resources.reservations.devices` to the Compose service) so PyTorch/YOLO can use CUDA inside the container.

### Notes

- Mount a volume for the Hugging Face cache (e.g. `-v $HOME/.cache/huggingface:/root/.cache/huggingface`, already wired in `docker-compose.yml`) to persist downloaded model weights across container runs.

---

## Controls

### Quit
- `q` or `Esc`: quit

### Detector selection
- `b`: switch detector backend **YOLO ↔ OWL-ViT**
- `d`: toggle detections ON/OFF (overlay + inference)
- `c`: toggle detector printing to console ON/OFF
- `p`: edit OWL prompts (comma-separated), then:
  - `Enter`: save
  - `Esc`: cancel

### Primary object mode (used for filling placeholders)
- `m`: toggle primary mode **AUTO ↔ MANUAL**
- `u`: clear manual primary selection
- **Mouse click** on a detection box (in MANUAL mode): set the primary *instance*

### Latent visualization
- `l`: toggle latent panel ON/OFF (right side)
- `h`: toggle latent heatmap overlay ON/OFF
- `g`: toggle patch grid ON/OFF

---

## Configuration (top of script)

Key parameters you may want to tune:

- `NUM_FRAMES`: clip length (more context vs more latency)
- `YOLO_EVERY_N_FRAMES`: run YOLO less frequently if slow
- `OWL_EVERY_N_FRAMES`: run OWL less frequently (OWL is slower)
- `OWL_PROMPTS_DEFAULT`: open-vocabulary prompt list for OWL
- `DISPLAY_SCALE`: window scale (1 small / 2 medium / 3 large)

---

## How primary object selection works

### AUTO mode
- Estimates a motion region from the clip (difference between first and last clip frames).
- Selects the "primary" object as the detection with best overlap with that motion region (or falls back to confidence).
- Optionally excludes labels (default excludes `person`) via `PRIMARY_EXCLUDE_LABELS_AUTO`.

### MANUAL mode (instance-specific)
- You click a box to select the primary instance.
- The script stores the box, label, and an HSV histogram appearance signature.
- Across frames, it matches the best same-label detection using a combined score:
  IoU continuity + proximity + size similarity + appearance similarity.

---

## Performance tips (macOS / MPS)

- Increase `YOLO_EVERY_N_FRAMES` (e.g., 5–10) if YOLO is heavy.
- Increase `OWL_EVERY_N_FRAMES` (e.g., 10–20) and reduce OWL prompt count.
- Turn off latent panel with `l` for higher FPS.
- Keep the 256×256 processing for speed (this demo is designed around the model crop).

---

## Troubleshooting

### YOLO not installed
```bash
pip install -U ultralytics
```

### Webcam not opening
Try another camera index:
```python
CAM_INDEX = 1
```

### OWL-ViT detects nothing
OWL-ViT only detects objects in the current prompt list. Press `p` to add relevant terms.

---

## Models used

- V-JEPA2 SSV2 checkpoint: `facebook/vjepa2-vitl-fpc16-256-ssv2`
- YOLOv8: `yolov8n.pt` (Ultralytics)
- OWL-ViT: `google/owlvit-base-patch32`

---

## Disclaimer

This is a research/demo script. Placeholder replacement is heuristic and not guaranteed to be correct.

**Please cite the original V-JEPA2 paper if you use this in academic work.**
