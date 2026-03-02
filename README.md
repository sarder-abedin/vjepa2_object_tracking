**V-JEPA2 Webcam Patch Similarity Tracker (OpenCV Demo)**

A small, **open-source demo** that uses a pretrained **V-JEPA2** video encoder to:

- split a webcam frame into a **16×16 grid of patches** (default: 256×256 input, 16×16 patch size),
- let you **click on a patch** to set an **anchor template** embedding,
- compute **cosine similarity** between the anchor embedding and all other patches,
- show a **heatmap overlay** + **anchor dot** + **Top-K most similar patch dots**,
- update the anchor position by choosing the **most similar patch (argmax)** → simple embedding-based tracking.

This is **not** object detection or segmentation. It visualizes and tracks based on **similarity in the model’s latent space**.

---

## Demo Preview

- **White dot**: the tracked/anchor patch.
- **Magenta dots (1..K)**: the **Top-K most similar** patches (excluding the best match).
- **Heatmap**: similarity map over the image (red = relatively more similar than blue).
- **Hover text**: cosine similarity of the patch under your mouse to the anchor template.

<img width="253" height="280" alt="image" src="https://github.com/user-attachments/assets/b4ebbd8e-fa22-4889-b655-392630e2bd55" />
<img width="256" height="280" alt="image" src="https://github.com/user-attachments/assets/cefe7afc-9f47-4e4e-943b-9093e2f60e80" />

<img width="254" height="278" alt="image" src="https://github.com/user-attachments/assets/a28fcfe6-6798-4cae-ac34-1a45e7c07438" />
<img width="670" height="283" alt="image" src="https://github.com/user-attachments/assets/9c97c85c-750c-4632-959c-8de613fc5e51" />




---

## How It Works (Conceptually)

1. Capture a short clip from webcam (`NUM_FRAMES`, default 16).
2. Run V-JEPA2 encoder → get patch embeddings (vectors of length **D=1024** for ViT-L).
3. Store the embedding of the clicked patch as the **template**.
4. Each update:
   - compute cosine similarity between the template and every patch embedding,
   - move the anchor to the patch with **maximum similarity** (simple tracking),
   - display the similarity heatmap + top-K similar patches.

---

## Requirements

- Python 3.9+ recommended
- A working webcam
- macOS / Linux / Windows supported

### Python packages

- `torch`
- `transformers`
- `huggingface_hub`
- `opencv-python`
- `numpy`

---

## Installation

### 1) Create an environment (optional)

Conda:
`conda create -n vjepa2-demo python=3.10 -y`
`conda activate vjepa2-demo`

### 2) Install dependencies
`pip install -U torch torchvision torchaudio`

`pip install -U transformers huggingface_hub opencv-python numpy`

## Notes on acceleration:

NVIDIA GPU: install the correct CUDA-enabled PyTorch build for your CUDA version.

Apple Silicon: use a PyTorch build that supports mps for GPU acceleration.

## Run
`python vjepa2_object_tracking_webcam.py`

## Controls

Left click: set template anchor patch

`r`: reset template + tracking

`g`: toggle patch grid

`h`: toggle heatmap overlay

`q` or `Esc`: quit

## Configuration

Edit at the top of the script:

`MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"`

`NUM_FRAMES = 16`

`RUN_EVERY_SEC = 0.7`

`TOPK = 8`

`LOST_THRESHOLD = 0.55`

## Interpreting the Heatmap

The heatmap visualizes similarity in the model’s embedding space.

Colors are normalized per update (red = most similar in that update).

Do not interpret red as an absolute similarity threshold.

## Troubleshooting
1) Hugging Face authentication errors

   `hf auth login`

2) Webcam not opening
   Try another camera index as, `CAM_INDEX = 1`

3) Slow performance

   3a) Increase `RUN_EVERY_SEC`
   
   3b) Decrease `NUM_FRAMES`
   
   3c) Use `mps` (Apple Silicon) or `cuda` (NVIDIA) if available

## Model Credit

This demo uses the pretrained V-JEPA2 checkpoint: `facebook/vjepa2-vitl-fpc64-256`

-------------------------------------------------------------------------------------
**V-JEPA2 SSV2 Action Recognition + (YOLO / OWL-ViT) Object Detection — Webcam Demo**

This project runs a live webcam demo that combines:

- **V-JEPA2 video action classification (SSV2 fine-tuned)** to predict an action label for short clips (e.g., *“Showing [something] next to [something]”*).
- **Object detection** using **either**:
  - **YOLOv8 (Ultralytics)** — fast, fixed-category COCO-style detector, or
  - **OWL-ViT (open-vocabulary)** — detects only what you request via text prompts.

The demo can **replace** the action label’s `[something]` placeholders with detected object names (best-effort heuristic) and optionally shows a **latent-space visualization** (heatmap + scatter) from the V-JEPA2 backbone.

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

## Important note about “something” labels

SSV2-style models are trained to recognize actions in an object-agnostic way (hence “something”).  
This demo **does not guarantee correct grounding** of “something #1” and “something #2”. The replacement is heuristic and can be wrong in cluttered scenes or when the detector misses objects.

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

Install:

```bash
pip install -U torch torchvision torchaudio
pip install -U transformers opencv-python numpy pillow
pip install -U ultralytics
```

> Hugging Face token warning: you can ignore it, but for higher rate limits:
> `huggingface-cli login` (or set `HF_TOKEN`).

---

## Run

```bash
python vjepa2_ssv2_action_prediction_webcam.py
```

The script automatically selects **CUDA / MPS / CPU** depending on availability.

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
- Selects the “primary” object as the detection with best overlap with that motion region (or falls back to confidence).
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












