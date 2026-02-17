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


Please cite the original V-JEPA2 paper if you use this in academic work.











