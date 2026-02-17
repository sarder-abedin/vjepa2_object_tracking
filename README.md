**V-JEPA2 Webcam Patch Similarity Tracker (OpenCV Demo)**

A small, open-source demo that uses a pretrained **V-JEPA2** video encoder to:

- split a webcam frame into a **16×16 grid of patches** (default: 256×256 input, 16×16 patch size),
- let you **click on a patch** to set an **anchor template** embedding,
- compute **cosine similarity** between the anchor embedding and all other patches,
- show a **heatmap overlay** + **anchor dot** + **Top-K most similar patch dots**,
- update the anchor position by choosing the **most similar patch (argmax)** → simple embedding-based tracking.

Please note that this is not object detection or segmentation. It visualizes and tracks based on similarity in the model’s latent space.

---

**Demo Preview (What you’ll see)**

- White dot: the tracked/anchor patch.
- Magenta dots (1..K): the **Top-K most similar** patches (excluding the best match).
- Heatmap: similarity map over the image (red = relatively more similar than blue).
- Hover text: cosine similarity of the patch under your mouse to the anchor template.

**How It Works (Conceptually)**

1. Capture a short clip from webcam (`NUM_FRAMES`, default 16).
2. Run V-JEPA2 encoder → get patch embeddings (vectors of length **D=1024** for ViT-L).
3. Store the embedding of the clicked patch as the **template**.
4. Each update:
   - compute cosine similarity between the template and every patch embedding,
   - move the anchor to the patch with **maximum similarity** (simple tracking),
   - display the similarity heatmap + top-K similar patches.

**Requirements**

- Python 3.9+ recommended
- A working webcam
- macOS / Linux / Windows supported

### Python packages

- `torch`
- `transformers`
- `huggingface_hub`
- `opencv-python`
- `numpy`

### Installation

1) Create and activate an environment

Using conda:

```bash
conda create -n vjepa2-demo python=3.10 -y
conda activate vjepa2-demo

Or using venv:
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

**Install dependencies**
pip install -U torch torchvision torchaudio
pip install -U transformers huggingface_hub opencv-python numpy

If you are on Apple Silicon (M1/M2/M3), install a PyTorch build that supports mps.
If you are on an NVIDIA GPU machine, install the CUDA-enabled PyTorch build suitable for your CUDA version.

### Running the Demo
python vjepa2_object_tracking_webcam.py

On startup, you should see logs like:
•	device (cpu / cuda / mps)
•	patch grid (16×16)
•	expected token counts

**Controls**
Left click: set template anchor patch
r: reset template + tracking
g: toggle patch grid overlay
h: toggle heatmap overlay
q or Esc: quit

**Configuration**
At the top of the script:
MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
NUM_FRAMES = 16
RUN_EVERY_SEC = 0.7
TOPK = 8
LOST_THRESHOLD = 0.55

**Notes**
•	NUM_FRAMES trades off speed vs stability. Larger = more stable but slower.
•	RUN_EVERY_SEC throttles how often inference runs. Increase if CPU is slow.
•	TOPK controls how many “most similar” patch dots to show.
•	LOST_THRESHOLD is only used for reporting ("LOST?") in the terminal logs.
Interpreting the Output
In the terminal, each update prints a report like:
•	the tracked patch location (i,j) in the 16×16 grid
•	best cosine similarity score
•	min/mean/max similarity values
•	Top-K patch coordinates with their cosine similarity
Important: the heatmap colors are normalized per update.
Red means “more similar than other patches in this frame,” not an absolute similarity threshold.

**Limitations**
This demo does not:
•	output object labels (“this is a face” / “this is a block”),
•	perform pixel-level segmentation,
•	guarantee stable identity tracking under occlusion,
•	track smoothly at high FPS on CPU.
It does:
•	show how V-JEPA2 embeddings cluster similar regions,
•	provide a simple “tracking-by-similarity” mechanism in latent space.

**Troubleshooting**
1) Hugging Face download / 401 Unauthorized
If Hugging Face requires a login on your system:
hf auth login

2) Slow FPS
Increase RUN_EVERY_SEC (e.g., 1.0–1.5)
Reduce NUM_FRAMES (keep multiple of tubelet size)
Try GPU acceleration if available (cuda or mps)
3) Webcam not opening
Change CAM_INDEX = 0 to CAM_INDEX = 1

**Citation / Model Credit**
This demo uses pretrained checkpoints from Meta’s V-JEPA2 via Hugging Face Transformers:
•	Model ID: facebook/vjepa2-vitl-fpc64-256
If you use this for academic work, please cite the original V-JEPA2 paper and the Transformers model documentation.

**TODO**:
•	faster tracking (local-window search, EMA template update),
•	better UI (show selected patch cell, show similarity numbers on screen),
•	optional re-acquisition after loss,
•	packaging as a small pip module.

**Acknowledgements**
•	Meta AI / FAIR for V-JEPA2
•	Hugging Face Transformers for model loading
•	OpenCV for webcam + visualization









<img width="451" height="688" alt="image" src="https://github.com/user-attachments/assets/93524360-e5cd-49de-81b5-5d4ae4f13a65" />

