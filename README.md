**V-JEPA2 Webcam Patch Similarity Tracker (OpenCV Demo)**

A small, **open-source demo** that uses a pretrained **V-JEPA2** video encoder to:

- split a webcam frame into a **16×16 grid of patches** (default: 256×256 input, 16×16 patch size),
- let you **click on a patch** to set an **anchor template** embedding,
- compute **cosine similarity** between the anchor embedding and all other patches,
- show a **heatmap overlay** + **anchor dot** + **Top-K most similar patch dots**,
- update the anchor position by choosing the **most similar patch (argmax)** → simple embedding-based tracking.

This is **not** object detection or segmentation. It visualizes and tracks based on **similarity in the model’s latent space**.

---

## Demo Preview (What you’ll see)

- **White dot**: the tracked/anchor patch.
- **Magenta dots (1..K)**: the **Top-K most similar** patches (excluding the best match).
- **Heatmap**: similarity map over the image (red = relatively more similar than blue).
- **Hover text**: cosine similarity of the patch under your mouse to the anchor template.

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



