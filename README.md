# V-JEPA2 Webcam Demos

This repository contains two **independent** live-webcam demos built around Meta's **V-JEPA2** video encoder. Each lives in its own folder with its own dependencies, README, and Dockerfile, so it can be installed, run, and containerized on its own.

## Projects

### 1. [`patch_tracker/`](patch_tracker/) — V-JEPA2 Patch Similarity Tracker

Click a patch in the live webcam frame and the demo tracks it by following the most similar patch — by cosine similarity in V-JEPA2's latent space — frame to frame. Shows a similarity heatmap, the tracked anchor, and the Top-K most similar patches. **Not** object detection — pure embedding-space tracking.

→ See [`patch_tracker/README.md`](patch_tracker/README.md) for installation, usage, controls, and Docker instructions.

### 2. [`action_recognition_yolo/`](action_recognition_yolo/) — V-JEPA2 SSV2 Action Recognition + YOLO/OWL-ViT

Classifies the action happening in a short clip with a V-JEPA2 model fine-tuned on Something-Something v2 (e.g., *"Showing [something] next to [something]"*), and fills in the `[something]` placeholders with object names detected live by YOLOv8 or OWL-ViT. Also shows an optional latent-space visualization panel.

→ See [`action_recognition_yolo/README.md`](action_recognition_yolo/README.md) for installation, usage, controls, and Docker instructions.

---

## Running with Docker

Each project ships its own `Dockerfile` + `requirements.txt`, producing an independent, minimally-sized image with only the dependencies that project needs. Build and run them directly:

```bash
# Patch similarity tracker
cd patch_tracker && docker build -t vjepa2-patch-tracker .

# SSV2 action recognition + YOLO/OWL-ViT
cd action_recognition_yolo && docker build -t vjepa2-action-yolo .
```

Or use the top-level [`docker-compose.yml`](docker-compose.yml) to build/run either with one command (wires up webcam device passthrough, X11 display, and a Hugging Face cache volume):

```bash
docker compose run --rm tracker
docker compose run --rm action-yolo
```

Both Dockerfiles produce standard **Linux-based images** that build and run identically through Docker on Linux, macOS, and Windows hosts — the only thing that differs per host OS is *how you grant the container access to your webcam and display*. Each project's README has step-by-step instructions for Linux, macOS (XQuartz), and Windows (WSL2 + usbipd-win), plus NVIDIA GPU passthrough.

---

## License

See [LICENSE](LICENSE).
