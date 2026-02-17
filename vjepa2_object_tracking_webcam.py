import time
from collections import deque

import cv2
import numpy as np
import torch
from transformers import AutoVideoProcessor, AutoModel

# ----------------------------
# Config
# ----------------------------
MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
NUM_FRAMES = 16
CAM_INDEX = 0
WINDOW_NAME = "V-JEPA2 tracking by embedding similarity (grid + anchor + TopK)"
OVERLAY_ALPHA = 0.45
RUN_EVERY_SEC = 0.7

SHOW_GRID = True
SHOW_HEATMAP = True

TOPK = 8               # K most similar patches (excluding the best match/anchor patch)
LOST_THRESHOLD = 0.55


# ----------------------------
# Helpers
# ----------------------------
def to_rgb_square(frame_bgr: np.ndarray, size: int = 256) -> np.ndarray:
    frame = cv2.resize(frame_bgr, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def normalize_01(x: np.ndarray) -> np.ndarray:
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn)).astype(np.float32)

def overlay_heatmap(frame_bgr: np.ndarray, heat_01_256: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat_u8 = np.clip(heat_01_256 * 255.0, 0, 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame_bgr, 1 - alpha, heat_color, alpha, 0)

def draw_patch_grid(img_bgr: np.ndarray, patch_px: int, color=(255, 255, 255), thickness=1):
    h, w = img_bgr.shape[:2]
    for x in range(0, w, patch_px):
        cv2.line(img_bgr, (x, 0), (x, h - 1), color, thickness, cv2.LINE_AA)
    for y in range(0, h, patch_px):
        cv2.line(img_bgr, (0, y), (w - 1, y), color, thickness, cv2.LINE_AA)

def patch_center_pixel(i: int, j: int, patch_px: int) -> tuple[int, int]:
    cx = j * patch_px + patch_px // 2
    cy = i * patch_px + patch_px // 2
    return cx, cy

def draw_anchor_dot(img_bgr: np.ndarray, anchor_ij: tuple[int, int], patch_px: int):
    i, j = anchor_ij
    cx, cy = patch_center_pixel(i, j, patch_px)
    cv2.circle(img_bgr, (cx, cy), 7, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(img_bgr, (cx, cy), 5, (255, 255, 255), -1, cv2.LINE_AA)

def draw_topk_dots(img_bgr: np.ndarray, topk_list: list[tuple[int, int, float]], patch_px: int):
    # Magenta dots + rank labels
    for rank, (i, j, s) in enumerate(topk_list, start=1):
        cx, cy = patch_center_pixel(i, j, patch_px)
        cv2.circle(img_bgr, (cx, cy), 6, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img_bgr, (cx, cy), 4, (255, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(img_bgr, str(rank), (cx + 6, cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 2, cv2.LINE_AA)

def topk_patches(sim: np.ndarray, exclude_ij: tuple[int, int], k: int) -> list[tuple[int, int, float]]:
    Gh, Gw = sim.shape
    flat = sim.flatten()
    ei, ej = exclude_ij
    exclude_idx = ei * Gw + ej

    idxs = np.argsort(-flat)[: (k + 30)]
    out = []
    for idx in idxs:
        if idx == exclude_idx:
            continue
        i = int(idx // Gw)
        j = int(idx % Gw)
        out.append((i, j, float(flat[idx])))
        if len(out) >= k:
            break
    return out


# ----------------------------
# Load model
# ----------------------------
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Running on = {device}")
print(f"Loading MODEL_ID = {MODEL_ID}")

processor = AutoVideoProcessor.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID).to(device)
model.eval()

patch_size = int(getattr(model.config, "patch_size", 16))
tubelet_size = int(getattr(model.config, "tubelet_size", 2))
crop_size = int(getattr(model.config, "crop_size", 256))

Gh = crop_size // patch_size
Gw = crop_size // patch_size

if NUM_FRAMES % tubelet_size != 0:
    raise ValueError(f"NUM_FRAMES must be multiple of tubelet_size={tubelet_size}")

Tt = NUM_FRAMES // tubelet_size
expected_tokens = Tt * Gh * Gw

print(f"crop_size={crop_size}, patch_size={patch_size} -> grid={Gh}x{Gw} patches")
print(f"tubelet_size={tubelet_size}, NUM_FRAMES={NUM_FRAMES} -> temporal_tokens={Tt}")
print(f"Expected spatiotemporal tokens = {expected_tokens} (+ optional CLS token)")
print("Controls: click=set target | r=reset | g=grid | h=heatmap | q=quit")


# ----------------------------
# Tracking state
# ----------------------------
frames_buf = deque(maxlen=NUM_FRAMES)

# selected_patch is the CURRENT tracked location (moves over time)
selected_patch = None
hover_patch = None

# anchor_ref is the stored template embedding (fixed unless you click again)
anchor_ref = None          # (D,)
anchor_ref_norm = None     # (D,)
anchor_pending = False     # click happened, but we will capture embedding at next inference

latest_sim_raw = None      # (Gh,Gw) similarity to anchor_ref
latest_topk = []           # list[(i,j,sim)]
latest_status = "Buffer frames then click on an object patch."
latest_base = None
latest_overlay = None

last_run_time = 0.0


def mouse_cb(event, x, y, flags, param):
    global selected_patch, hover_patch, anchor_pending, latest_status

    i = int(y / crop_size * Gh)
    j = int(x / crop_size * Gw)
    i = max(0, min(Gh - 1, i))
    j = max(0, min(Gw - 1, j))
    hover_patch = (i, j)

    if event == cv2.EVENT_LBUTTONDOWN:
        # Set initial location; we'll grab anchor_ref embedding on next model update
        selected_patch = (i, j)
        anchor_pending = True
        latest_status = f"Clicked patch (i={i}, j={j}) — capturing template embedding..."


cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, mouse_cb)

cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Try CAM_INDEX=1 or check permissions.")


# ----------------------------
# Main loop
# ----------------------------
while True:
    ok, frame_bgr = cap.read()
    if not ok:
        break

    frame_rgb = to_rgb_square(frame_bgr, crop_size)
    frame_bgr_256 = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    frames_buf.append(frame_rgb)
    latest_base = frame_bgr_256.copy()

    now = time.time()

    # Run model when buffer full, and time throttled
    if len(frames_buf) == NUM_FRAMES and (now - last_run_time) > RUN_EVERY_SEC:
        last_run_time = now
        t0 = time.time()

        inputs = processor(list(frames_buf), return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            try:
                outputs = model(**inputs, skip_predictor=True)
            except TypeError:
                outputs = model(**inputs)
            tokens = outputs.last_hidden_state  # (B, seq_len, D)

        tok = tokens[0].float().cpu().numpy()
        seq_len, D = tok.shape

        # Remove CLS token if present
        if seq_len == expected_tokens + 1:
            tok = tok[1:]
            seq_len -= 1

        if seq_len != expected_tokens:
            latest_sim_raw = None
            latest_topk = []
            latest_overlay = latest_base
            latest_status = f"Token mismatch: seq_len={seq_len}, expected={expected_tokens}. Try NUM_FRAMES=64."
        else:
            tok = tok.reshape(Tt, Gh, Gw, D)
            patch_feats = tok.mean(axis=0)  # (Gh,Gw,D)

            # Normalize patch vectors once (for fast cosine to template)
            feats = patch_feats.astype(np.float32)
            feats_norm = feats / (np.linalg.norm(feats, axis=-1, keepdims=True) + 1e-8)

            # If user just clicked, capture template embedding from that patch
            if anchor_pending and selected_patch is not None:
                ai, aj = selected_patch
                anchor_ref = feats[ai, aj].copy()
                anchor_ref_norm = anchor_ref / (np.linalg.norm(anchor_ref) + 1e-8)
                anchor_pending = False

            # If we have a template, compute similarity-to-template for all patches
            if anchor_ref_norm is not None:
                sim = (feats_norm * anchor_ref_norm[None, None, :]).sum(axis=-1)  # (Gh,Gw)
                latest_sim_raw = sim

                # Tracking step: new position = argmax similarity-to-template
                best_idx = int(np.argmax(sim))
                best_i, best_j = int(best_idx // Gw), int(best_idx % Gw)
                selected_patch = (best_i, best_j)
                best_sim = float(sim[best_i, best_j])

                # Top-K most similar patches excluding the current best match patch
                latest_topk = topk_patches(sim, selected_patch, TOPK)

                # Heatmap (normalized only for visualization)
                sim01 = normalize_01(sim)
                sim_256 = cv2.resize(sim01, (crop_size, crop_size), interpolation=cv2.INTER_CUBIC)

                out = latest_base.copy()
                if SHOW_HEATMAP:
                    out = overlay_heatmap(out, sim_256, alpha=OVERLAY_ALPHA)
                if SHOW_GRID:
                    draw_patch_grid(out, patch_size)

                # Draw tracked location + TopK dots
                if selected_patch is not None:
                    draw_anchor_dot(out, selected_patch, patch_size)
                    draw_topk_dots(out, latest_topk, patch_size)

                latest_overlay = out

                sim_min, sim_mean, sim_max = float(sim.min()), float(sim.mean()), float(sim.max())
                infer_ms = (time.time() - t0) * 1000.0
                lost_flag = "LOST?" if best_sim < LOST_THRESHOLD else "OK"

                # Terminal print (useful for explanation)
                print("\n--- V-JEPA2 tracking report ---")
                print(f"Template is FIXED (from your click). Tracking chooses argmax(similarity-to-template).")
                print(f"Tracked patch: {selected_patch} | best_sim={best_sim:.3f} ({lost_flag}) | D={D} | infer={infer_ms:.0f} ms")
                print(f"Cos sim stats: min={sim_min:.3f}, mean={sim_mean:.3f}, max={sim_max:.3f}")
                print(f"Top-{TOPK} similar patches (excluding tracked best):")
                for rank, (i, j, s) in enumerate(latest_topk, start=1):
                    print(f"  #{rank:02d} patch({i:02d},{j:02d}) sim={s:.3f}")

                latest_status = f"Track={selected_patch} best={best_sim:.2f}({lost_flag}) | D={D} | {infer_ms:.0f}ms"
            else:
                # No template yet: just show base frame + grid (if enabled)
                out = latest_base.copy()
                if SHOW_GRID:
                    draw_patch_grid(out, patch_size)
                latest_overlay = out
                latest_status = "Click on an object patch to set the tracking template."

    show = latest_overlay if latest_overlay is not None else latest_base
    ifn = show is None
    if not show is None:
        # Hover readout
        hover_text = ""
        if anchor_ref_norm is not None and latest_sim_raw is not None and hover_patch is not None:
            hi, hj = hover_patch
            hover_text = f"Hover {hover_patch} cos_to_template={float(latest_sim_raw[hi, hj]):.3f}"

        # UI text
        if len(frames_buf) < NUM_FRAMES:
            header = f"Buffering... {len(frames_buf)}/{NUM_FRAMES}"
        else:
            header = latest_status

        cv2.putText(show, header, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(show, hover_text, (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(show, "Click=set template | r reset | g grid | h heatmap | q quit",
                    (8, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, show)

    key = cv2.waitKey(1) & 0xFF
    if key in [ord("q"), 27]:
        break
    if key == ord("r"):
        selected_patch = None
        hover_patch = None
        anchor_ref = None
        anchor_ref_norm = None
        anchor_pending = False
        latest_sim_raw = None
        latest_topk = []
        latest_overlay = None
        latest_status = "Reset. Buffer frames then click on an object patch."
    if key == ord("g"):
        SHOW_GRID = not SHOW_GRID
        latest_overlay = None
    if key == ord("h"):
        SHOW_HEATMAP = not SHOW_HEATMAP
        latest_overlay = None

cap.release()
cv2.destroyAllWindows()
