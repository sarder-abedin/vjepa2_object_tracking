import cv2
import time
import numpy as np
import torch
from collections import deque
from dataclasses import dataclass
import re
from PIL import Image

WINDOW_NAME = "Display"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)  # resizable

# Delay transformers imports until after OpenCV init
from transformers import AutoVideoProcessor, VJEPA2ForVideoClassification
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# YOLO (Ultralytics)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False


# ----------------------------
# Config
# ----------------------------
VJEPA2_MODEL_ID = "facebook/vjepa2-vitl-fpc16-256-ssv2"
CAM_INDEX = 0

NUM_FRAMES = 7
CROP_SIZE = 256
FRAME_SIZE = (CROP_SIZE, CROP_SIZE)

# UI modes
SHOW_GRID = False
SHOW_LATENT = True
SHOW_LATENT_HEATMAP = True

DISPLAY_SCALE = 2  # 1 small, 2 medium, 3 large
TOPK_ACTIONS = 5

# Primary selection (auto)
PRIMARY_EXCLUDE_LABELS_AUTO = {"person"}

# Detector selection
DETECTOR_MODE = "yolo"   # "yolo" or "owl"
SHOW_DETECTIONS = True
PRINT_DETS_TO_CONSOLE = True

# YOLO settings
YOLO_MODEL_ID = "yolov8n.pt"
YOLO_CONF = 0.25
YOLO_IOU = 0.45
YOLO_EVERY_N_FRAMES = 3

# OWL-ViT settings
OWL_MODEL_ID = "google/owlvit-base-patch32"
OWL_SCORE_THRESHOLD = 0.10
OWL_EVERY_N_FRAMES = 6
OWL_PROMPTS_DEFAULT = [
    "person", "hand",
    "bottle", "cup", "mug", "glass",
    "phone", "laptop", "keyboard", "mouse",
    "book", "remote", "pen",
    "chair", "table",
    "bag", "backpack",
    "cat", "dog",
]


# ----------------------------
# Device selection
# ----------------------------
def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_str_for_yolo(device: torch.device) -> str:
    # Ultralytics accepts: "cpu", "mps", or "0" for first CUDA GPU
    if device.type == "cuda":
        return "0"
    if device.type == "mps":
        return "mps"
    return "cpu"


# ----------------------------
# Geometry / utility helpers
# ----------------------------
def iou_xyxy(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    iw = max(0.0, inter_x1 - inter_x0)
    ih = max(0.0, inter_y1 - inter_y0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter + 1e-9
    return float(inter / union)


def bbox_center(b):
    x0, y0, x1, y1 = b
    return (0.5 * (x0 + x1), 0.5 * (y0 + y1))


def center_dist(a, b) -> float:
    ax, ay = bbox_center(a)
    bx, by = bbox_center(b)
    return float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)


def point_in_box(px, py, box) -> bool:
    x0, y0, x1, y1 = box
    return (x0 <= px <= x1) and (y0 <= py <= y1)


def estimate_motion_bbox_from_clip(frames_bgr_256, thresh=25, min_area=400):
    if len(frames_bgr_256) < 2:
        return None

    f0 = cv2.cvtColor(frames_bgr_256[0], cv2.COLOR_BGR2GRAY)
    f1 = cv2.cvtColor(frames_bgr_256[-1], cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(f1, f0)
    _, mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
    mask = cv2.medianBlur(mask, 5)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    best = None
    best_area = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area > best_area:
            best_area = area
            best = c

    if best is None or best_area < min_area:
        return None

    x, y, w, h = cv2.boundingRect(best)
    return (float(x), float(y), float(x + w), float(y + h))


def fill_something_placeholders(action_label: str, names: list[str]) -> str:
    s = action_label
    for name in names:
        if "[something]" in s:
            s = s.replace("[something]", name, 1)
        else:
            break
    for name in names:
        if re.search(r"\bsomething\b", s):
            s = re.sub(r"\bsomething\b", name, s, count=1)
        else:
            break
    return s


# ----------------------------
# Visualization helpers
# ----------------------------
def draw_patch_grid(img_bgr: np.ndarray, patch_px: int, color=(255, 255, 255), thickness=1) -> None:
    h, w = img_bgr.shape[:2]
    for x in range(0, w, patch_px):
        cv2.line(img_bgr, (x, 0), (x, h - 1), color, thickness, cv2.LINE_AA)
    for y in range(0, h, patch_px):
        cv2.line(img_bgr, (0, y), (w - 1, y), color, thickness, cv2.LINE_AA)


def normalize_01(x: np.ndarray) -> np.ndarray:
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn)).astype(np.float32)


def overlay_heatmap(frame_bgr: np.ndarray, heat_01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat_u8 = np.clip(heat_01 * 255.0, 0, 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame_bgr, 1 - alpha, heat_color, alpha, 0)


def make_blank_panel(size: int, text: str) -> np.ndarray:
    panel = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.putText(panel, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def scale_for_display(img_bgr: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return img_bgr
    h, w = img_bgr.shape[:2]
    return cv2.resize(img_bgr, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)


def draw_dets(img_bgr: np.ndarray, dets: list[dict], primary_box=None) -> None:
    for d in dets:
        x0, y0, x1, y1 = float(d["x0"]), float(d["y0"]), float(d["x1"]), float(d["y1"])
        box = (x0, y0, x1, y1)

        is_primary = primary_box is not None and iou_xyxy(box, primary_box) > 0.90
        color = (0, 255, 0) if is_primary else (255, 255, 255)
        thick = 3 if is_primary else 2

        cv2.rectangle(img_bgr, (int(x0), int(y0)), (int(x1), int(y1)), color, thick, cv2.LINE_AA)
        txt = f'{d["label"]} {d["conf"]:.2f}'
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_bgr, (int(x0), max(0, int(y0) - th - 8)), (int(x0) + tw + 8, int(y0)), (0, 0, 0), -1)
        cv2.putText(img_bgr, txt, (int(x0) + 4, int(y0) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


# ----------------------------
# V-JEPA2 helpers
# ----------------------------
def preprocess_clip(frames_bgr, processor, device: torch.device):
    resized_rgb = []
    for f in frames_bgr:
        f = cv2.resize(f, FRAME_SIZE, interpolation=cv2.INTER_AREA)
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        resized_rgb.append(f)
    inputs = processor(resized_rgb, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


def get_encoder_module(model):
    if hasattr(model, "vjepa2"):
        return model.vjepa2
    for name in ["backbone", "encoder", "model", "base_model"]:
        if hasattr(model, name):
            return getattr(model, name)
    return None


def compute_latent_viz_from_tokens(tokens: torch.Tensor, crop_size: int, patch_size: int, proj_2d: np.ndarray | None):
    tok = tokens[0].float().detach().cpu().numpy()
    seq_len, D = tok.shape

    Gh = crop_size // patch_size
    Gw = crop_size // patch_size
    spatial = Gh * Gw

    if (seq_len - 1) % spatial == 0:
        tok = tok[1:]
        seq_len -= 1

    if seq_len % spatial != 0:
        return None, None, proj_2d

    Tt = seq_len // spatial
    tok = tok.reshape(Tt, Gh, Gw, D)
    patch_feats = tok.mean(axis=0).astype(np.float32)

    feats = patch_feats.reshape(-1, D)
    feats_norm = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)

    global_vec = feats_norm.mean(axis=0)
    global_vec = global_vec / (np.linalg.norm(global_vec) + 1e-8)
    sim = (feats_norm * global_vec[None, :]).sum(axis=1).reshape(Gh, Gw)

    sim01 = normalize_01(sim)
    heat_256 = cv2.resize(sim01, (crop_size, crop_size), interpolation=cv2.INTER_CUBIC)

    if proj_2d is None or proj_2d.shape[0] != D:
        rng = np.random.default_rng(0)
        proj_2d = rng.normal(size=(D, 2)).astype(np.float32)

    pts2 = feats_norm @ proj_2d
    pts2 = pts2 - pts2.min(axis=0, keepdims=True)
    denom = (pts2.max(axis=0, keepdims=True) - pts2.min(axis=0, keepdims=True) + 1e-8)
    pts2 = pts2 / denom
    pts2 = (pts2 * 255.0).astype(np.int32)

    scatter = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
    for x, y in pts2:
        x = int(np.clip(x, 0, crop_size - 1))
        y = int(np.clip(y, 0, crop_size - 1))
        cv2.circle(scatter, (x, y), 2, (255, 255, 255), -1, cv2.LINE_AA)

    cv2.putText(scatter, "Latent scatter (proj)", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return heat_256, scatter, proj_2d


# ----------------------------
# Instance-specific primary selection: HSV histogram
# ----------------------------
def clamp_box_xyxy(box, w, h):
    x0, y0, x1, y1 = box
    x0 = float(max(0, min(w - 1, x0)))
    y0 = float(max(0, min(h - 1, y0)))
    x1 = float(max(0, min(w - 1, x1)))
    y1 = float(max(0, min(h - 1, y1)))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return (x0, y0, x1, y1)


def box_area(box):
    x0, y0, x1, y1 = box
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def compute_hsv_hist(frame_bgr_256: np.ndarray, box_xyxy, bins_h=16, bins_s=16) -> np.ndarray | None:
    h, w = frame_bgr_256.shape[:2]
    x0, y0, x1, y1 = clamp_box_xyxy(box_xyxy, w, h)
    x0i, y0i, x1i, y1i = int(x0), int(y0), int(x1), int(y1)
    if x1i <= x0i or y1i <= y0i:
        return None
    roi = frame_bgr_256[y0i:y1i, x0i:x1i]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [bins_h, bins_s], [0, 180, 0, 256])
    hist = cv2.normalize(hist, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX).astype(np.float32)
    return hist


def hist_similarity(h1: np.ndarray | None, h2: np.ndarray | None) -> float:
    if h1 is None or h2 is None:
        return 0.0
    s = float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))  # [-1,1]
    return max(0.0, min(1.0, 0.5 * (s + 1.0)))             # -> [0,1]


@dataclass
class PrimaryState:
    mode: str = "auto"  # "auto" or "manual"
    selected_label: str | None = None
    selected_box: tuple[float, float, float, float] | None = None
    selected_conf: float = 0.0
    selected_hist: np.ndarray | None = None

    def clear(self):
        self.selected_label = None
        self.selected_box = None
        self.selected_conf = 0.0
        self.selected_hist = None


def update_manual_primary_tracking(primary: PrimaryState, dets: list[dict], frame_bgr_256: np.ndarray | None):
    if primary.mode != "manual":
        return
    if primary.selected_label is None or primary.selected_box is None:
        return
    if not dets or frame_bgr_256 is None:
        return

    prev_box = primary.selected_box
    prev_area = box_area(prev_box) + 1e-9

    H, W = frame_bgr_256.shape[:2]
    diag = (W * W + H * H) ** 0.5

    candidates = []
    for d in dets:
        if d["label"] != primary.selected_label:
            continue

        box = (d["x0"], d["y0"], d["x1"], d["y1"])
        iou = iou_xyxy(prev_box, box)
        dist_n = center_dist(prev_box, box) / (diag + 1e-9)

        area = box_area(box) + 1e-9
        size_sim = 1.0 - min(1.0, abs(area - prev_area) / prev_area)

        cand_hist = compute_hsv_hist(frame_bgr_256, box)
        app_sim = hist_similarity(primary.selected_hist, cand_hist)

        if dist_n > 0.45 and iou < 0.02 and app_sim < 0.25:
            continue

        score = (2.0 * iou) + (1.2 * app_sim) + (0.5 * size_sim) - (0.6 * dist_n)
        candidates.append((score, d, box, cand_hist))

    if not candidates:
        return

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, best_d, best_box, best_hist = candidates[0]

    primary.selected_box = best_box
    primary.selected_conf = float(best_d["conf"])

    if primary.selected_hist is not None and best_hist is not None:
        primary.selected_hist = (0.9 * primary.selected_hist + 0.1 * best_hist).astype(np.float32)
    elif primary.selected_hist is None:
        primary.selected_hist = best_hist


def pick_primary_secondary_names(
    primary: PrimaryState,
    dets: list[dict],
    motion_bbox: tuple[float, float, float, float] | None
) -> tuple[list[str], tuple[float, float, float, float] | None]:
    if not dets:
        return ([], None)

    boxes = [(d["x0"], d["y0"], d["x1"], d["y1"]) for d in dets]

    # MANUAL primary
    if primary.mode == "manual" and primary.selected_label is not None and primary.selected_box is not None:
        primary_name = primary.selected_label
        primary_box = primary.selected_box

        secondary_name = None
        if len(dets) >= 2:
            best_j = None
            best_dist = 1e9
            for j, b in enumerate(boxes):
                if dets[j]["label"] == primary_name and iou_xyxy(b, primary_box) > 0.2:
                    continue
                d = center_dist(primary_box, b)
                if d < best_dist:
                    best_dist = d
                    best_j = j
            if best_j is not None:
                secondary_name = dets[best_j]["label"]

        names = [primary_name] + ([secondary_name] if secondary_name else [])
        return (names, primary_box)

    # AUTO primary
    valid_idxs = [i for i, d in enumerate(dets) if d["label"] not in PRIMARY_EXCLUDE_LABELS_AUTO]
    if not valid_idxs:
        valid_idxs = list(range(len(dets)))

    if motion_bbox is not None:
        ious = [iou_xyxy(boxes[i], motion_bbox) for i in valid_idxs]
        primary_idx = valid_idxs[int(np.argmax(ious))]
    else:
        confs = [dets[i]["conf"] for i in valid_idxs]
        primary_idx = valid_idxs[int(np.argmax(confs))]

    primary_name = dets[primary_idx]["label"]
    primary_box = boxes[primary_idx]

    secondary_name = None
    if len(dets) >= 2:
        best_j = None
        best_dist = 1e9
        for j, b in enumerate(boxes):
            if j == primary_idx:
                continue
            d = center_dist(primary_box, b)
            if d < best_dist:
                best_dist = d
                best_j = j
        if best_j is not None:
            secondary_name = dets[best_j]["label"]

    names = [primary_name] + ([secondary_name] if secondary_name else [])
    return (names, primary_box)


# ----------------------------
# Detector: YOLO  (THIS WAS MISSING IN YOUR FILE)
# ----------------------------
def run_yolo_on_frame(yolo_model, frame_bgr_256: np.ndarray, yolo_device: str) -> list[dict]:
    """
    Returns dets in 256x256 coords:
      [{"x0","y0","x1","y1","label","conf"}, ...]
    """
    if yolo_model is None:
        return []

    results = yolo_model.predict(
        source=frame_bgr_256,
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        verbose=False,
        device=yolo_device
    )

    if not results:
        return []

    r0 = results[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return []

    dets = []
    names = r0.names
    xyxy = r0.boxes.xyxy.detach().cpu().numpy()
    cls = r0.boxes.cls.detach().cpu().numpy().astype(int)
    conf = r0.boxes.conf.detach().cpu().numpy()

    for (x0, y0, x1, y1), c, p in zip(xyxy, cls, conf):
        dets.append({
            "x0": float(x0),
            "y0": float(y0),
            "x1": float(x1),
            "y1": float(y1),
            "label": str(names.get(int(c), f"class{int(c)}")),
            "conf": float(p),
        })
    return dets


# ----------------------------
# Detector: OWL-ViT (robust postprocess)
# ----------------------------
def owl_post_process(owl_processor, outputs, H: int, W: int, threshold: float):
    target_sizes = torch.tensor([[H, W]], dtype=torch.long)  # CPU tensor for compatibility

    if hasattr(owl_processor, "post_process_object_detection"):
        return owl_processor.post_process_object_detection(outputs=outputs, threshold=threshold, target_sizes=target_sizes)[0]
    if hasattr(owl_processor, "post_process_grounded_object_detection"):
        return owl_processor.post_process_grounded_object_detection(outputs=outputs, threshold=threshold, target_sizes=target_sizes)[0]

    if hasattr(owl_processor, "image_processor") and hasattr(owl_processor.image_processor, "post_process_object_detection"):
        return owl_processor.image_processor.post_process_object_detection(outputs=outputs, threshold=threshold, target_sizes=target_sizes)[0]
    if hasattr(owl_processor, "image_processor") and hasattr(owl_processor.image_processor, "post_process_grounded_object_detection"):
        return owl_processor.image_processor.post_process_grounded_object_detection(outputs=outputs, threshold=threshold, target_sizes=target_sizes)[0]

    raise AttributeError("No OWL-ViT postprocess function found on processor or image_processor.")


def run_owl_on_frame(owl_processor, owl_model, frame_bgr_256: np.ndarray, prompts: list[str], device: torch.device) -> list[dict]:
    if owl_processor is None or owl_model is None or len(prompts) == 0:
        return []

    frame_rgb = cv2.cvtColor(frame_bgr_256, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)

    inputs = owl_processor(text=[prompts], images=pil_img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = owl_model(**inputs)

    H, W = frame_bgr_256.shape[:2]
    results = owl_post_process(owl_processor, outputs, H, W, OWL_SCORE_THRESHOLD)

    boxes_t = results.get("boxes", None)
    scores_t = results.get("scores", None)
    labels_t = results.get("labels", None)
    text_labels = results.get("text_labels", None)

    if boxes_t is None or scores_t is None or (labels_t is None and text_labels is None):
        return []

    boxes = boxes_t.detach().cpu().numpy()
    scores = scores_t.detach().cpu().numpy()
    if boxes.shape[0] == 0:
        return []

    if text_labels is not None:
        label_names = list(text_labels)
    else:
        labels = labels_t.detach().cpu().numpy().astype(int)
        label_names = []
        for li in labels:
            if 0 <= li < len(prompts):
                label_names.append(prompts[li])
            else:
                label_names.append(f"label{li}")

    order = np.argsort(-scores)
    dets = []
    for idx in order:
        x0, y0, x1, y1 = boxes[idx].tolist()
        dets.append({
            "x0": float(x0),
            "y0": float(y0),
            "x1": float(x1),
            "y1": float(y1),
            "label": str(label_names[idx]),
            "conf": float(scores[idx]),
        })
    return dets


# ----------------------------
# Window sizing
# ----------------------------
def set_window_medium(latent_on: bool):
    base_w = CROP_SIZE * (2 if latent_on else 1)
    base_h = CROP_SIZE
    cv2.resizeWindow(WINDOW_NAME, base_w * DISPLAY_SCALE, base_h * DISPLAY_SCALE)


# ----------------------------
# App state
# ----------------------------
@dataclass
class AppState:
    primary: PrimaryState
    last_dets: list
    last_heat_256: np.ndarray | None
    last_scatter: np.ndarray
    proj_2d: np.ndarray | None
    last_pred_text: str
    frame_idx: int
    last_left_frame_256: np.ndarray | None
    detector_mode: str
    owl_prompts: list[str]
    editing_prompts: bool = False
    prompt_buffer: str = ""


def make_mouse_cb(state: AppState):
    def mouse_cb(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        sx = x / DISPLAY_SCALE
        sy = y / DISPLAY_SCALE

        # click must be in left panel
        if sx < 0 or sy < 0 or sx >= CROP_SIZE or sy >= CROP_SIZE:
            return

        if state.primary.mode != "manual":
            return

        dets = state.last_dets
        if not dets:
            return

        px, py = float(sx), float(sy)
        candidates = []
        for d in dets:
            box = (d["x0"], d["y0"], d["x1"], d["y1"])
            if point_in_box(px, py, box):
                area = max(1.0, box_area(box))
                candidates.append((area, d, box))

        if not candidates:
            return

        candidates.sort(key=lambda t: t[0])  # smallest containing box
        _, best_d, best_box = candidates[0]

        state.primary.selected_label = best_d["label"]
        state.primary.selected_box = best_box
        state.primary.selected_conf = float(best_d["conf"])

        if state.last_left_frame_256 is not None:
            state.primary.selected_hist = compute_hsv_hist(state.last_left_frame_256, best_box)
        else:
            state.primary.selected_hist = None

        print(f"[MANUAL PRIMARY] Selected instance: {state.primary.selected_label}  conf={state.primary.selected_conf:.2f}  box={tuple(map(int, best_box))}")

    return mouse_cb


# ----------------------------
# Main
# ----------------------------
def main():
    global SHOW_GRID, SHOW_LATENT, SHOW_LATENT_HEATMAP
    global DETECTOR_MODE, SHOW_DETECTIONS, PRINT_DETS_TO_CONSOLE

    device = pick_device()
    print(f"Device selected for V-JEPA2/OWL: {device}")

    # V-JEPA2 action model
    v_processor = AutoVideoProcessor.from_pretrained(VJEPA2_MODEL_ID)
    v_model = VJEPA2ForVideoClassification.from_pretrained(VJEPA2_MODEL_ID).to(device).eval()
    patch_size = int(getattr(v_model.config, "patch_size", 16))
    crop_size = int(getattr(v_model.config, "crop_size", CROP_SIZE))
    encoder = get_encoder_module(v_model)

    # Detector objects (lazy load)
    yolo_model = None
    yolo_device = device_str_for_yolo(device)
    owl_processor = None
    owl_model = None

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    frame_buffer = deque(maxlen=NUM_FRAMES)

    state = AppState(
        primary=PrimaryState(mode="auto"),
        last_dets=[],
        last_heat_256=None,
        last_scatter=make_blank_panel(crop_size, "Latent pending..."),
        proj_2d=None,
        last_pred_text="Buffering...",
        frame_idx=0,
        last_left_frame_256=None,
        detector_mode=DETECTOR_MODE,
        owl_prompts=OWL_PROMPTS_DEFAULT[:],
    )

    cv2.setMouseCallback(WINDOW_NAME, make_mouse_cb(state))
    set_window_medium(SHOW_LATENT)

    def ensure_detector_loaded():
        nonlocal yolo_model, owl_processor, owl_model

        if state.detector_mode == "yolo":
            if yolo_model is None:
                if not YOLO_AVAILABLE:
                    print("YOLO unavailable: install with `pip install ultralytics`.")
                    return
                print(f"Loading YOLO: {YOLO_MODEL_ID} (device={yolo_device})")
                yolo_model = YOLO(YOLO_MODEL_ID)
        else:
            if owl_model is None or owl_processor is None:
                print(f"Loading OWL-ViT: {OWL_MODEL_ID} (device={device})")
                owl_processor = OwlViTProcessor.from_pretrained(OWL_MODEL_ID)
                owl_model = OwlViTForObjectDetection.from_pretrained(OWL_MODEL_ID).to(device).eval()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            state.frame_idx += 1
            frame_buffer.append(frame.copy())

            left = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)
            state.last_left_frame_256 = left.copy()

            # --- Detection (periodic) ---
            if SHOW_DETECTIONS:
                ensure_detector_loaded()

                if state.detector_mode == "yolo":
                    run_detector = (state.frame_idx % YOLO_EVERY_N_FRAMES == 0)
                else:
                    run_detector = (state.frame_idx % OWL_EVERY_N_FRAMES == 0)

                if run_detector:
                    if state.detector_mode == "yolo":
                        state.last_dets = run_yolo_on_frame(yolo_model, left, yolo_device) if yolo_model is not None else []
                    else:
                        state.last_dets = run_owl_on_frame(owl_processor, owl_model, left, state.owl_prompts, device) if owl_model is not None else []

                    if PRINT_DETS_TO_CONSOLE:
                        if len(state.last_dets) == 0:
                            print(f"{state.detector_mode.upper()}: no objects")
                        else:
                            summary = ", ".join([f'{d["label"]}:{d["conf"]:.2f}' for d in state.last_dets[:8]])
                            print(f"{state.detector_mode.upper()}: {summary}{' ...' if len(state.last_dets) > 8 else ''}")

            # manual primary tracking (instance-specific)
            update_manual_primary_tracking(state.primary, state.last_dets, state.last_left_frame_256)

            # latent heatmap overlay (stable)
            if SHOW_LATENT and SHOW_LATENT_HEATMAP and state.last_heat_256 is not None:
                left = overlay_heatmap(left, state.last_heat_256, alpha=0.45)

            # --- Action inference when buffer full ---
            primary_box_for_highlight = None
            if len(frame_buffer) == NUM_FRAMES:
                t0 = time.time()
                inputs = preprocess_clip(list(frame_buffer), v_processor, device)

                with torch.no_grad():
                    outputs = v_model(**inputs)

                logits = outputs.logits
                probs = torch.softmax(logits[0].float().cpu(), dim=-1).numpy()
                topk_idx = np.argsort(-probs)[:TOPK_ACTIONS]

                pred_id = int(topk_idx[0])
                pred_label = v_model.config.id2label[pred_id]
                pred_prob = float(probs[pred_id])

                clip_frames_256 = [cv2.resize(f, FRAME_SIZE, interpolation=cv2.INTER_AREA) for f in list(frame_buffer)]
                motion_bbox = estimate_motion_bbox_from_clip(clip_frames_256)

                names, primary_box_for_highlight = pick_primary_secondary_names(state.primary, state.last_dets, motion_bbox)
                pretty_label = fill_something_placeholders(pred_label, names)
                state.last_pred_text = f"{pretty_label} ({pred_prob:.2f})"

                print("\n--- V-JEPA2 (SSV2) Action Prediction ---")
                print(f"Detector: {state.detector_mode.upper()} | Primary mode: {state.primary.mode.upper()}")
                if state.primary.mode == "manual":
                    if state.primary.selected_label:
                        print(f"Manual primary (instance): {state.primary.selected_label} conf={state.primary.selected_conf:.2f}")
                    else:
                        print("Manual primary (instance): (none selected; click a box)")
                print(f"Action (raw):    {pred_label}")
                print(f"Action (filled): {pretty_label}")
                for r, idx in enumerate(topk_idx, start=1):
                    print(f"  #{r}: {v_model.config.id2label[int(idx)]}  prob={float(probs[int(idx)]):.3f}")
                print(f"Infer time: {(time.time() - t0) * 1000:.0f} ms")

                # latent update at inference time
                if SHOW_LATENT and encoder is not None:
                    with torch.no_grad():
                        tokens = None
                        try:
                            enc_out = encoder(**inputs)
                            tokens = enc_out.last_hidden_state if hasattr(enc_out, "last_hidden_state") else enc_out[0]
                        except Exception:
                            tokens = None

                    if tokens is not None:
                        heat_256, scatter, state.proj_2d = compute_latent_viz_from_tokens(
                            tokens=tokens,
                            crop_size=crop_size,
                            patch_size=patch_size,
                            proj_2d=state.proj_2d,
                        )
                        if heat_256 is not None:
                            state.last_heat_256 = heat_256
                        if scatter is not None:
                            state.last_scatter = scatter

                frame_buffer.clear()

            # highlight: manual selection wins
            if state.primary.mode == "manual" and state.primary.selected_box is not None:
                highlight_box = state.primary.selected_box
            else:
                highlight_box = primary_box_for_highlight

            # draw detections
            if SHOW_DETECTIONS and len(state.last_dets) > 0:
                draw_dets(left, state.last_dets, primary_box=highlight_box)

            if SHOW_GRID:
                draw_patch_grid(left, patch_size)

            # stable canvas
            if SHOW_LATENT:
                right = state.last_scatter.copy()
                canvas = np.hstack([left, right])
            else:
                canvas = left

            # prompt edit overlay
            if state.editing_prompts:
                cv2.rectangle(canvas, (6, 54), (canvas.shape[1] - 6, 92), (0, 0, 0), -1)
                msg = f"Edit OWL prompts (comma-separated), Enter=save, Esc=cancel: {state.prompt_buffer}_"
                cv2.putText(canvas, msg[:170], (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 1, cv2.LINE_AA)

            # on-screen text
            det_txt = f"DETECTOR: {state.detector_mode.upper()} (b) | DETS: {'ON' if SHOW_DETECTIONS else 'OFF'} (d)"
            prim_txt = f"PRIMARY: {state.primary.mode.upper()} (m) | click box in MANUAL | u unset"
            cv2.putText(canvas, state.last_pred_text, (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, det_txt, (8, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, prim_txt, (8, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)

            footer = "q/esc quit | b detector | d det on/off | c det console | p edit OWL prompts | m primary | u unset | l latent | h heatmap | g grid"
            cv2.putText(canvas, footer, (8, canvas.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

            canvas_big = scale_for_display(canvas, DISPLAY_SCALE)
            cv2.imshow(WINDOW_NAME, canvas_big)

            key = cv2.waitKey(1) & 0xFF

            # prompt editing consumes keys
            if state.editing_prompts:
                if key in [13, 10]:  # Enter
                    new_list = [p.strip() for p in state.prompt_buffer.split(",") if p.strip()]
                    if len(new_list) > 0:
                        state.owl_prompts = new_list
                        print(f"[OWL PROMPTS] set ({len(state.owl_prompts)}): {', '.join(state.owl_prompts[:20])}{' ...' if len(state.owl_prompts) > 20 else ''}")
                    state.editing_prompts = False
                    state.prompt_buffer = ""
                elif key == 27:  # Esc
                    state.editing_prompts = False
                    state.prompt_buffer = ""
                elif key in [8, 127]:
                    state.prompt_buffer = state.prompt_buffer[:-1]
                else:
                    if 32 <= key <= 126:
                        state.prompt_buffer += chr(key)
                continue

            if key in [ord("q"), 27]:
                break
            if key == ord("g"):
                SHOW_GRID = not SHOW_GRID
            if key == ord("l"):
                SHOW_LATENT = not SHOW_LATENT
                set_window_medium(SHOW_LATENT)
            if key == ord("h"):
                SHOW_LATENT_HEATMAP = not SHOW_LATENT_HEATMAP
            if key == ord("c"):
                PRINT_DETS_TO_CONSOLE = not PRINT_DETS_TO_CONSOLE
                print(f"Detector console printing: {'ON' if PRINT_DETS_TO_CONSOLE else 'OFF'}")
            if key == ord("d"):
                SHOW_DETECTIONS = not SHOW_DETECTIONS
                print(f"Detections overlay: {'ON' if SHOW_DETECTIONS else 'OFF'}")
                if not SHOW_DETECTIONS:
                    state.last_dets = []
                    state.primary.clear()
            if key == ord("m"):
                state.primary.mode = "manual" if state.primary.mode == "auto" else "auto"
                print(f"[PRIMARY MODE] Now: {state.primary.mode.upper()}")
            if key == ord("u"):
                state.primary.clear()
                print("[MANUAL PRIMARY] cleared selection")
            if key == ord("p"):
                state.editing_prompts = True
                state.prompt_buffer = ", ".join(state.owl_prompts)
            if key == ord("b"):
                state.detector_mode = "owl" if state.detector_mode == "yolo" else "yolo"
                print(f"[DETECTOR] Switched to: {state.detector_mode.upper()}")
                state.last_dets = []
                state.primary.clear()

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam released and windows closed.")


if __name__ == "__main__":
    main()