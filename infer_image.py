"""
infer_image.py — Single-image GCN inference (14-class Kaggle model)
Usage:
    python infer_image.py --image test_imgs_vids/Fighting003_x264_1040.png
                          --weights weights/best_gcn_weights.pth
"""

import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from collections import deque

# ──────────────────────────────────────────────────────────────────
# 14 UCF-Crime Classes (must match Kaggle training order)
# ──────────────────────────────────────────────────────────────────
CLASSES = [
    'NormalVideos', 'Abuse',    'Arrest',       'Arson',
    'Assault',      'Burglary', 'Explosion',    'Fighting',
    'RoadAccidents','Robbery',  'Shooting',     'Shoplifting',
    'Stealing',     'Vandalism'
]
NUM_CLASSES  = len(CLASSES)   # 14
SEQUENCE_LEN = 30
NUM_JOINTS   = 17

# Class → colour (BGR)
CLASS_COLOURS = {
    'NormalVideos': (0, 200, 0),
    'Fighting':     (0, 0, 255),
    'Assault':      (0, 0, 255),
    'Abuse':        (0, 0, 200),
    'Shooting':     (0, 0, 180),
    'Robbery':      (30, 30, 255),
    'Arrest':       (0, 120, 255),
    'Vandalism':    (0, 60, 255),
}
DEFAULT_COLOUR = (0, 165, 255)   # orange for other suspicious classes


# ──────────────────────────────────────────────────────────────────
# Kaggle Model Architecture  (must match weights exactly)
# ──────────────────────────────────────────────────────────────────

class Graph:
    def __init__(self):
        self.num_node = NUM_JOINTS
        self.neighbor = [
            (15,13),(13,11),(16,14),(14,12),(11,12),
            (5,11),(6,12),(5,6),(5,7),(7,9),(6,8),(8,10),
            (1,2),(0,1),(0,2),(1,3),(2,4),(0,5),(0,6)
        ]
        self.A = self._build_adjacency()

    def _build_adjacency(self):
        A = np.zeros((3, self.num_node, self.num_node))
        for i, j in self.neighbor:
            A[0, i, i] = 1           # self-loop partition
            A[1, i, j] = 1          # neighbour partition
            A[1, j, i] = 1
        return A


class STGCN_Block(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        K = 3
        self.sgcn = nn.Conv2d(in_ch, out_ch * K, kernel_size=1)
        self.tcn  = nn.Sequential(
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, (9, 1), (stride, 1), (4, 0))
        )
        if in_ch != out_ch or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, (stride, 1)),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.residual = lambda x: x

    def forward(self, x, A):
        res     = self.residual(x)
        n, c, t, v = x.size()
        x = self.sgcn(x).view(n, A.size(0), -1, t, v)
        x = torch.einsum('nkctv,kvw->nctw', x, A)
        return self.tcn(x) + res


class ActionRecognitionGCN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        graph   = Graph()
        self.A  = nn.Parameter(
            torch.tensor(graph.A, dtype=torch.float32), requires_grad=False
        )
        self.data_bn = nn.BatchNorm1d(3 * NUM_JOINTS)
        self.layers  = nn.ModuleList([
            STGCN_Block(3,   64),
            STGCN_Block(64,  128, stride=2),
            STGCN_Block(128, 256, stride=2),
        ])
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        n, c, t, v, m = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(n * m, v * c, t)
        x = self.data_bn(x).view(n * m, v, c, t).permute(0, 2, 3, 1)
        for layer in self.layers:
            x = layer(x, self.A)
        x = F.avg_pool2d(x, x.size()[2:]).view(n, m, -1).mean(dim=1)
        return self.fc(x)


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def extract_keypoints(frame, pose_model):
    """
    Run YOLOv8-pose on a single frame; return keypoints (17,3) and bbox or None.
    Tries progressively lower confidence thresholds and upscaling for
    small/distant people common in surveillance footage.
    """
    # Try with increasingly lenient settings
    for conf in (0.25, 0.10, 0.05):
        results = pose_model(frame, classes=0, conf=conf, verbose=False)
        for result in results:
            if result.keypoints is None or len(result.keypoints.data) == 0:
                continue
            if result.boxes is not None and len(result.boxes) > 0:
                best_idx = int(result.boxes.conf.argmax().item())
            else:
                best_idx = 0
            kpts = result.keypoints.data[best_idx].cpu().numpy()   # (17, 3)
            bbox = result.boxes.xyxy[best_idx].cpu().numpy()        # (4,)
            print(f"   ✅  Person found  (conf threshold={conf})")
            return kpts, bbox

    # Last resort: upscale 2× for distant/small subjects
    print("   ⚠️  Retrying with 2× upscale...")
    h, w = frame.shape[:2]
    upscaled = cv2.resize(frame, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
    results  = pose_model(upscaled, classes=0, conf=0.05, verbose=False)
    for result in results:
        if result.keypoints is None or len(result.keypoints.data) == 0:
            continue
        if result.boxes is not None and len(result.boxes) > 0:
            best_idx = int(result.boxes.conf.argmax().item())
        else:
            best_idx = 0
        kpts = result.keypoints.data[best_idx].cpu().numpy()
        bbox = result.boxes.xyxy[best_idx].cpu().numpy()
        # Scale coords back to original resolution
        kpts[:, :2] /= 2.0
        bbox        /= 2.0
        print("   ✅  Person found via 2× upscale")
        return kpts, bbox

    return None, None


def build_sequence(kpts, seq_len=SEQUENCE_LEN):
    """Tile a single keypoint frame into a (seq_len, 17, 3) sequence."""
    return np.tile(kpts[np.newaxis], (seq_len, 1, 1))   # (T, 17, 3)


def preprocess(sequence):
    """
    Normalize to hip-centre, reshape to (1, 3, T, 17, 1).
    sequence: (T, 17, 3)
    """
    seq = sequence.copy().astype(np.float32)
    # Hip-centre normalisation (joints 11 & 12)
    centre = (seq[:, 11, :2] + seq[:, 12, :2]) / 2.0
    seq[:, :, :2] -= centre[:, np.newaxis, :]
    # (T, 17, 3) → (3, T, 17) → (1, 3, T, 17, 1)
    t = np.transpose(seq, (2, 0, 1))
    return np.expand_dims(np.expand_dims(t, 0), -1)


def draw_skeleton(frame, kpts, colour=(0, 255, 0), scale=1.0):
    """Overlay skeleton joints and limbs on the frame (scale adjusts for upscaled images)."""
    LIMBS = [
        (5,7),(7,9),(6,8),(8,10),          # arms
        (5,6),(5,11),(6,12),(11,12),        # torso
        (11,13),(13,15),(12,14),(14,16),    # legs
        (0,1),(0,2),(1,3),(2,4),            # face
        (0,5),(0,6)
    ]
    r         = max(6, int(8 * scale))     # joint dot radius
    thickness = max(3, int(4 * scale))     # limb line thickness
    for x, y, c in kpts:
        if c > 0.2:
            cv2.circle(frame, (int(x), int(y)), r, colour, -1)
            cv2.circle(frame, (int(x), int(y)), r + 1, (255,255,255), 1)  # white outline
    for a, b in LIMBS:
        xa, ya, ca = kpts[a]
        xb, yb, cb = kpts[b]
        if ca > 0.2 and cb > 0.2:
            cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), colour, thickness)
    return frame


def annotate(frame, bbox, label, confidence, colour, scale=1.0):
    """Draw bounding box + label on frame."""
    x1, y1, x2, y2 = map(int, bbox)
    box_thick  = max(3, int(4 * scale))
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, box_thick)

    text       = f"{label}  {confidence*100:.1f}%"
    font_scale = max(0.9, 1.2 * scale)
    thickness  = max(2, int(3 * scale))
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness
    )
    ty = max(y1 - 14, th + 8)
    # Solid background bar
    cv2.rectangle(frame, (x1, ty - th - 8), (x1 + tw + 10, ty + baseline + 4), colour, -1)
    cv2.putText(
        frame, text, (x1 + 5, ty - 4),
        cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
    )
    return frame


def upscale_image(frame, target_min_dim=960):
    """Upscale image so its smaller dimension is at least target_min_dim px."""
    h, w = frame.shape[:2]
    scale = max(target_min_dim / min(h, w), 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC), scale


# ──────────────────────────────────────────────────────────────────
# Main inference
# ──────────────────────────────────────────────────────────────────

def run_inference(image_path: str, weights_path: str, save: bool = True):
    print(f"\n{'─'*55}")
    print(f"  🔍  Image   : {image_path}")
    print(f"  🏋️  Weights : {weights_path}")
    print(f"{'─'*55}\n")

    # ── Load image ──
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # ── Load YOLOv8-pose ──
    print("⚙️  Loading YOLOv8-pose model...")
    pose_model = YOLO("yolov8n-pose.pt")

    # ── Extract keypoints ──
    print("🦴  Detecting pose...")
    kpts, bbox = extract_keypoints(frame, pose_model)

    if kpts is None:
        print("⚠️  No person detected in image. Cannot run GCN inference.")
        cv2.imshow("GCN Inference — No Pose", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    print(f"   ✅  Keypoints shape : {kpts.shape}")

    # ── Build & preprocess sequence ──
    sequence  = build_sequence(kpts, SEQUENCE_LEN)
    gcn_input = preprocess(sequence)                       # (1,3,T,17,1)

    # ── Load GCN model ──
    print("🧠  Loading GCN model weights...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ActionRecognitionGCN(num_classes=NUM_CLASSES).to(device)

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"   ✅  Model on : {device}")

    # ── Inference ──
    x      = torch.tensor(gcn_input, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(x)
        probs  = F.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx    = int(probs.argmax())
    pred_class  = CLASSES[pred_idx]
    confidence  = probs[pred_idx]
    colour      = CLASS_COLOURS.get(pred_class, DEFAULT_COLOUR)

    # ── Print results ──
    print(f"\n{'─'*55}")
    print(f"  🎯  Prediction : {pred_class}  ({confidence*100:.1f}%)")
    print(f"{'─'*55}")
    print("  Top-5 Predictions:")
    for i in probs.argsort()[::-1][:5]:
        bar = '█' * int(probs[i] * 30)
        print(f"    {CLASSES[i]:<15}  {probs[i]*100:5.1f}%  {bar}")
    print(f"{'─'*55}\n")

    # ── Upscale for visibility (surveillance frames are tiny) ──
    display, scale = upscale_image(frame.copy(), target_min_dim=960)

    # Scale kpts & bbox to match upscaled dimensions
    kpts_scaled       = kpts.copy()
    kpts_scaled[:, :2] *= scale
    bbox_scaled        = bbox * scale

    # ── Annotate on upscaled frame ──
    display = draw_skeleton(display, kpts_scaled, colour, scale=scale)
    display = annotate(display, bbox_scaled, pred_class, confidence, colour, scale=scale)

    # ── Save output ──
    if save:
        out_dir  = "test_imgs_vids/output"
        os.makedirs(out_dir, exist_ok=True)
        base     = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(out_dir, f"{base}_gcn_result.png")
        cv2.imwrite(out_path, display)
        print(f"💾  Saved annotated image → {out_path}")
        print(f"   Image size : {display.shape[1]}×{display.shape[0]}  (upscaled {scale:.1f}×)")

    # ── Show in large resizable window ──
    WIN = "🚨 GCN Surveillance — Activity Detection"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, min(display.shape[1], 1400), min(display.shape[0], 900))
    cv2.imshow(WIN, display)
    print("\n[Press any key in the image window to close]")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN single-image inference (14 classes)")
    parser.add_argument(
        "--image",   default="test_imgs_vids/Fighting003_x264_1040.png",
        help="Path to input image"
    )
    parser.add_argument(
        "--weights", default="weights/best_gcn_weights.pth",
        help="Path to trained GCN weights (.pth)"
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Do not save annotated image"
    )
    args = parser.parse_args()

    run_inference(
        image_path   = args.image,
        weights_path = args.weights,
        save         = not args.no_save
    )
