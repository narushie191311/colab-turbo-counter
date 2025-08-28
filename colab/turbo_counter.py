#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Colab向け 超高速(ターボ)人物カウント: GUIなし・オーバーレイなし・初回観測のみCSV出力。

出力:
  - first_seen.csv: camera_id,track_id,first_ts
  - (任意) turbo_embeddings.npz: key="{camera_id}_{track_id}" -> L2正規化Embedding(np.float32)

依存:
  pip install ultralytics torch torchvision opencv-python-headless shapely pandas tqdm
"""
import os
import re
import csv
import time
import argparse
import datetime as dt
import numpy as np
import cv2
import torch
from tqdm import tqdm
from ultralytics import YOLO
from torchvision import models, transforms as T


def select_device_prefer_cuda() -> str:
    try:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
    except Exception:
        pass
    return "cpu"


def parse_time_from_filename(path: str):
    name = os.path.basename(path)
    m = re.search(r'(\d{8})_(\d{4})', name)
    if not m:
        return None
    try:
        return dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M")
    except Exception:
        return None


def iso_ts(base_ts, frame_idx: int, fps: float) -> str:
    seconds = frame_idx / max(float(fps), 1e-6)
    if base_ts is None:
        return (dt.datetime(1970, 1, 1) + dt.timedelta(seconds=seconds)).isoformat()
    return (base_ts + dt.timedelta(seconds=seconds)).isoformat()


class SimpleEmbedder:
    def __init__(self, device: str):
        self.device = device
        # 軽量で十分: ResNet18の最終層を除去
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = torch.nn.Identity()
        self.model.eval().to(self.device)
        self.tf = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.inference_mode()
    def embed(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = self.tf(rgb).unsqueeze(0).to(self.device)
        feat = self.model(x).detach().cpu().numpy().reshape(-1)
        n = np.linalg.norm(feat) + 1e-9
        return (feat / n).astype(np.float32)


def run(args):
    os.makedirs(args.outdir, exist_ok=True)

    # 進捗計算用に総フレーム/映像fpsを取得
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    base_ts = parse_time_from_filename(args.video)

    device = args.device
    if device == "auto":
        device = select_device_prefer_cuda()

    model = YOLO(args.model)

    # Embedding
    embedder = None
    if args.alias_mode in ("multi", "embed") or args.save_embeddings:
        embedder = SimpleEmbedder(device="cuda" if torch.cuda.is_available() else "cpu")

    first_seen = {}  # canonical_tid -> ts string
    embs = {}        # key -> vector (for saving)
    alias = {}       # tid -> canonical_tid

    # マルチ手がかり用に保持（canonical_tid -> features）
    canon_embed = {}   # canonical_tid -> np.ndarray (L2 normalized)
    canon_color = {}   # canonical_tid -> np.ndarray (L2 normalized HSV hist)
    canon_height = {}  # canonical_tid -> float (0-1)

    def canonical_tid(tid: int) -> int:
        cur = tid
        visited = set()
        while cur in alias and cur not in visited:
            visited.add(cur)
            cur = alias[cur]
        return cur

    def compute_color_hist(bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256])
        vec = np.concatenate([h_hist.flatten(), s_hist.flatten(), v_hist.flatten()]).astype(np.float32)
        n = np.linalg.norm(vec) + 1e-9
        return (vec / n).astype(np.float32)

    def sim_cos(a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None:
            return 0.0
        return float(np.dot(a, b))

    def sim_height(h1: float, h2: float) -> float:
        d = abs(h1 - h2)
        return float(max(0.0, 1.0 - d))

    def maybe_alias(new_tid: int, roi_img: np.ndarray, box_h_norm: float) -> int:
        # 特徴作成
        vec = None
        if embedder is not None:
            try:
                vec = embedder.embed(roi_img)
            except Exception:
                vec = None
        ch = None
        try:
            ch = compute_color_hist(roi_img)
        except Exception:
            ch = None

        # 類似探索
        best_sim = -1.0
        best_can = None
        for can_id in canon_height.keys() | canon_embed.keys() | canon_color.keys():
            se = sim_cos(vec, canon_embed.get(can_id)) if vec is not None and canon_embed.get(can_id) is not None else 0.0
            sc = sim_cos(ch, canon_color.get(can_id)) if ch is not None and canon_color.get(can_id) is not None else 0.0
            sh = sim_height(box_h_norm, canon_height.get(can_id, box_h_norm)) if canon_height.get(can_id) is not None else 0.0
            sim = float(args.alias_w_embed) * se + float(args.alias_w_color) * sc + float(args.alias_w_height) * sh
            if sim > best_sim:
                best_sim = sim
                best_can = can_id

        if best_sim >= float(args.alias_threshold) and best_can is not None:
            alias[new_tid] = int(best_can)
            return int(best_can)

        # 新規canonicalとして登録
        canon_embed[new_tid] = vec
        canon_color[new_tid] = ch
        canon_height[new_tid] = box_h_norm
        return new_tid

    # Ultralyticsストリーム
    stream = model.track(
        source=args.video,
        classes=[0],
        conf=float(args.conf),
        imgsz=int(args.imgsz),
        vid_stride=int(args.vid_stride),
        max_det=int(args.max_det),
        device=device,
        stream=True,
        tracker="bytetrack.yaml",
        verbose=False,
    )

    frame_idx = 0
    last_log = time.time()
    pbar = None
    if args.progress:
        pbar = tqdm(total=max(1, total_frames), desc="processing", unit="f")

    for r in stream:
        frame_idx += 1
        # 進捗表示
        now = time.time()
        if args.progress and (now - last_log) >= args.progress_sec:
            pct = 100.0 * frame_idx / max(1, total_frames)
            print(f"[PROGRESS] ts={iso_ts(base_ts, frame_idx, fps)}  {pct:.2f}%  people={len(first_seen)}")
            last_log = now

        if pbar is not None:
            try:
                pbar.n = min(frame_idx, max(1, total_frames))
                pbar.refresh()
            except Exception:
                pass

        boxes = []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy() if r.boxes.id is not None else [None] * len(xyxy)
            for (x1, y1, x2, y2), tid in zip(xyxy, ids):
                tid_i = None if tid is None else int(tid)
                boxes.append((tid_i, int(x1), int(y1), int(x2), int(y2)))

        # 初回観測 + エイリアス（重複削減）
        for tid, x1, y1, x2, y2 in boxes:
            if tid is None:
                continue
            # ROI
            x1c = max(0, x1); y1c = max(0, y1); x2c = max(x1c+1, x2); y2c = max(y1c+1, y2)
            roi = r.orig_img[y1c:y2c, x1c:x2c]

            can = tid
            if roi.size > 0:
                if args.alias_mode == "none":
                    can = tid
                elif args.alias_mode in ("multi", "embed"):
                    fh = float(y2c - y1c) / max(1.0, r.orig_img.shape[0])
                    can = maybe_alias(tid, roi, fh)
                can = canonical_tid(can)

            if can not in first_seen:
                ts = iso_ts(base_ts, frame_idx, fps)
                first_seen[can] = ts
                # 保存用Embedding（NPZ）
                if embedder is not None and roi.size > 0:
                    try:
                        vec = embedder.embed(roi)
                        embs[f"{args.cam_id}_{can}"] = vec
                    except Exception:
                        pass

    if pbar is not None:
        try:
            pbar.close()
        except Exception:
            pass

    # 保存
    first_path = os.path.join(args.outdir, "first_seen.csv")
    with open(first_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["camera_id", "track_id", "first_ts"])
        for tid in sorted(first_seen.keys()):
            w.writerow([args.cam_id, tid, first_seen[tid]])
    print("[SAVE]", first_path)

    if args.save_embeddings and embedder is not None and len(embs) > 0:
        np.savez(os.path.join(args.outdir, "turbo_embeddings.npz"), **embs)
        print("[SAVE]", os.path.join(args.outdir, "turbo_embeddings.npz"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--cam-id", default="cam")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--vid-stride", type=int, default=1)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--progress-sec", type=float, default=2.0)
    # 同一人物統合モードと重み
    ap.add_argument("--alias-mode", default="multi", choices=["multi", "embed", "none"], help="同一人物推定の手がかりモード")
    ap.add_argument("--alias-threshold", type=float, default=0.92, help="統合の類似度しきい値（0-1）")
    ap.add_argument("--alias-w-embed", type=float, default=0.6, help="埋め込み類似の重み")
    ap.add_argument("--alias-w-color", type=float, default=0.3, help="色ヒスト類似の重み")
    ap.add_argument("--alias-w-height", type=float, default=0.1, help="身長近似類似の重み")
    ap.add_argument("--save-embeddings", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()


