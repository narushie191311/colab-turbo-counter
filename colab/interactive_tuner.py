#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ローカル向け インタラクティブ・パラメータ調整ツール

機能:
- cv2.imshow でオーバーレイ表示
- キー操作で conf/imgsz/max_det/vid_stride、同一人物統合の重み・しきい値をリアルタイム変更
- 逐次保存（first_seen/last_seen）、アトミック書き込み

依存:
  pip install ultralytics torch torchvision opencv-python shapely pandas tqdm
"""
import os
import csv
import time
import argparse
import tempfile
import datetime as dt

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from torchvision import models, transforms as T


def select_device_prefer_cuda() -> str:
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def parse_time_from_filename(path: str):
    import re
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


def _atomic_write_csv(path: str, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=os.path.dirname(path), encoding="utf-8", newline="") as tf:
        tmp = tf.name
        w = csv.writer(tf)
        for row in rows:
            w.writerow(row)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--outdir", default="interactive_out")
    ap.add_argument("--cam-id", default="cam")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--max-det", type=int, default=120)
    ap.add_argument("--vid-stride", type=int, default=1)
    ap.add_argument("--alias-threshold", type=float, default=0.92)
    ap.add_argument("--alias-w-embed", type=float, default=0.6)
    ap.add_argument("--alias-w-color", type=float, default=0.3)
    ap.add_argument("--alias-w-height", type=float, default=0.1)
    ap.add_argument("--autosave-sec", type=float, default=30.0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    device = args.device if args.device != "auto" else select_device_prefer_cuda()
    model = YOLO(args.model)

    embedder = SimpleEmbedder(device="cuda" if torch.cuda.is_available() else "cpu")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    base_ts = parse_time_from_filename(args.video)

    first_seen = {}
    last_seen = {}
    alias = {}
    canon_embed = {}
    canon_color = {}
    canon_height = {}

    def canonical_tid(tid: int) -> int:
        cur = tid
        visited = set()
        while cur in alias and cur not in visited:
            visited.add(cur)
            cur = alias[cur]
        return cur

    def maybe_alias(new_tid: int, roi_img: np.ndarray, box_h_norm: float) -> int:
        vec = None
        try:
            vec = embedder.embed(roi_img)
        except Exception:
            pass
        ch = None
        try:
            ch = compute_color_hist(roi_img)
        except Exception:
            pass
        best_sim = -1.0
        best_can = None
        keys = set(list(canon_embed.keys()) + list(canon_color.keys()) + list(canon_height.keys()))
        for can_id in keys:
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
        # register new canonical
        canon_embed[new_tid] = vec
        canon_color[new_tid] = ch
        canon_height[new_tid] = box_h_norm
        return new_tid

    def save_csvs():
        # first_last_attrs.csv の保存
        rows = [["camera_id", "canonical_id", "first_ts", "last_ts"]]
        for cid in sorted(first_seen.keys()):
            rows.append([args.cam_id, cid, first_seen[cid], last_seen.get(cid, first_seen[cid])])
        _atomic_write_csv(os.path.join(args.outdir, "first_last_attrs.csv"), rows)
        print("[SAVE]", os.path.join(args.outdir, "first_last_attrs.csv"))

    cv2.namedWindow("tuner", cv2.WINDOW_NORMAL)
    frame_idx = 0
    last_autosave = time.time()
    imgsz_choices = [320, 416, 512, 640, 768]

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if (frame_idx % max(1, int(args.vid_stride))) != 0:
            frame_idx += 1
            continue
        frame_disp = frame.copy()

        res = model.predict(frame, classes=[0], conf=float(args.conf), imgsz=int(args.imgsz), max_det=int(args.max_det), device=device, verbose=False)
        det = res[0]
        boxes = []
        if det.boxes is not None and len(det.boxes) > 0:
            xyxy = det.boxes.xyxy.cpu().numpy()
            for (x1, y1, x2, y2) in xyxy:
                boxes.append((int(x1), int(y1), int(x2), int(y2)))

        for x1, y1, x2, y2 in boxes:
            x1c = max(0, x1); y1c = max(0, y1); x2c = max(x1c+1, x2); y2c = max(y1c+1, y2)
            roi = frame[y1c:y2c, x1c:x2c]
            fh = float(y2c - y1c) / max(1.0, frame.shape[0])
            can = maybe_alias(len(first_seen) + len(alias) + 1, roi, fh) if roi.size > 0 else (len(first_seen) + len(alias) + 1)
            can = canonical_tid(can)
            ts = iso_ts(parse_time_from_filename(args.video), frame_idx, fps)
            if can not in first_seen:
                first_seen[can] = ts
            last_seen[can] = ts
            cv2.rectangle(frame_disp, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame_disp, f"ID:{can}", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        # HUD
        hud = f"conf:{args.conf:.2f} imgsz:{args.imgsz} max_det:{args.max_det} stride:{args.vid_stride} thr:{args.alias_threshold:.2f} w:[{args.alias_w_embed:.2f},{args.alias_w_color:.2f},{args.alias_w_height:.2f}] unique:{len(first_seen)}"
        cv2.putText(frame_disp, hud, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,255,50), 2)
        cv2.imshow("tuner", frame_disp)

        if args.autosave_sec > 0 and (time.time() - last_autosave) >= float(args.autosave_sec):
            save_csvs()
            last_autosave = time.time()

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord('s'):
            save_csvs()
        elif key == ord('+') or key == ord('='):
            args.conf = min(0.99, args.conf + 0.02)
        elif key == ord('-') or key == ord('_'):
            args.conf = max(0.01, args.conf - 0.02)
        elif key == ord(','):
            # smaller imgsz
            try:
                i = imgsz_choices.index(args.imgsz)
                args.imgsz = imgsz_choices[max(0, i - 1)]
            except Exception:
                args.imgsz = 416
        elif key == ord('.'):
            # larger imgsz
            try:
                i = imgsz_choices.index(args.imgsz)
                args.imgsz = imgsz_choices[min(len(imgsz_choices) - 1, i + 1)]
            except Exception:
                args.imgsz = 416
        elif key == ord('['):
            args.max_det = max(10, args.max_det - 10)
        elif key == ord(']'):
            args.max_det = min(500, args.max_det + 10)
        elif key == ord('{'):
            args.vid_stride = max(1, args.vid_stride - 1)
        elif key == ord('}'):
            args.vid_stride = min(10, args.vid_stride + 1)
        elif key == ord('1'):
            args.alias_threshold = max(0.50, round(args.alias_threshold - 0.01, 2))
        elif key == ord('!'):
            args.alias_threshold = min(0.99, round(args.alias_threshold + 0.01, 2))
        elif key == ord('2'):
            args.alias_w_embed = min(1.0, args.alias_w_embed + 0.05)
        elif key == ord('@'):
            args.alias_w_embed = max(0.0, args.alias_w_embed - 0.05)
        elif key == ord('3'):
            args.alias_w_color = min(1.0, args.alias_w_color + 0.05)
        elif key == ord('#'):
            args.alias_w_color = max(0.0, args.alias_w_color - 0.05)
        elif key == ord('4'):
            args.alias_w_height = min(1.0, args.alias_w_height + 0.05)
        elif key == ord('$'):
            args.alias_w_height = max(0.0, args.alias_w_height - 0.05)
        # 正規化（合計が0ならデフォルトに戻す）
        sw = args.alias_w_embed + args.alias_w_color + args.alias_w_height
        if sw <= 1e-6:
            args.alias_w_embed, args.alias_w_color, args.alias_w_height = 0.6, 0.3, 0.1
        else:
            args.alias_w_embed /= sw; args.alias_w_color /= sw; args.alias_w_height /= sw

        frame_idx += 1

    # 終了処理
    save_csvs()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


if __name__ == "__main__":
    main()


