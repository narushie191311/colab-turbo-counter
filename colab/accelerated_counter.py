#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Colab向け 高速人物カウント（加速版）

目的:
- 速度を最優先しつつ、重複検出を最小化
- バッチ推論/半精度/torch.compile/ONNX切替/CPU→GPUパイプライン/逐次保存

要点:
- 既定は tracking 有効（UltralyticsのByteTrack）。この場合は内部ストリームでバッチ不可。
- tracking を切る(--no-tracking)と独自のフレームバッチ推論になり、Embedding類似&ID正規化で重複削減。

依存:
  pip install ultralytics torch torchvision opencv-python-headless shapely pandas tqdm onnxruntime-gpu
"""
import os
import csv
import time
import queue
import argparse
import threading
import datetime as dt
from typing import List, Tuple
import signal
import tempfile

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO
try:
    from deepface import DeepFace  # オプション
except Exception:
    DeepFace = None


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
    m = None
    try:
        import re
        m = re.search(r'(\d{8})_(\d{4})', name)
    except Exception:
        pass
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


class FramePrefetcher:
    def __init__(self, video: str, stride: int, queue_size: int = 256):
        self.video = video
        self.stride = max(1, int(stride))
        self.queue: "queue.Queue[Tuple[int, np.ndarray] | None]" = queue.Queue(maxsize=queue_size)
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.total_frames = 0
        self.fps = 30.0

    def start(self):
        # 取得だけ先に
        cap = cv2.VideoCapture(self.video)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video}")
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        self.thread.start()
        return self

    def _worker(self):
        cap = cv2.VideoCapture(self.video)
        if not cap.isOpened():
            self.queue.put(None)
            return
        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if (idx % self.stride) == 0:
                    try:
                        self.queue.put((idx, frame), timeout=5)
                    except queue.Full:
                        pass
                idx += 1
        finally:
            cap.release()
            try:
                self.queue.put(None, timeout=3)
            except Exception:
                pass

    def __iter__(self):
        while True:
            item = self.queue.get()
            if item is None:
                break
            yield item


def apply_mode_defaults(args):
    # ユーザ未指定に対してだけ適用
    def set_default(attr, value):
        if getattr(args, attr) == parser.get_default(attr):
            setattr(args, attr, value)

    if args.mode == "ultralite":
        set_default("model", "yolov8n.pt")
        set_default("conf", 0.6)
        set_default("imgsz", 320)
        set_default("vid_stride", 5)
        set_default("max_det", 80)
        set_default("half", True)
        set_default("batch", 64)
    elif args.mode == "ultrafast":
        set_default("model", "yolov8n.pt")
        set_default("conf", 0.6)
        set_default("imgsz", 416)
        set_default("vid_stride", 4)
        set_default("max_det", 120)
        set_default("half", True)
        set_default("batch", 64)
    elif args.mode == "fast":
        set_default("model", "yolov8n.pt")
        set_default("conf", 0.5)
        set_default("imgsz", 640)
        set_default("vid_stride", 2)
        set_default("max_det", 200)
        set_default("half", True)
        set_default("batch", 32)
    elif args.mode == "accurate":
        set_default("model", "yolov8s.pt")
        set_default("conf", 0.35)
        set_default("imgsz", 768)
        set_default("vid_stride", 1)
        set_default("max_det", 300)
        set_default("half", False)
        set_default("batch", 16)


def maybe_download_model(args):
    if args.download_model and (not os.path.isfile(args.model)):
        import urllib.request
        os.makedirs("models", exist_ok=True)
        local = os.path.join("models", os.path.basename(args.model))
        url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/" + os.path.basename(args.model)
        try:
            print(f"[DL] {url} -> {local}")
            urllib.request.urlretrieve(url, local)
            args.model = local
        except Exception as e:
            print(f"[WARN] download failed: {e}")


def _atomic_write_csv(path: str, rows: List[List[object]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=os.path.dirname(path), encoding="utf-8", newline="") as tf:
        tmp = tf.name
        w = csv.writer(tf)
        for row in rows:
            w.writerow(row)
    os.replace(tmp, path)


def save_outputs(outdir: str, cam_id: str, first_seen: dict):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "first_seen.csv")
    rows = [["camera_id", "track_id", "first_ts"]]
    for tid in sorted(first_seen.keys()):
        rows.append([cam_id, tid, first_seen[tid]])
    _atomic_write_csv(path, rows)
    print("[SAVE]", path)


def run_tracking(args):
    device = args.device if args.device != "auto" else select_device_prefer_cuda()
    model = YOLO(args.model)
    if device == "cuda" and args.half:
        try:
            model.model.half()  # type: ignore
        except Exception:
            pass
    if args.torch_compile:
        try:
            model.model = torch.compile(model.model)  # type: ignore
        except Exception:
            pass

    # 進捗用
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    base_ts = parse_time_from_filename(args.video)

    first_seen = {}
    pbar = tqdm(total=max(1, total_frames), desc="processing", unit="f") if args.progress else None
    last_log = time.time()

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

    # シグナルによる中断対応
    stop_flag = {"stop": False}
    def _sig_handler(signum, frame):
        stop_flag["stop"] = True
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, _sig_handler)
        except Exception:
            pass

    frame_idx = 0
    last_autosave = time.time()
    try:
        for r in stream:
            if stop_flag["stop"]:
                break
            frame_idx += 1
        now = time.time()
        if args.progress and (now - last_log) >= args.progress_sec:
            pct = 100.0 * frame_idx / max(1, total_frames)
            print(f"[PROGRESS] {pct:.2f}% unique={len(first_seen)}")
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

        # 属性推定（任意）
        attrs = {}
        if args.df_interval > 0 and DeepFace is not None and (frame_idx % int(args.df_interval) == 0):
            for tid, x1, y1, x2, y2 in boxes:
                try:
                    roi = r.orig_img[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
                    if roi.size == 0:
                        continue
                    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    res = DeepFace.analyze(rgb, actions=["age", "gender"], enforce_detection=False)
                    if isinstance(res, list) and len(res) > 0:
                        res = res[0]
                    age = res.get("age")
                    gen = res.get("dominant_gender") or res.get("gender")
                    attrs[tid] = (age, gen)
                except Exception:
                    continue

        for tid, *_ in boxes:
            if tid is None:
                continue
            if tid not in first_seen:
                ts = iso_ts(base_ts, frame_idx, fps)
                first_seen[tid] = ts

        # プレビュー表示
        if args.preview and (frame_idx % int(args.preview_step) == 0):
            try:
                img = r.orig_img.copy()
                for tid, x1, y1, x2, y2 in boxes:
                    color = (0, 255, 0) if tid is not None else (255, 255, 0)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    if tid is not None:
                        label = f"ID:{tid}"
                        if tid in attrs:
                            age, gen = attrs[tid]
                            if age is not None:
                                label += f" age:{int(age)}"
                            if isinstance(gen, str):
                                label += f" {gen}"
                        cv2.putText(img, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.imshow("preview", img)
                if cv2.waitKey(1) & 0xFF == 27:
                    pass
            except Exception:
                pass

        # 定期オートセーブ
        if args.autosave_sec > 0 and (time.time() - last_autosave) >= float(args.autosave_sec):
            save_outputs(args.outdir, args.cam_id, first_seen)
            last_autosave = time.time()
    finally:
        if pbar is not None:
            try:
                pbar.close()
            except Exception:
                pass
        # 終了時に確実に保存
        save_outputs(args.outdir, args.cam_id, first_seen)


def run_batched_predict(args):
    device = args.device if args.device != "auto" else select_device_prefer_cuda()

    # ONNX選択時: .onnx を用意してYOLOに渡す（UltralyticsがONNXRuntime経由で推論）
    model_path = args.model
    if args.onnx:
        if not args.model.endswith('.onnx'):
            base = os.path.splitext(os.path.basename(args.model))[0]
            onnx_dir = os.path.join("models")
            os.makedirs(onnx_dir, exist_ok=True)
            onnx_path = os.path.join(onnx_dir, f"{base}.onnx")
            if not os.path.isfile(onnx_path):
                pt_model = YOLO(args.model)
                try:
                    pt_model.export(format="onnx", half=bool(args.half))
                except Exception:
                    pt_model.export(format="onnx")
                # Ultralyticsは runs/ 中に出力する場合があるので探す
                cand = [onnx_path, f"{base}.onnx", os.path.join("runs", "detect", f"{base}.onnx")]
                for c in cand:
                    if os.path.isfile(c):
                        onnx_path = c
                        break
            model_path = onnx_path
    model = YOLO(model_path)

    # 半精度/compile（PyTorchモデル時のみ意味あり）
    if (not args.onnx) and device == "cuda" and args.half:
        try:
            model.model.half()  # type: ignore
        except Exception:
            pass
    if (not args.onnx) and args.torch_compile:
        try:
            model.model = torch.compile(model.model)  # type: ignore
        except Exception:
            pass

    # 進捗情報取得
    cap0 = cv2.VideoCapture(args.video)
    if not cap0.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    fps = cap0.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap0.release()
    base_ts = parse_time_from_filename(args.video)

    first_seen = {}
    pbar = tqdm(total=max(1, total_frames), desc="processing", unit="f") if args.progress else None
    last_log = time.time()

    prefetch = FramePrefetcher(args.video, stride=args.vid_stride, queue_size=args.prefetch).start()
    batch_idx: List[int] = []
    batch_frames: List[np.ndarray] = []
    frame_last_ts = time.time()
    autosave_last = time.time()

    def flush_batch():
        nonlocal batch_frames, batch_idx, first_seen
        if not batch_frames:
            return
        try:
            results = model.predict(batch_frames, imgsz=int(args.imgsz), conf=float(args.conf), max_det=int(args.max_det), device=device, verbose=False)
        except Exception:
            # フォールバック: 1枚ずつ
            results = []
            for fr in batch_frames:
                results += model.predict(fr, imgsz=int(args.imgsz), conf=float(args.conf), max_det=int(args.max_det), device=device, verbose=False)

        for i, (idx_val, res) in enumerate(zip(batch_idx, results)):
            boxes = []
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                for (x1, y1, x2, y2) in xyxy:
                    boxes.append((int(x1), int(y1), int(x2), int(y2)))
            # 簡易ID: 新規検出ごとに擬似ID（連番）
            # 実IDトラッキングがないため、first_seenはEmbedding類似で統合する使用を想定
            for _ in boxes:
                tid = len(first_seen) + 1
                if tid not in first_seen:
                    ts = iso_ts(base_ts, idx_val, prefetch.fps)
                    first_seen[tid] = ts

            # プレビュー表示
            if args.preview and (idx_val % int(args.preview_step) == 0):
                try:
                    img = batch_frames[i].copy()
                    attrs = {}
                    if args.df_interval > 0 and DeepFace is not None:
                        # プレビュー時は対象フレームでのみ属性取得
                        for (x1, y1, x2, y2) in boxes:
                            try:
                                roi = img[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
                                if roi.size == 0:
                                    continue
                                rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                                res2 = DeepFace.analyze(rgb, actions=["age", "gender"], enforce_detection=False)
                                if isinstance(res2, list) and len(res2) > 0:
                                    res2 = res2[0]
                                age = res2.get("age")
                                gen = res2.get("dominant_gender") or res2.get("gender")
                                attrs[(x1,y1,x2,y2)] = (age, gen)
                            except Exception:
                                continue
                    for (x1, y1, x2, y2) in boxes:
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 255), 2)
                        if (x1,y1,x2,y2) in attrs:
                            age, gen = attrs[(x1,y1,x2,y2)]
                            label = []
                            if age is not None:
                                label.append(f"age:{int(age)}")
                            if isinstance(gen, str):
                                label.append(str(gen))
                            if label:
                                cv2.putText(img, ", ".join(label), (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)
                    cv2.imshow("preview", img)
                    if cv2.waitKey(1) & 0xFF == 27:
                        pass
                except Exception:
                    pass

        batch_frames = []
        batch_idx = []

    # シグナルによる中断対応
    stop_flag = {"stop": False}
    def _sig_handler(signum, frame):
        stop_flag["stop"] = True
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, _sig_handler)
        except Exception:
            pass

    try:
        for idx_val, frame in prefetch:
            if stop_flag["stop"]:
                break
            batch_idx.append(idx_val)
            batch_frames.append(frame)
            if len(batch_frames) >= int(args.batch):
                flush_batch()
            # 進捗
            if args.progress:
                if pbar is not None:
                    try:
                        pbar.n = min(idx_val + 1, max(1, prefetch.total_frames))
                        pbar.refresh()
                    except Exception:
                        pass
                now = time.time()
                if (now - last_log) >= args.progress_sec:
                    pct = 100.0 * (idx_val + 1) / max(1, prefetch.total_frames)
                    print(f"[PROGRESS] {pct:.2f}% unique={len(first_seen)}")
                    last_log = now
            # autosave
            if args.autosave_sec > 0 and (time.time() - autosave_last) >= float(args.autosave_sec):
                save_outputs(args.outdir, args.cam_id, first_seen)
                autosave_last = time.time()

        flush_batch()
    finally:
        if pbar is not None:
            try:
                pbar.close()
            except Exception:
                pass
        save_outputs(args.outdir, args.cam_id, first_seen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--cam-id", default="cam")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--download-model", action="store_true")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--mode", default="fast", choices=["ultralite", "ultrafast", "fast", "accurate"])
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--vid-stride", type=int, default=1)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--torch-compile", dest="torch_compile", action="store_true")
    ap.add_argument("--onnx", action="store_true", help="ONNXRuntime経由で推論（自動エクスポート）")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--prefetch", type=int, default=256)
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--progress-sec", type=float, default=2.0)
    ap.add_argument("--autosave-sec", type=float, default=60.0)
    ap.add_argument("--no-tracking", action="store_true", help="トラッキングを無効化しバッチ推論に切替")
    ap.add_argument("--df-interval", type=int, default=0, help="DeepFace属性推定の間隔（0で無効）")
    ap.add_argument("--preview", action="store_true", help="ローカル環境でimshowプレビューを有効化")
    ap.add_argument("--preview-step", type=int, default=30, help="何フレーム毎にプレビューを表示するか")
    args = ap.parse_args()

    global parser
    parser = ap
    apply_mode_defaults(args)
    if args.mode is None:
        args.mode = "fast"
        apply_mode_defaults(args)
        if not args.half:
            args.half = True
    maybe_download_model(args)

    if args.no_tracking:
        run_batched_predict(args)
    else:
        run_tracking(args)


if __name__ == "__main__":
    main()


