#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Colab向け 簡易プレビュー（数フレームだけimshow表示）

注意:
- Colabの標準ノートブックでは cv2.imshow は使えないため、Matplotlibで表示します。
- ローカル環境なら --use-imshow を付けると cv2.imshow で連続表示します。
"""
import argparse
import cv2
import time
import matplotlib.pyplot as plt


def preview(video: str, step: int, limit: int, use_imshow: bool):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    shown = 0
    idx = 0
    if use_imshow:
        cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
    while shown < limit:
        ok, frame = cap.read()
        if not ok:
            break
        if (idx % step) == 0:
            bgr = frame
            if use_imshow:
                cv2.imshow("preview", bgr)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            else:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                plt.figure(figsize=(6,4))
                plt.imshow(rgb)
                plt.title(f"frame {idx}")
                plt.axis('off')
                plt.show()
                time.sleep(0.1)
            shown += 1
        idx += 1
    cap.release()
    if use_imshow:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--step", type=int, default=60, help="何フレームごとに表示するか")
    ap.add_argument("--limit", type=int, default=5, help="最大表示枚数")
    ap.add_argument("--use-imshow", action="store_true", help="ローカル環境向けにimshowを利用")
    args = ap.parse_args()
    preview(args.video, args.step, args.limit, args.use_imshow)


if __name__ == "__main__":
    main()


