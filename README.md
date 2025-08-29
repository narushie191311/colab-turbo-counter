# colab-turbo-counter

Colab向けの超高速人物カウンタ（GUIなし）。YOLO + ByteTrackでトラックIDを生成し、初回観測時刻のみをCSV出力。重複はEmbedding類似でエイリアス統合。A100/L4/T4 GPUで高速動作。

## クイックスタート（Colab / A100最適化）
1. ランタイム: GPU（A100 推奨）を選択
2. 依存導入と実行

total 0
drwxr-xr-x@   4 toshi  staff   128 Aug 29 01:54 .
drwx------+ 204 toshi  staff  6528 Aug 29 01:54 ..
drwxr-xr-x@  12 toshi  staff   384 Aug 29 02:50 .git
drwxr-xr-x@   4 toshi  staff   128 Aug 29 01:54 colab
On branch main
Your branch is up to date with 'origin/main'.

nothing to commit, working tree clean
commit 124975add9d7ac176e0dbc878fccdf9f9b88d28b
Author: narushie191311 <toshiki.narushima@ha-lu.jp>
Date:   Fri Aug 29 01:54:21 2025 +0900

    Add Colab turbo runner (script + notebook)

A	colab/Colab_Run_Turbo.ipynb
A	colab/turbo_counter.py
[main 7391d16] feat(colab): add accelerated_counter with batching/half/compile/ONNX and show_preview utility
 2 files changed, 491 insertions(+)
 create mode 100644 colab/accelerated_counter.py
 create mode 100644 colab/show_preview.py
[main 1ea71e1] docs: add A100 quickstart, accelerated/show_preview usage, modes
 1 file changed, 87 insertions(+)
 create mode 100644 README.md
[main ad38c7b] docs(colab): add A100 quickstart cells for turbo/accelerated/onnx/preview
 1 file changed, 89 insertions(+)
[main aa23d83] feat(colab): add merge_turbo script and notebook cells for cross-day dedupe
 2 files changed, 269 insertions(+)
 create mode 100644 colab/merge_turbo.py
[main 6dbf4f0] feat(accel): add --preview/--preview-step for local imshow in tracking and batch paths
 1 file changed, 30 insertions(+), 1 deletion(-)
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt to 'yolov8n.pt':  84% ━━━━━━━━━━── 5.2/6.2MB 52.3MB/s 0.1s[KDownloading https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt to 'yolov8n.pt': 100% ━━━━━━━━━━━━ 6.2/6.2MB 54.0MB/s 0.1s[KDownloading https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt to 'yolov8n.pt': 100% ━━━━━━━━━━━━ 6.2/6.2MB 53.8MB/s 0.1s
[PROGRESS] 8.37% unique=31
[PROGRESS] 25.99% unique=59
[PROGRESS] 43.61% unique=83
[PROGRESS] 61.23% unique=111
[PROGRESS] 78.85% unique=163
[PROGRESS] 96.48% unique=222
[SAVE] /Users/toshi/Downloads/カメラ２/turbo_local_30m_ultralite/first_seen.csv
[PROGRESS] 0.11% unique=0
[PROGRESS] 3.19% unique=16
[PROGRESS] 6.50% unique=19
[PROGRESS] 10.13% unique=23
[PROGRESS] 13.88% unique=24
[PROGRESS] 17.40% unique=28
[PROGRESS] 20.93% unique=29
[PROGRESS] 24.67% unique=32
[PROGRESS] 28.30% unique=33
[PROGRESS] 31.72% unique=37
[PROGRESS] 34.80% unique=40
[PROGRESS] 37.56% unique=44
[PROGRESS] 40.31% unique=49
[PROGRESS] 43.06% unique=52
[PROGRESS] 45.81% unique=57
[PROGRESS] 48.68% unique=59
[SAVE] /Users/toshi/Downloads/カメラ２/turbo_local_30m_track/first_seen.csv
[main 7b88e04] feat(accel): optional DeepFace overlay via --df-interval; improve CSV generation reliability
 1 file changed, 57 insertions(+), 1 deletion(-)
[main d486aab] feat(accel): robust saving (atomic CSV), periodic autosave in tracking, signal-safe finalize
 1 file changed, 84 insertions(+), 42 deletions(-)
[PROGRESS] 8.37% unique=31
[PROGRESS] 34.80% unique=75
[PROGRESS] 61.23% unique=111
[PROGRESS] 87.67% unique=198
[SAVE] /Users/toshi/Downloads/カメラ２/turbo_local_30m_ultralite/first_seen.csv
[SAVE] /Users/toshi/Downloads/カメラ２/turbo_local_30m_ultralite/first_seen.csv
[PROGRESS] 50.00% unique=0
[SAVE] /Users/toshi/Downloads/カメラ２/turbo_local_30m_track/first_seen.csv
[SAVE] /Users/toshi/Downloads/カメラ２/turbo_local_30m_track/first_seen.csv
[main 0160958] fix(preview): refresh window every frame; draw overlays only on preview_step (both paths)
 1 file changed, 15 insertions(+), 14 deletions(-)
[main f2298ef] feat(turbo): default multi-cue aliasing (embed+color+height) with tunable weights
 1 file changed, 100 insertions(+), 13 deletions(-)
[main ee3b85a] feat(tool): add interactive_tuner for realtime param tuning with imshow and alias weights
 1 file changed, 306 insertions(+)
 create mode 100644 colab/interactive_tuner.py
[main b415994] feat(turbo): add --autosave-sec with atomic writes and periodic saving
 1 file changed, 26 insertions(+), 5 deletions(-)
Downloading: "https://download.pytorch.org/models/resnet18-f37072fd.pth" to /Users/toshi/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
[SAVE] /Users/toshi/Downloads/カメラ２/tuning_out/first_last_attrs.csv
[SAVE] /Users/toshi/Downloads/カメラ２/tuning_out/first_last_attrs.csv
[SAVE] /Users/toshi/Downloads/カメラ２/tuning_out/first_last_attrs.csv
[SAVE] /Users/toshi/Downloads/カメラ２/tuning_out/first_last_attrs.csv
[SAVE] /Users/toshi/Downloads/カメラ２/tuning_out/first_last_attrs.csv
[SAVE] /Users/toshi/Downloads/カメラ２/tuning_out/first_last_attrs.csv
[SAVE] /Users/toshi/Downloads/カメラ２/tuning_out_quick/first_last_attrs.csv

- 出力: out_20250816/first_seen.csv, out_20250816/stats.csv
- Embedding保存:  → turbo_embeddings.npz
- Googleスプレッドシート同期（任意）: 

## モード（速度/精度プリセット）
- （最速・さらに軽量）: imgsz=320, vid_stride=5, max_det=80, conf=0.6, half
- : imgsz=416, vid_stride=4, max_det=120, conf=0.6, half
- （既定）: imgsz=640, vid_stride=2, max_det=200, half
- : yolov8s.pt, imgsz=768, vid_stride=1, half無効, 

上書き例: 。

## A100向けのポイント
-  を基本。から必要に応じ  で調整。
- 初回の重みダウンロードを避けるには:


## 逐次保存
-  で 60秒毎に first_seen.csv と stats.csv を保存（パスもログ出力）。

## 注意
- DeepFace属性（年齢/性別）は重いので既定オフ。必要時に  を設定。

