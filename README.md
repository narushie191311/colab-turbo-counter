# colab-turbo-counter

Colab向けの超高速人物カウンタ（GUIなし）。YOLO + ByteTrackでトラックIDを生成し、初回観測時刻のみをCSV出力。重複はEmbedding類似でエイリアス統合。A100/L4/T4 GPUで高速動作。

## クイックスタート（Colab / A100最適化）
1. ランタイム: GPU（A100 推奨）を選択
2. 依存導入と実行

```bash
%cd /content/colab-turbo-counter
!pip -q install ultralytics torch torchvision opencv-python-headless shapely pandas tqdm deepface gspread oauth2client onnxruntime-gpu

# 進捗は2秒おき。fast + half（A100向け）が既定
!python colab/turbo_counter.py \
  --video "/content/drive/MyDrive/アオハル祭/input_videos/merged_20250816_1141-1951.mkv" \
  --outdir "out_20250816" \
  --cam-id 20250816 \
  --device cuda --progress --progress-sec 2 --autosave-sec 60
```

- 出力: out_20250816/first_seen.csv, out_20250816/stats.csv
- Embedding保存: `--save-embeddings` → turbo_embeddings.npz
- Googleスプレッドシート同期（任意）: `--sheets-id "<ID>" --sheets-ws "first_seen" --sheets-ws-stats "stats"`

## 新規スクリプト
- `colab/accelerated_counter.py`: バッチ推論、FP16、torch.compile、ONNX切替、プリフェッチ、追跡ON/OFF対応の高速版
- `colab/show_preview.py`: プレビュー用。Colabではmatplotlib、ローカルは `--use-imshow` でcv2.imshow

### accelerated_counter.py（最速: 追跡OFFのバッチ）
```bash
!python colab/accelerated_counter.py \
  --video "/content/drive/MyDrive/アオハル祭/input_videos/merged_20250816_1141-1951.mkv" \
  --outdir "out_20250816_ultralite" \
  --cam-id 20250816 \
  --device cuda --mode ultralite --half --no-tracking \
  --batch 128 --imgsz 320 --vid-stride 5 --max-det 80 \
  --progress --progress-sec 2 --autosave-sec 60
```

### accelerated_counter.py（追跡あり: 重複抑制重視）
```bash
!python colab/accelerated_counter.py \
  --video "/content/drive/MyDrive/アオハル祭/input_videos/merged_20250816_1141-1951.mkv" \
  --outdir "out_20250816_track" \
  --cam-id 20250816 \
  --device cuda --mode fast --half \
  --progress --progress-sec 2 --autosave-sec 60
```

### accelerated_counter.py（ONNXRuntimeで推論）
```bash
!python colab/accelerated_counter.py \
  --video "/content/drive/MyDrive/アオハル祭/input_videos/merged_20250816_1141-1951.mkv" \
  --outdir "out_20250816_onnx" \
  --cam-id 20250816 \
  --device cuda --mode ultrafast --half --no-tracking --onnx \
  --batch 128 --imgsz 416 --vid-stride 4 --max-det 120 \
  --progress --progress-sec 2
```

### show_preview.py（Colab表示）
```bash
!python colab/show_preview.py --video "/content/drive/MyDrive/アオハル祭/input_videos/merged_20250816_1141-1951.mkv" --step 120 --limit 3
```

## モード（速度/精度プリセット）
- `ultralite`（最速・さらに軽量）: imgsz=320, vid_stride=5, max_det=80, conf=0.6, half
- `ultrafast`: imgsz=416, vid_stride=4, max_det=120, conf=0.6, half
- `fast`（既定）: imgsz=640, vid_stride=2, max_det=200, half
- `accurate`: yolov8s.pt, imgsz=768, vid_stride=1, half無効, `--df-interval 15`

上書き例: `--imgsz 512 --vid-stride 3 --max-det 150`。

## A100向けのポイント
- `--device cuda --half` を基本。`--mode fast/ultrafast/ultralite` から `--batch` 64-128、`--imgsz` 320-416、`--vid-stride` 3-5、`--max-det` 80-150 を調整。
- 初回の重みダウンロードを避けるには:
```bash
!python colab/turbo_counter.py --download-model --model yolov8n.pt \
  --video "..." --outdir "..." --cam-id 0 --device cuda
```

## 逐次保存
- `--autosave-sec 60` で 60秒毎に first_seen.csv と stats.csv を保存（パスもログ出力）。

```text
Repo: https://github.com/narushie191311/colab-turbo-counter
```
