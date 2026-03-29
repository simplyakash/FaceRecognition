<div align="center">

# FaceRecognition

**Detect faces with YOLOv8 → encode with FaceNet → match against a gallery**

*Python · PyTorch · OpenCV · Ultralytics*

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-neural%20inference-ee4c2c?style=flat&logo=pytorch)](https://pytorch.org/)
<br/>

</div>

---

## Table of contents

1. [Overview](#overview)
2. [Pipeline at a glance](#pipeline-at-a-glance)
3. [Requirements](#requirements)
4. [Model weights](#model-weights)
5. [Quick start](#quick-start)
6. [CLI reference](#cli-reference)
7. [Camera and display tips](#camera-and-display-tips)
8. [Documentation](#documentation)
9. [Project layout](#project-layout)

---

## Overview

This repo is a **face detection + recognition** playground:

- **Detection:** Ultralytics **YOLO** with a face checkpoint (`yolov8n-face.pt`).
- **Embeddings:** **InceptionResnetV1** (FaceNet-style) from **facenet_pytorch**, pretrained on **VGGFace2**.
- **Matching:** **Cosine similarity** via L2-normalized dot products against rows in a NumPy **`.npz`** gallery.

You can **build a gallery** from a folder of images, **query a single photo**, run **live webcam** matching, or **verify two images** offline.

---

## Pipeline at a glance

```
Images / webcam
      │
      ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ YOLO faces  │ ──▶ │ 160×160 crop │ ──▶ │ FaceNet 512 │ ──▶ compare to gallery (.npz)
└─────────────┘     └──────────────┘     └─────────────┘
```

---

## Requirements

Install in your environment (example):

```bash
pip install torch torchvision  # or follow pytorch.org for your CUDA build
pip install ultralytics opencv-python facenet-pytorch numpy
```

Typical imports used by the code:

| Area | Packages |
|------|----------|
| Detection | `ultralytics` (YOLO) |
| Encoding | `torch`, `facenet_pytorch` |
| I/O & UI | `opencv-python` (`cv2`) |
| Gallery | `numpy` (`.npz`) |

---

## Model weights

- **`yolov8n-face.pt`** — place at the **working directory** from which you run the scripts (often repo root), or adjust the path in `yolo_facenet_pipeline.py`.  
  A usable public build is often hosted on Hugging Face (e.g. community face-YOLO checkpoints); ensure you comply with that model’s license.

- **FaceNet weights** — loaded automatically on first use via **facenet_pytorch** (`pretrained="vggface2"`).

---

## Quick start

Run from the **repository root** (so `yolov8n-face.pt` and paths resolve as you expect).

### 1. Build a gallery

```bash
python src/batch_extract_embeddings.py path/to/your_photos -o embeddings.npz --crops-dir ./crops
```

Add **`-r`** to include subfolders. Each detected face becomes **one row** in `embeddings.npz` (`embeddings`, `image_paths`, `face_index`, `boxes_xyxy`).

### 2. Match one query image

```bash
python src/yolo_facenet_pipeline.py \
  --match-image path/to/query.jpg \
  --embeddings embeddings.npz \
  --top-k 5 \
  --match-out match_result.png
```

### 3. Live webcam (with single-file snapshot)

No GUI, overwrite one JPEG each frame (good for headless / SSH):

```bash
python src/yolo_facenet_pipeline.py \
  --webcam \
  --embeddings embeddings.npz \
  --dump-file src/recent.jpg \
  --target-fps 5
```

### 4. Compare two images (no gallery)

```bash
python src/yolo_facenet_pipeline.py src/face1.jpeg src/face2.jpeg
```

### 5. Probe which camera index works

```bash
python cvtestcam.py
```

---

## CLI reference

### `yolo_facenet_pipeline.py` (main)

| Mode | Example |
|------|---------|
| Verify two paths | `python src/yolo_facenet_pipeline.py imgA.jpg imgB.jpg` |
| Match vs gallery | `--match-image FILE --embeddings FILE.npz` |
| Webcam | `--webcam --embeddings FILE.npz` |

Useful flags:

| Flag | Purpose |
|------|---------|
| `--camera N` | Fixed device index (omit for auto-probe) |
| `--max-camera-scan N` | Upper index when auto-probing |
| `--headless` | No window; print scores |
| `--dump-dir DIR` | Save numbered `frame_XXXXXX.jpg` |
| `--dump-file PATH` | **Single image path**, overwritten each frame (use `.jpg` or extension auto-fixed) |
| `--out-video FILE.mp4` | Record combined view |
| `--target-fps F` | Sleep-based loop cap; also sets writer FPS when recording |
| `--max-frames N` | Stop after N frames |
| `--faces largest\|first\|all` | With `--match-image`, which faces to score |
| `--top-k K` | Show **K** gallery neighbors |
| `--match-out PATH` | Save query vs top-K **collage** image |

### `batch_extract_embeddings.py`

| Flag | Purpose |
|------|---------|
| `input_dir` | Folder of images (optional; default `sample_images`) |
| `-o` / `--output` | Output `.npz` (default `embeddings.npz`) |
| `--crops-dir` | Save `*_faceN.png` crops (default `crops`) |
| `-r` / `--recursive` | Walk subfolders |

---

## Camera and display tips

- **“No working camera”** — close other apps using the device; on Linux check `video` group and try `--camera 1`. After a crash: `fuser -v /dev/video0` / `sudo fuser -k /dev/video0` (use with care).
- **Qt / “could not connect to display”** — use **`--headless`**, **`--dump-file`**, or **`--out-video`**, or fix `DISPLAY` / Wayland / `ssh -X`.

---

## Documentation

| File | Contents |
|------|----------|
| [docs/pipeline_flowchart.md](docs/pipeline_flowchart.md) | Mermaid flowcharts, library map, gallery math |
| [FLOW.md](FLOW.md) | Extended flow / notes (if present) |

---

## Project layout

```
FaceRecognition/
├── README.md                 ← you are here
├── cvtestcam.py              ← probe OpenCV camera indices
├── src/
│   ├── yolo_facenet_pipeline.py   ← detection, embedding, match, webcam, verify
│   └── batch_extract_embeddings.py ← folder → embeddings.npz (+ crops)
└── docs/
    └── pipeline_flowchart.md
```

---

<div align="center">

<sub>Built for experimentation and learning—tune thresholds and datasets for your own use case.</sub>

</div>
