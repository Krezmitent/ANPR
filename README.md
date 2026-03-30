# ANPR System with Speed Detection

**Automatic Number Plate Recognition** — detects vehicles, reads UK licence plates, calculates perspective-corrected speed, and logs everything to CSV. Includes a full PyQt6 GUI with live video preview and an interactive road calibration tool.

Built as a Computer Science Engineering project at GGSIPU (Batch 2024–28).

---

## What It Does

- Detects vehicles in a video feed using YOLOv8/YOLOv12
- Tracks multiple vehicles simultaneously across frames using SORT
- Detects and reads UK licence plates using `fast-plate-ocr` (European model)
- Calculates vehicle speed using a **4-point perspective calibration** — the user marks a rectangular road section on the video frame and enters its real-world dimensions; OpenCV computes a homography that corrects for camera angle so speed is accurate regardless of where in the frame a vehicle crosses
- Handles traffic in both directions automatically
- Saves all detections to `output/detections.csv`
- Provides a dark-themed PyQt6 GUI with live video preview, stat cards, and detection table

---

## Project Structure

```
C:\Projects\ANPR\
│
├── src/
│   ├── __init__.py              # Makes src a Python package
│   ├── plate_reader.py          # fast-plate-ocr + UK format validation + correction
│   ├── vehicle_tracker.py       # Per-vehicle state, ROI line crossing, character-level voting
│   ├── perspective_speed.py     # 4-point homography, world-space speed calculation
│   └── utils.py                 # Drawing helpers and CSV I/O
│
├── sort/                        # SORT tracker (cloned from GitHub)
│   └── sort.py
│
├── models/
│   ├── yolov8n.pt               # Vehicle detector (auto-downloaded on first run)
│   └── license_plate_detector.pt  # UK plate detector (download manually)
│
├── data/
│   └── your_video.mp4           # Place your test video here
│
├── output/
│   └── detections.csv           # Generated automatically when pipeline runs
│
├── calibration.json             # Written by find_calibration.py or GUI — auto-loaded
├── main.py                      # Headless entry point (no GUI)
├── gui.py                       # PyQt6 GUI entry point
├── config.py                    # All settings and constants
├── find_calibration.py          # Interactive 4-point road calibration tool
├── verify_setup.py              # Environment sanity check
└── requirements.txt             # Python dependencies
```

---

## Setup

### Requirements

- Windows 10/11 (64-bit)
- Python 3.11+ (64-bit)
- NVIDIA GPU with CUDA driver 12.x or 13.x (tested on RTX 3050 with CUDA 13.1)
- Git

### Step 1 — Clone SORT

```powershell
cd C:\Projects\ANPR
git clone https://github.com/abewley/sort.git
```

### Step 2 — Create Virtual Environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Step 3 — Install PyTorch with CUDA (Do This First)

PyTorch must be installed before `ultralytics` to prevent ultralytics from overwriting the CUDA build with a CPU-only version from PyPI.

```powershell
# For CUDA 13.x drivers (e.g. RTX 3050 with latest drivers):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# For CUDA 12.x drivers:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Verify GPU is detected:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# Expected: True  /  NVIDIA GeForce RTX 3050
```

### Step 4 — Install All Other Dependencies

```powershell
pip install -r requirements.txt
```

### Step 5 — Download the Plate Detection Model

Download a YOLOv8/YOLOv12 model trained on UK licence plates and place it in `models/`:

- Search **Roboflow Universe** for `UK number plate detection YOLOv8`
- Rename the file to match `PLATE_MODEL_PATH` in `config.py`

The vehicle detection model (`yolov8n.pt`) downloads automatically on first run.

### Step 6 — Add Your Video

Place a traffic video in `data/` and update `VIDEO_PATH` in `config.py`.

### Step 7 — Verify Setup

```powershell
python verify_setup.py
```

All items should show ✓ before proceeding.

### Step 8 — Calibrate the Road ROI

This is the most important step for accurate speed readings.

**Option A — GUI (recommended):**
```powershell
python gui.py
# Click OPEN VIDEO → then click CALIBRATE
```

**Option B — standalone tool:**
```powershell
python find_calibration.py
```

Both tools show the first frame of your video and ask you to click 4 corners of a rectangular road section, then enter its real-world dimensions. The result is saved to `calibration.json` and loaded automatically on every subsequent run.

---

## Running

### GUI Mode (Recommended)

```powershell
python gui.py
```

Click **OPEN VIDEO** → **CALIBRATE** (first time) → **START**. Results appear in the detection table in real time and are saved to `output/detections.csv`. Use **EXPORT** to save the CSV elsewhere.

### Headless Mode

```powershell
python main.py
```

Processes the video from `config.py` without a GUI. Useful for batch processing or running on a server.

---

## Calibration

### How It Works

Speed measurement uses a **perspective homography** rather than two fixed horizontal lines. The user marks four corners of a rectangular road section whose real-world size is known. OpenCV's `getPerspectiveTransform` computes a 3×3 matrix H that warps those four points into a bird's-eye rectangle where every pixel represents the same real-world distance uniformly — correcting for camera angle, zoom, and lens distortion.

Vehicle speed is then:

```
speed = world_distance / time

world_distance = ||project(ground_point_B) - project(ground_point_A)||  (metres)
time           = (frame_B - frame_A) / FPS                               (seconds)
```

Where `project()` applies H to convert an image pixel coordinate to real-world metres.

### Marking the 4 Points

Click the corners in this exact order:

```
  1 ──────────── 2       ← Line A  (far edge, closer to horizon)
  │              │
  │   road ROI   │
  │              │
  4 ──────────── 3       ← Line B  (near edge, closer to camera)
```

| Point | Location |
|---|---|
| 1 | Top-left — far-left corner |
| 2 | Top-right — far-right corner |
| 3 | Bottom-right — near-right corner |
| 4 | Bottom-left — near-left corner |

### Good Calibration Targets

| Feature | Typical dimensions |
|---|---|
| Single UK lane width | 3.65 m |
| Dual carriageway (2 lanes) | 7.3 m |
| Standard car length | ~4.5 m |
| Pedestrian crossing bar spacing | measure on-site |
| Junction to stop-line | measure on-site |

### Validating Your Calibration

After running the pipeline, check the **DIST** column in the detections table. Every vehicle should show a distance very close to your `ROI_REAL_HEIGHT_M` value (the depth you entered). Consistent large deviations mean the ROI dimensions need adjusting.

---

## Configuration

All tunable values are in `config.py`. ROI geometry is loaded from `calibration.json` if it exists (written by the calibration tool). You only need to edit `config.py` for non-geometry settings:

| Setting | Default | Description |
|---|---|---|
| `VIDEO_PATH` | `data/test_video.mp4` | Path to input video |
| `ROI_POINTS` | (loaded from calibration.json) | 4 × [x, y] corners of the road ROI |
| `ROI_REAL_WIDTH_M` | (loaded from calibration.json) | Real-world width of the ROI (metres) |
| `ROI_REAL_HEIGHT_M` | (loaded from calibration.json) | Real-world depth of the ROI (metres) |
| `LINE_TOLERANCE` | `30` | Pixel tolerance for line-crossing detection |
| `FRAME_SKIP` | `2` | Process every Nth frame |
| `VEHICLE_CONF` | `0.5` | YOLO confidence threshold for vehicles |
| `PLATE_CONF` | `0.4` | YOLO confidence threshold for plates |

---

## How OCR Works

Plate reading uses a multi-stage pipeline:

1. **Plate detection** — YOLOv8/YOLOv12 locates the plate region within the vehicle crop
2. **Preprocessing** — grayscale, 2× upscale, CLAHE contrast enhancement, Gaussian blur
3. **OCR** — `fast-plate-ocr` (European MobileViT model) reads characters
4. **Format detection** — matches against UK plate formats (current AB12CDE, prefix A123BCD, suffix ABC123D)
5. **Position-aware correction** — fixes OCR confusions using format knowledge: `S→5` at digit positions, `0→O` at letter positions, etc.
6. **Strict validation** — rejects any string that still has type mismatches after correction
7. **Character-level voting** — across all frames a vehicle is tracked, each character position is voted on independently. The most common character at each position wins.

---

## Output Format

`output/detections.csv`:

| track_id | plate | speed_kmph | speed_mps | direction | timestamp | ocr_reads | dist_m |
|---|---|---|---|---|---|---|---|
| 12 | KN05ZZK | 38.6 | 10.7 | Top → Bottom | 2026-03-28 21:04:11 | 8 | 19.8 |
| 5 | BG65USJ | 52.1 | 14.5 | Top → Bottom | 2026-03-28 21:04:23 | 5 | 20.1 |

`dist_m` is the perspective-corrected real-world distance travelled between Line A and Line B. It should be close to `ROI_REAL_HEIGHT_M` for every vehicle — use it to validate your calibration.

---

## Known Limitations

- **Rear plates** on receding vehicles are harder to read than front plates on approaching vehicles.
- **Speed accuracy** depends on the precision of `ROI_REAL_WIDTH_M` and `ROI_REAL_HEIGHT_M`. A 10% error in those values produces a 10% error in speed readings.
- **Calibration requires a flat road** — the perspective homography assumes the road is a flat plane. Hills, ramps, or curved roads will reduce accuracy.
- **ONNX Runtime GPU** (`onnxruntime-gpu`) is not compatible with CUDA 13.x drivers. The pipeline runs `fast-plate-ocr` on CPU, which is still fast since the model is only 4.75MB.
- **4K video** is downscaled to 1080p before processing. ROI coordinates must be in 1080p space (0–1920 × 0–1080). The calibration tools handle this automatically.

---

## Key Dependencies

| Library | Version | Purpose |
|---|---|---|
| `torch` + CUDA | 2.6.0+ | GPU-accelerated deep learning |
| `ultralytics` | 8.2.0+ | YOLOv8/YOLOv12 inference |
| `fast-plate-ocr` | latest | Licence plate OCR (European model) |
| `onnxruntime` | latest | ONNX Runtime CPU backend |
| `opencv-python` | 4.9.0+ | Video processing, perspective transform, drawing |
| `PyQt6` | 6.6.0+ | GUI framework |
| `filterpy` | 1.4.5+ | Kalman filtering for SORT |
| `pandas` | 2.1.0+ | CSV output |
| `numpy` | 2.0.0+ | Array maths |

---

## Troubleshooting

**`ImportError: cannot import name 'COLOR_LINE' from 'config'`**
`src/utils.py` has the old import. Change the import line to:
```python
from config import (COLOR_VEHICLE, COLOR_PLATE, COLOR_ROI, COLOR_SPEED, OUTPUT_CSV)
```
Also remove `LINE_1_Y` and `LINE_2_Y` from the same import — they no longer exist in config.

**`OSError: [WinError 1114] c10.dll`**
PyTorch and PyQt6 have a known conflict on Windows when PyQt6 loads before PyTorch. Fix: ensure `import torch` appears before any PyQt6 imports in your entry point. Also install the VC++ Redistributable from `https://aka.ms/vs/17/release/vc_redist.x64.exe`.

**CSV is empty after running**
Your ROI points are likely outside the processed frame area, or vehicles are not entering the ROI. Check that `ROI_POINTS` values are in 1080p range (0–1920 × 0–1080) — run the calibration tool and check the frame it shows.

**DIST column shows values far from ROI_REAL_HEIGHT_M**
Your real-world dimensions are wrong. Re-run calibration and carefully measure the physical distance between Line A and Line B on the actual road.

**`ModuleNotFoundError: No module named 'sort'`**
Run `git clone https://github.com/abewley/sort.git` from inside `C:\Projects\ANPR\`.

**`PlatePrediction object has no attribute 'upper'`**
Access the plate text via `prediction.plate`. This is handled in the current `plate_reader.py`.

**Video preview shows wrong area / no detections**
Calibration may have been done on a different resolution. The pipeline downscales 4K → 1080p before processing. Always use the calibration tools built into this project — they apply the same downscale before showing you the frame to click on.