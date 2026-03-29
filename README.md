# ANPR System with Speed Detection

**Automatic Number Plate Recognition** — detects vehicles, reads UK licence plates, calculates speed, and logs everything to CSV. Includes a full PyQt6 GUI with live video preview.

Built as a Computer Science Engineering project at GGSIPU (Batch 2024–28).

---

## What It Does

- Detects vehicles in a video feed using YOLOv12/YOLOv8
- Tracks multiple vehicles simultaneously across frames using SORT
- Detects and reads UK licence plates using `fast-plate-ocr` (European model)
- Calculates vehicle speed using two calibrated reference lines
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
│   ├── vehicle_tracker.py       # Per-vehicle state, line crossing, character-level voting
│   ├── speed_calculator.py      # Speed computation (distance / time)
│   └── utils.py                 # Drawing helpers and CSV I/O
│
├── sort/                        # SORT tracker (cloned from GitHub)
│   └── sort.py
│
├── models/
│   ├── yolov8n.pt               # Vehicle detector (auto-downloaded on first run)
│   └── license_plate_detector.pt  # UK plate detector (YOLOv12n, download manually)
│
├── data/
│   └── your_video.mp4           # Place your test video here
│
├── output/
│   └── detections.csv           # Generated automatically when pipeline runs
│
├── main.py                      # Headless entry point (no GUI)
├── gui.py                       # PyQt6 GUI entry point
├── config.py                    # All settings and constants
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
# For CUDA 12.x drivers:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# For CUDA 13.x drivers (e.g. RTX 3050 with latest drivers):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
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

---

## Running

### GUI Mode (Recommended)

```powershell
python gui.py
```

Click **OPEN VIDEO** to select a video, then **START** to begin processing. Results appear in the detection table in real time and are saved to `output/detections.csv`. Use **EXPORT** to save the CSV to a custom location.

### Headless Mode

```powershell
python main.py
```

Processes the video configured in `config.py` without a GUI. Faster for batch processing.

---

## Configuration

All tunable values are in `config.py`. The most important ones to set for your specific video:

| Setting | Default | Description |
|---|---|---|
| `VIDEO_PATH` | `data/test_video.mp4` | Path to input video |
| `LINE_1_Y` | `500` | Y pixel coordinate of first speed reference line |
| `LINE_2_Y` | `720` | Y pixel coordinate of second speed reference line |
| `REAL_DISTANCE_METERS` | `8.5` | Real-world distance between the two lines (metres) |
| `LINE_TOLERANCE` | `30` | Pixel tolerance for line crossing detection |
| `FRAME_SKIP` | `2` | Process every Nth frame (1 = every frame, 2 = every other) |
| `VEHICLE_CONF` | `0.5` | YOLO confidence threshold for vehicle detection |
| `PLATE_CONF` | `0.4` | YOLO confidence threshold for plate detection |

### Calibrating Reference Lines

The reference lines define the speed measurement zone. To find the right values for your video, run:

```powershell
python find_calibration.py
```

This opens an interactive window where you click two horizontal positions on the frame. The terminal prints the Y values to copy into `config.py`.

**Important:** If your video is 4K (2160px tall), the pipeline downscales every frame to 1080px before processing. Use `LINE_1_Y` and `LINE_2_Y` values in the 1080p range (0–1080), not the 4K range.

---

## How Speed Is Calculated

```
Speed = Real World Distance / Time

Time = (frame number at Line 2 − frame number at Line 1) / FPS
```

A vehicle is timed between the two reference lines. The real-world distance between them must be measured or estimated from the video (e.g. a standard car is ~4.5m long, a lane width is ~3.5m).

Direction (Top→Bottom or Bottom→Top) is determined automatically by which line the vehicle crosses first.

---

## How OCR Works

Plate reading uses a multi-stage pipeline designed to handle motion blur, perspective distortion, and font confusion:

1. **Plate detection** — YOLOv12n locates the plate region within the vehicle crop
2. **Preprocessing** — grayscale, 2× upscale, CLAHE contrast enhancement, Gaussian blur
3. **OCR** — `fast-plate-ocr` (European MobileViT model) reads characters
4. **Format detection** — matches against UK plate formats (current AB12CDE, prefix A123BCD, suffix ABC123D)
5. **Position-aware correction** — fixes OCR confusions using format knowledge: `S→5` at digit positions, `0→O` at letter positions, etc.
6. **Strict validation** — rejects any string that still has type mismatches after correction
7. **Character-level voting** — across all frames a vehicle is tracked, each character position is voted on independently. The most common character at each position wins.

This means a plate read as `EE1SNER` seven times and `EE15NER` three times will correctly produce `EE15NER` — because position 3 votes `S` over `5`, but the format correction then fixes `S→5` since position 3 in the current format must be a digit.

---

## Output Format

`output/detections.csv`:

| track_id | plate | speed_kmph | speed_mps | direction | timestamp | ocr_reads |
|---|---|---|---|---|---|---|
| 12 | KN05ZZK | 38.6 | 10.7 | Top → Bottom | 2026-03-28 21:04:11 | 8 |
| 5 | BG65USJ | 52.1 | 14.5 | Top → Bottom | 2026-03-28 21:04:23 | 5 |

`ocr_reads` shows how many frames contributed votes for that plate — higher numbers indicate more confident readings.

---

## Known Limitations

- **Rear plates** on receding vehicles are harder to read than front plates on approaching vehicles. OCR accuracy is naturally higher for vehicles moving toward the camera.
- **Speed accuracy** depends on the precision of `REAL_DISTANCE_METERS`. A 20% error in that value produces a 20% error in all speed readings.
- **ONNX Runtime GPU** (`onnxruntime-gpu`) is not compatible with CUDA 13.x drivers (missing `cublasLt64_12.dll`). The pipeline runs `fast-plate-ocr` on CPU, which is still fast since the model is only 4.75MB.
- **4K video** is downscaled to 1080p before processing. This is intentional — YOLO doesn't benefit from 4K resolution and the downscale dramatically reduces inference time.

---

## Key Dependencies

| Library | Version | Purpose |
|---|---|---|
| `torch` + CUDA | 2.6.0+ | GPU-accelerated deep learning |
| `ultralytics` | 8.2.0+ | YOLOv8/YOLOv12 inference |
| `fast-plate-ocr` | latest | Licence plate OCR (European model) |
| `onnxruntime` | latest | ONNX Runtime CPU backend |
| `opencv-python` | 4.9.0+ | Video processing and drawing |
| `PyQt6` | 6.6.0+ | GUI framework |
| `filterpy` | 1.4.5+ | Kalman filtering for SORT |
| `pandas` | 2.1.0+ | CSV output |

---

## Troubleshooting

**`OSError: [WinError 1114] c10.dll`**
PyTorch and PyQt6 have a known conflict on Windows when PyQt6 loads before PyTorch. Fix: ensure `import torch` appears before any PyQt6 imports in your entry point file. Also install VC++ Redistributable from `https://aka.ms/vs/17/release/vc_redist.x64.exe`.

**CSV is empty after running**
Your `LINE_2_Y` value is likely outside the processed frame height. If your video is 4K, remember the pipeline downscales to 1080p — use values between 0 and 1080.

**Video preview shows only top-left corner**
Windows DPI scaling conflict with OpenCV's `imshow()`. The GUI (`gui.py`) handles this correctly. If using `main.py`, add `cv2.resize(frame, (1280, 720))` before `cv2.imshow()`.

**`ModuleNotFoundError: No module named 'sort'`**
The SORT repo wasn't cloned into your project folder, or you're running from the wrong directory. Run `git clone https://github.com/abewley/sort.git` from inside `C:\Projects\ANPR\`.

**`PlatePrediction object has no attribute 'upper'`**
Access the plate text via `prediction.plate`, not directly as a string. This is handled in the current `plate_reader.py`.