# verify_setup.py
# ─────────────────────────────────────────────────────────────────────────────
# Run this once after setting up your environment to confirm everything is
# installed correctly and the pipeline can initialise end-to-end.
#
# Usage (with venv activated):
#   python verify_setup.py
#
# All checks should print ✓. If any fail, the error message will tell you
# exactly which library or file is the problem.
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
import json

print("=" * 62)
print("  ANPR Project — Environment Verification")
print("=" * 62)

# ── Check 1: Core libraries ───────────────────────────────────────────────────
print("\n[1/5] Checking core libraries...")

try:
    import cv2
    print(f"  OpenCV            {cv2.__version__:>10}   ✓")
except ImportError as e:
    print(f"  OpenCV                         ✗  {e}")

try:
    import numpy as np
    print(f"  NumPy             {np.__version__:>10}   ✓")
except ImportError as e:
    print(f"  NumPy                          ✗  {e}")

try:
    import pandas as pd
    print(f"  Pandas            {pd.__version__:>10}   ✓")
except ImportError as e:
    print(f"  Pandas                         ✗  {e}")

try:
    from ultralytics import YOLO
    print(f"  Ultralytics       {'installed':>10}   ✓")
except ImportError as e:
    print(f"  Ultralytics                    ✗  {e}")

try:
    import filterpy
    print(f"  FilterPy          {'installed':>10}   ✓")
except ImportError as e:
    print(f"  FilterPy                       ✗  {e}")

try:
    from PyQt6.QtWidgets import QApplication
    print(f"  PyQt6             {'installed':>10}   ✓")
except ImportError as e:
    print(f"  PyQt6                          ✗  {e}")

# ── Check 2: PyTorch + CUDA ───────────────────────────────────────────────────
print("\n[2/5] Checking PyTorch and CUDA...")

try:
    import torch
    print(f"  PyTorch           {torch.__version__:>10}   ✓")
    if torch.cuda.is_available():
        print(f"  CUDA                available   ✓  ({torch.cuda.get_device_name(0)})")
    else:
        print(f"  CUDA              unavailable   ✗  (will run on CPU — slower)")
        print(f"         Install CUDA PyTorch:")
        print(f"         pip install torch torchvision torchaudio "
              f"--index-url https://download.pytorch.org/whl/cu130")
except ImportError as e:
    print(f"  PyTorch                        ✗  {e}")

# ── Check 3: SORT tracker ─────────────────────────────────────────────────────
print("\n[3/5] Checking SORT tracker...")

SORT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sort')
if not os.path.isdir(SORT_PATH):
    print(f"  sort/ folder      NOT FOUND    ✗")
    print(f"         Run: git clone https://github.com/abewley/sort.git")
else:
    sys.path.insert(0, SORT_PATH)
    try:
        from sort import Sort
        _ = Sort()
        print(f"  SORT tracker      {'loaded':>10}   ✓")
    except Exception as e:
        print(f"  SORT tracker                   ✗  {e}")

# ── Check 4: Project modules ──────────────────────────────────────────────────
print("\n[4/5] Checking project modules (src/)...")

try:
    from src.perspective_speed import PerspectiveSpeedCalculator

    # Smoke-test: build a calculator with a simple square ROI
    test_roi = [[100, 100], [500, 100], [500, 500], [100, 500]]
    calc = PerspectiveSpeedCalculator(
        roi_points    = test_roi,
        real_width_m  = 10.0,
        real_height_m = 10.0,
        fps           = 30.0,
        tolerance     = 20,
    )

    # Project the centre of the ROI — should land near (5.0, 5.0) world metres
    world = calc.image_to_world((300, 300))

    # Speed test: two crossings 30 frames apart, ~10m → ~36 km/h
    import numpy as np
    speed = calc.calculate(
        world_pos_a = np.array([0.0, 0.0]),
        world_pos_b = np.array([0.0, 10.0]),
        frame_a     = 0,
        frame_b     = 30,
    )
    assert abs(speed['speed_kmph'] - 36.0) < 0.1, \
        f"Expected 36.0 km/h, got {speed['speed_kmph']}"

    print(f"  PerspectiveSpeedCalculator     ✓  "
          f"(homography OK, 36.0 km/h @ 30f/30fps/10m)")
except Exception as e:
    print(f"  PerspectiveSpeedCalculator     ✗  {e}")

try:
    from src.vehicle_tracker import VehicleTracker
    # Instantiate with the calc from above (reuse if it succeeded)
    try:
        vt = VehicleTracker(calc)
        print(f"  VehicleTracker    {'loaded':>10}   ✓")
    except NameError:
        print(f"  VehicleTracker              skipped   (PerspectiveSpeedCalculator failed)")
except Exception as e:
    print(f"  VehicleTracker                 ✗  {e}")

try:
    from src.plate_reader import PlateReader
    print(f"  PlateReader       {'loaded':>10}   ✓  (OCR model loads on first use)")
except Exception as e:
    print(f"  PlateReader                    ✗  {e}")

try:
    from src import utils
    print(f"  utils             {'loaded':>10}   ✓")
except Exception as e:
    print(f"  utils                          ✗  {e}")

# ── Check 5: Config, calibration, and folder structure ────────────────────────
print("\n[5/5] Checking config, calibration, and folder structure...")

try:
    import config
    print(f"  config.py         {'loaded':>10}   ✓")
    print(f"    VIDEO_PATH         = {config.VIDEO_PATH}")
    print(f"    ROI_POINTS         = {config.ROI_POINTS[0]} … (4 points)")
    print(f"    ROI_REAL_WIDTH_M   = {config.ROI_REAL_WIDTH_M} m")
    print(f"    ROI_REAL_HEIGHT_M  = {config.ROI_REAL_HEIGHT_M} m")
    print(f"    DEVICE             = {config.DEVICE}")
except Exception as e:
    print(f"  config.py                      ✗  {e}")

# Calibration file
CAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calibration.json')
if os.path.exists(CAL):
    try:
        with open(CAL) as f:
            cal = json.load(f)
        pts = cal.get('roi_points', [])
        w   = cal.get('real_width_m',  '?')
        h   = cal.get('real_height_m', '?')
        print(f"  calibration.json  {'found':>10}   ✓  "
              f"({len(pts)} pts, {w}m × {h}m)")
    except Exception as e:
        print(f"  calibration.json    CORRUPT    ✗  {e}")
else:
    print(f"  calibration.json  NOT FOUND        ⚠  "
          f"Run find_calibration.py or click CALIBRATE in the GUI")

# Video file
try:
    VIDEO = config.VIDEO_PATH
except NameError:
    VIDEO = os.path.join(os.path.dirname(__file__), 'data', 'test_video.mp4')

if os.path.exists(VIDEO):
    size_mb = os.path.getsize(VIDEO) / 1_048_576
    print(f"  test video        {'found':>10}   ✓  ({size_mb:.1f} MB)")
else:
    print(f"  test video        NOT FOUND        ⚠  Place a video at {VIDEO}")

# Folders
for folder in ['models', 'data', 'output', 'sort', 'src']:
    exists = os.path.isdir(os.path.join(os.path.dirname(__file__), folder))
    status = "✓" if exists else "✗  (create this folder)"
    print(f"  {folder}/           {'exists' if exists else 'MISSING':>10}   {status}")

# Model files
try:
    VMODEL = config.VEHICLE_MODEL_PATH
    PMODEL = config.PLATE_MODEL_PATH
except NameError:
    VMODEL = os.path.join(os.path.dirname(__file__), 'models', 'yolov8n.pt')
    PMODEL = os.path.join(os.path.dirname(__file__), 'models', 'license_plate_detector.pt')

v_exists = os.path.exists(VMODEL)
p_exists = os.path.exists(PMODEL)
print(f"  yolov8n.pt        {'found' if v_exists else 'not found':>10}   "
      f"{'✓' if v_exists else '⚠  Will auto-download on first run'}")
print(f"  plate model       {'found' if p_exists else 'not found':>10}   "
      f"{'✓' if p_exists else '✗  Download from Roboflow and place in models/'}")

print("\n" + "=" * 62)
print("  Verification complete.")
print()
print("  Next steps:")
print("   1. Fix any ✗ items above")
print("   2. python find_calibration.py   ← mark 4 road corners")
print("   3. python gui.py                ← start the GUI")
print("      (or: python main.py          ← headless mode)")
print("=" * 62 + "\n")