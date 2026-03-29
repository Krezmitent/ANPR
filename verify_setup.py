# verify_setup.py
# ─────────────────────────────────────────────────────────────────────────────
# Run this ONCE after setting up your environment to confirm everything
# is installed correctly and importable.
#
# Usage (with venv activated):
#   python verify_setup.py
#
# All checks should print OK. If any fail, the error message will tell you
# exactly which library is the problem, making it much easier to fix than
# discovering the issue mid-project.
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os

print("=" * 55)
print("  ANPR Project — Environment Verification")
print("=" * 55)

# ── Check 1: Core libraries ───────────────────────────────────────────────────
print("\n[1/4] Checking core libraries...")

try:
    import cv2
    print(f"  OpenCV       {cv2.__version__:>10}   ✓")
except ImportError as e:
    print(f"  OpenCV                     ✗  {e}")

try:
    import numpy as np
    print(f"  NumPy        {np.__version__:>10}   ✓")
except ImportError as e:
    print(f"  NumPy                      ✗  {e}")

try:
    import pandas as pd
    print(f"  Pandas       {pd.__version__:>10}   ✓")
except ImportError as e:
    print(f"  Pandas                     ✗  {e}")

try:
    import easyocr
    print(f"  EasyOCR      {'installed':>10}   ✓")
except ImportError as e:
    print(f"  EasyOCR                    ✗  {e}")

try:
    from ultralytics import YOLO
    print(f"  Ultralytics  {'installed':>10}   ✓")
except ImportError as e:
    print(f"  Ultralytics                ✗  {e}")

try:
    import filterpy
    print(f"  FilterPy     {'installed':>10}   ✓")
except ImportError as e:
    print(f"  FilterPy                   ✗  {e}")

# ── Check 2: SORT tracker ─────────────────────────────────────────────────────
print("\n[2/4] Checking SORT tracker...")

SORT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sort')
if not os.path.isdir(SORT_PATH):
    print(f"  sort/ folder    NOT FOUND  ✗")
    print(f"  Run: git clone https://github.com/abewley/sort.git")
else:
    sys.path.insert(0, SORT_PATH)
    try:
        from sort import Sort
        _ = Sort()   # Instantiate to confirm it fully loads
        print(f"  SORT tracker  {'loaded':>10}   ✓")
    except Exception as e:
        print(f"  SORT tracker               ✗  {e}")

# ── Check 3: Our own modules ──────────────────────────────────────────────────
print("\n[3/4] Checking project modules (src/)...")

try:
    from src.speed_calculator import SpeedCalculator
    sc = SpeedCalculator(fps=30.0, real_distance_meters=10.0)
    result = sc.calculate(100, 160)
    assert result['speed_kmph'] == 18.0, "Speed calculation mismatch"
    print(f"  SpeedCalculator  loaded + tested   ✓  (18.0 km/h @ 60 frames/30fps/10m)")
except Exception as e:
    print(f"  SpeedCalculator                    ✗  {e}")

try:
    from src.vehicle_tracker import VehicleTracker
    print(f"  VehicleTracker    {'loaded':>10}   ✓")
except Exception as e:
    print(f"  VehicleTracker                     ✗  {e}")

try:
    from src.plate_reader import PlateReader
    print(f"  PlateReader       {'loaded':>10}   ✓  (EasyOCR model loads on first use)")
except Exception as e:
    print(f"  PlateReader                        ✗  {e}")

try:
    from src import utils
    print(f"  utils             {'loaded':>10}   ✓")
except Exception as e:
    print(f"  utils                              ✗  {e}")

# ── Check 4: Config and folder structure ──────────────────────────────────────
print("\n[4/4] Checking config and folder structure...")

try:
    import config
    print(f"  config.py         {'loaded':>10}   ✓")
    print(f"    VIDEO_PATH   = {config.VIDEO_PATH}")
    print(f"    LINE_1_Y     = {config.LINE_1_Y}")
    print(f"    LINE_2_Y     = {config.LINE_2_Y}")
    print(f"    REAL_DIST    = {config.REAL_DISTANCE_METERS} m")
except Exception as e:
    print(f"  config.py                          ✗  {e}")

for folder in ['models', 'data', 'output', 'sort', 'src']:
    exists = os.path.isdir(os.path.join(os.path.dirname(__file__), folder))
    status = "✓" if exists else "✗  (create this folder)"
    print(f"  {folder}/    {'exists' if exists else 'MISSING':>10}   {status}")

print("\n" + "=" * 55)
print("  Verification complete.")
print("  If all items show ✓, you are ready to run main.py")
print("=" * 55 + "\n")