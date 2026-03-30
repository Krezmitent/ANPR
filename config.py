# config.py
# ─────────────────────────────────────────────────────────────────────────────
# Central configuration file for the ANPR project.
#
# Speed measurement is now calibrated with 4 road points instead of two
# horizontal lines.  Run  python find_calibration.py  (or click CALIBRATE in
# the GUI) to set ROI_POINTS, ROI_REAL_WIDTH_M, and ROI_REAL_HEIGHT_M for
# your specific camera and road.  The calibration tool saves a
# calibration.json file in the project root; config.py loads it automatically.
#
# If no calibration.json exists the DEFAULT values below are used.  They are
# intentionally set to obviously wrong numbers so you know you need to
# calibrate — do not rely on them for real speed readings.
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import torch

# Auto-select GPU / CPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[Config] Using device: {DEVICE}")

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VIDEO_PATH         = os.path.join(BASE_DIR, "data",   "test_video.mp4")
VEHICLE_MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8n.pt")
PLATE_MODEL_PATH   = os.path.join(BASE_DIR, "models", "license_plate_detector.pt")
OUTPUT_DIR         = os.path.join(BASE_DIR, "output")
OUTPUT_CSV         = os.path.join(OUTPUT_DIR, "detections.csv")
OUTPUT_VIDEO       = os.path.join(OUTPUT_DIR, "annotated_output.avi")
DEBUG_FRAMES_DIR   = os.path.join(OUTPUT_DIR, "debug_frames")
CALIBRATION_FILE   = os.path.join(BASE_DIR, "calibration.json")

# ── ROI Calibration ───────────────────────────────────────────────────────────
#
# ROI_POINTS is a list of four [x, y] pixel coordinates (in 1080p space) that
# mark the four corners of a rectangular section of road.  Order:
#
#   [0] top-left     — far-left  corner (upper part of frame)
#   [1] top-right    — far-right corner (upper part of frame)
#   [2] bottom-right — near-right corner (lower part of frame)
#   [3] bottom-left  — near-left  corner (lower part of frame)
#
# ROI_REAL_WIDTH_M  = real-world width  of that rectangle (metres, left→right)
# ROI_REAL_HEIGHT_M = real-world height of that rectangle (metres, far→near)
#
# These defaults are placeholders — run the calibration tool to set correct values.
#
# ── Loading priority ──────────────────────────────────────────────────────────
#   1. calibration.json  (written by find_calibration.py or GUI calibration)
#   2. hardcoded defaults below

_DEFAULT_ROI_POINTS = [
    [400,  350],   # 0  top-left
    [1520, 350],   # 1  top-right
    [1800, 750],   # 2  bottom-right
    [120,  750],   # 3  bottom-left
]
_DEFAULT_REAL_WIDTH_M  = 7.0
_DEFAULT_REAL_HEIGHT_M = 20.0

if os.path.exists(CALIBRATION_FILE):
    try:
        with open(CALIBRATION_FILE, 'r') as _f:
            _cal = json.load(_f)
        ROI_POINTS         = _cal['roi_points']
        ROI_REAL_WIDTH_M   = float(_cal['real_width_m'])
        ROI_REAL_HEIGHT_M  = float(_cal['real_height_m'])
        print(f"[Config] Loaded calibration from {CALIBRATION_FILE}")
        print(f"         ROI: {ROI_POINTS}")
        print(f"         {ROI_REAL_WIDTH_M}m wide × {ROI_REAL_HEIGHT_M}m deep")
    except Exception as _e:
        print(f"[Config] WARNING: could not read calibration.json ({_e}). "
              "Using defaults — run find_calibration.py to calibrate.")
        ROI_POINTS        = _DEFAULT_ROI_POINTS
        ROI_REAL_WIDTH_M  = _DEFAULT_REAL_WIDTH_M
        ROI_REAL_HEIGHT_M = _DEFAULT_REAL_HEIGHT_M
else:
    print("[Config] No calibration.json found — using default ROI placeholders.")
    print("         Run  python find_calibration.py  or click CALIBRATE in the GUI.")
    ROI_POINTS        = _DEFAULT_ROI_POINTS
    ROI_REAL_WIDTH_M  = _DEFAULT_REAL_WIDTH_M
    ROI_REAL_HEIGHT_M = _DEFAULT_REAL_HEIGHT_M

# ── Detection Thresholds ──────────────────────────────────────────────────────

VEHICLE_CONF = 0.5
PLATE_CONF   = 0.4

VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

# ── SORT Tracker Parameters ───────────────────────────────────────────────────

SORT_MAX_AGE       = 30
SORT_MIN_HITS      = 3
SORT_IOU_THRESHOLD = 0.3

# ── Line Crossing Tolerance ───────────────────────────────────────────────────
# Perpendicular pixel distance from the vehicle's ground point to a trigger
# line that counts as a crossing.  Larger values help when FRAME_SKIP is high
# (vehicles jump more pixels per frame).

LINE_TOLERANCE = 30

# ── Frame Skipping ────────────────────────────────────────────────────────────

FRAME_SKIP = 2

# ── Display ───────────────────────────────────────────────────────────────────

SHOW_VIDEO = True
SAVE_VIDEO = False

COLOR_VEHICLE = (255, 100,  0)   # Orange  — vehicle boxes
COLOR_PLATE   = (  0, 255,  0)   # Green   — plate boxes & text
COLOR_ROI     = (  0,   0, 255)  # Red     — ROI polygon border
COLOR_LINE_A  = (  0, 180, 255)  # Amber   — Line A (far)
COLOR_LINE_B  = (  0, 180, 255)  # Amber   — Line B (near)
COLOR_SPEED   = (  0, 255, 255)  # Yellow  — speed text