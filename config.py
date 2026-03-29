# config.py
# ─────────────────────────────────────────────────────────────────────────────
# Central configuration file for the ANPR project.
# ALL tunable values live here. When you switch to a different video or camera,
# this is the ONLY file you need to edit. Nothing in src/ or main.py should
# contain hardcoded numbers — they should always import from here.
# ─────────────────────────────────────────────────────────────────────────────

import os
import torch

# Automatically use GPU if available, fall back to CPU if not.
# This means your code will work on both your laptop and any other machine.
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[Config] Using device: {DEVICE}")

# ── Paths ─────────────────────────────────────────────────────────────────────

# Base directory = wherever this config.py file lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input: path to your test video. Use 0 for a webcam feed.
VIDEO_PATH = os.path.join(BASE_DIR, "data", "test_video.mp4") 

# Model weights
VEHICLE_MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8n.pt")
PLATE_MODEL_PATH   = os.path.join(BASE_DIR, "models", "license_plate_detector.pt")

# Output files
OUTPUT_DIR         = os.path.join(BASE_DIR, "output")
OUTPUT_CSV         = os.path.join(OUTPUT_DIR, "detections.csv")
OUTPUT_VIDEO       = os.path.join(OUTPUT_DIR, "annotated_output.avi")
DEBUG_FRAMES_DIR   = os.path.join(OUTPUT_DIR, "debug_frames")

# ── Speed Measurement Calibration ────────────────────────────────────────────
# These two Y pixel coordinates define horizontal lines drawn across the frame.
# A vehicle is timed between these two lines to compute its speed.
#
# HOW TO SET THESE VALUES:
#   1. Run main.py once and pause it (press spacebar, or just note a frame).
#   2. Look at the frame and pick two Y positions that a car will definitely
#      cross — ideally 1/3 and 2/3 down the frame.
#   3. Measure the REAL-WORLD distance between those two positions in meters.
#      You can do this by pausing footage and using a known object for scale
#      (e.g., a standard car is ~4.5m long, a lane is ~3.5m wide).

LINE_1_Y = 860          # Y coordinate (pixels) of the first reference line
LINE_2_Y = 1035          # Y coordinate (pixels) of the second reference line
REAL_DISTANCE_METERS = 27.0   # Real-world distance between the two lines (meters)

# ── Detection Thresholds ──────────────────────────────────────────────────────
# Confidence scores are between 0 and 1. Higher = more strict = fewer detections.
# Lower = more lenient = more false positives.
# 0.5 for vehicles and 0.4 for plates is a good starting point.

VEHICLE_CONF = 0.5
PLATE_CONF   = 0.4

# COCO dataset class IDs for vehicles. YOLOv8n is trained on COCO by default.
# 2=car, 3=motorcycle, 5=bus, 7=truck
VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

# ── SORT Tracker Parameters ───────────────────────────────────────────────────
# max_age:       How many frames a track survives without a matching detection.
#                Increase if vehicles disappear behind obstructions briefly.
# min_hits:      How many frames a detection must appear before it's confirmed
#                as a real track (filters out one-frame ghost detections).
# iou_threshold: How much two bounding boxes must overlap to be considered
#                the same object between frames. IOU = Intersection Over Union.

SORT_MAX_AGE       = 30
SORT_MIN_HITS      = 3
SORT_IOU_THRESHOLD = 0.3

# How many pixels either side of a reference line counts as a crossing.
# With frame skipping active, cars jump more pixels between frames, so
# this needs to be wider than the default 8px to avoid missed crossings.
LINE_TOLERANCE = 30

# Process every Nth frame to improve speed. 2 = process frames 2, 4, 6...
# Set to 1 to process every frame (only if your GPU can keep up in real time).
FRAME_SKIP = 2

# ── Display Settings ──────────────────────────────────────────────────────────
SHOW_VIDEO   = True    # Set False to run headlessly (e.g. on a server)
SAVE_VIDEO   = False   # Set True to write annotated video to OUTPUT_VIDEO

# Colours used for drawing (BGR format — OpenCV uses Blue-Green-Red, not RGB)
COLOR_VEHICLE  = (255, 100, 0)   # Orange for vehicle bounding boxes
COLOR_PLATE    = (0, 255, 0)     # Green for plate bounding boxes and text
COLOR_LINE     = (0, 0, 255)     # Red for speed measurement reference lines
COLOR_SPEED    = (0, 255, 255)   # Yellow for speed text overlay