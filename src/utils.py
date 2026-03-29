# src/utils.py
# ─────────────────────────────────────────────────────────────────────────────
# Responsibility: Reusable helper functions that don't belong to any single
# class but are used in multiple places throughout the project.
#
# Currently contains two categories:
#   1. Drawing helpers — functions that take a frame and annotate it with
#      bounding boxes, text, and reference lines.
#   2. File I/O helpers — functions that save detection records to disk.
#
# Why isolate drawing here?
# Drawing code tends to be verbose and repetitive. Putting it here means
# main.py stays clean and readable — it just calls draw_vehicle_box() rather
# than containing 6 lines of cv2.rectangle and cv2.putText calls inline.
# ─────────────────────────────────────────────────────────────────────────────

import cv2
import pandas as pd
import os
from config import (
    COLOR_VEHICLE, COLOR_PLATE, COLOR_LINE, COLOR_SPEED,
    LINE_1_Y, LINE_2_Y, OUTPUT_CSV
)


# ── Drawing functions ─────────────────────────────────────────────────────────

def draw_vehicle_box(frame, bbox: tuple, track_id: int,
                     plate_text: str | None, speed_kmph: float | None):
    """
    Draw a bounding box around a detected vehicle with its ID, plate, and speed.

    frame:       The current video frame (numpy array, modified in-place).
    bbox:        (x1, y1, x2, y2) bounding box coordinates in pixels.
    track_id:    The SORT-assigned persistent ID for this vehicle.
    plate_text:  The current best OCR reading for this vehicle, or None.
    speed_kmph:  Speed if it has been calculated yet, or None.
    """
    x1, y1, x2, y2 = bbox

    # Draw the vehicle bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_VEHICLE, 2)

    # Build the label string dynamically — only include fields that are known
    label = f"ID:{track_id}"
    if plate_text:
        label += f"  {plate_text}"
    if speed_kmph is not None:
        label += f"  {speed_kmph:.1f} km/h"

    # Draw a filled rectangle behind the text so it's readable on any background
    (text_w, text_h), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
    )
    cv2.rectangle(frame,
                  (x1, y1 - text_h - baseline - 4),
                  (x1 + text_w, y1),
                  COLOR_VEHICLE, -1)   # -1 = filled rectangle

    cv2.putText(frame, label, (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)   # black text


def draw_plate_box(frame, plate_bbox: tuple, full_offset: tuple, plate_text: str):
    """
    Draw a bounding box around the detected license plate.

    plate_bbox:  (px1, py1, px2, py2) — coordinates WITHIN the vehicle crop.
    full_offset: (x1, y1) — top-left corner of the vehicle crop in the full frame.
                 We need this to convert plate coordinates back to full-frame space.
    plate_text:  The OCR-read text to display above the plate box.
    """
    ox, oy = full_offset
    px1, py1, px2, py2 = plate_bbox

    # Convert from crop-relative to full-frame coordinates
    fx1, fy1 = ox + px1, oy + py1
    fx2, fy2 = ox + px2, oy + py2

    cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), COLOR_PLATE, 2)
    cv2.putText(frame, plate_text, (fx1, fy1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_PLATE, 2)


def draw_reference_lines(frame):
    """
    Draw the two horizontal speed-measurement reference lines across the full
    width of the frame, with labels.
    """
    h, w = frame.shape[:2]

    cv2.line(frame, (0, LINE_1_Y), (w, LINE_1_Y), COLOR_LINE, 2)
    cv2.putText(frame, "Line 1", (10, LINE_1_Y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_LINE, 1)

    cv2.line(frame, (0, LINE_2_Y), (w, LINE_2_Y), COLOR_LINE, 2)
    cv2.putText(frame, "Line 2", (10, LINE_2_Y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_LINE, 1)


def draw_frame_info(frame, frame_num: int, active_count: int):
    """
    Draw a small HUD in the top-right corner showing the current frame
    number and how many vehicles are currently being tracked.
    This is helpful during development and debugging.
    """
    h, w = frame.shape[:2]
    info = f"Frame: {frame_num}  Tracking: {active_count}"
    cv2.putText(frame, info, (w - 280, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


# ── File I/O functions ────────────────────────────────────────────────────────

def save_records_to_csv(records: list[dict], path: str = OUTPUT_CSV):
    """
    Save records to CSV, appending if file exists.
    Correctly handles headers by checking if the file has content,
    not just whether it exists.
    """
    if not records:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(records)

    # Check file size rather than just existence —
    # an empty file should still get headers written
    file_has_content = os.path.isfile(path) and os.path.getsize(path) > 0
    df.to_csv(path, mode='a', index=False, header=not file_has_content)
    
    print(f"[Utils] Saved {len(records)} records to {path}")


def save_debug_frame(frame, track_id: int, frame_num: int, output_dir: str):
    """
    Save a single annotated frame to disk for debugging OCR issues.
    Call this whenever you want to inspect what the program is 'seeing'
    for a particular vehicle at a particular frame.

    This is optional — it's purely a debugging aid.
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"id{track_id}_frame{frame_num}.jpg")
    cv2.imwrite(filename, frame)