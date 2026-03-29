# main.py
# ─────────────────────────────────────────────────────────────────────────────
# Entry point for the ANPR system.
# This file's only job is to wire all the pieces together and run the main loop.
# It should read almost like a plain-English description of the pipeline:
#
#   1. Load models and open video
#   2. For each frame:
#       a. Detect vehicles
#       b. Update tracker
#       c. For each tracked vehicle, detect plate and read OCR
#       d. Update VehicleTracker state (crossings, speed)
#       e. Draw annotations and display
#   3. Save records and clean up
#
# If you find yourself adding complex logic directly in main.py, that's a sign
# it probably belongs in one of the src/ modules instead.
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
import numpy as np
import cv2
import config

# ── Path setup for SORT ───────────────────────────────────────────────────────
# SORT is not a pip package — it's a cloned GitHub repo sitting in our project
# folder. We need to tell Python where to find it before importing it.
# os.path.abspath ensures this works regardless of which directory you run
# main.py from — it always inserts the path relative to THIS file's location.
SORT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sort')
sys.path.insert(0, SORT_PATH)

from sort import Sort
from ultralytics import YOLO

# ── Our own modules ───────────────────────────────────────────────────────────
# Because src/ has an __init__.py, Python treats it as a package.
# This means we can use clean 'from src.X import Y' syntax everywhere.
from src.plate_reader    import PlateReader
from src.vehicle_tracker import VehicleTracker
from src.utils           import (
    draw_vehicle_box, draw_plate_box,
    draw_reference_lines, draw_frame_info,
    save_records_to_csv
)
import config


def run():
    # ── 1. Initialise models ──────────────────────────────────────────────────

    print("[Main] Loading vehicle detection model...")
    vehicle_model = YOLO(config.VEHICLE_MODEL_PATH)
    # yolov8n.pt will be auto-downloaded to the models/ folder on first run
    # if it doesn't already exist there.

    print("[Main] Loading license plate detection model...")
    # This model is NOT auto-downloaded — you need to place it in models/ manually.
    # See README.md for the download link.
    if not os.path.exists(config.PLATE_MODEL_PATH):
        print(
            f"[Main] WARNING: Plate model not found at {config.PLATE_MODEL_PATH}\n"
            f"       Plate detection will be skipped until you download the model.\n"
            f"       See README.md for instructions."
        )
        plate_model = None
    else:
        plate_model   = YOLO(config.PLATE_MODEL_PATH) if os.path.exists(config.PLATE_MODEL_PATH) else None

    plate_reader = PlateReader()

    # ── 2. Initialise SORT tracker ────────────────────────────────────────────
    # SORT maintains a Kalman filter for each tracked object and uses IOU
    # (Intersection Over Union) matching to associate detections across frames.
    # These parameters come from config.py — see there for explanations of each.
    sort_tracker = Sort(
        max_age       = config.SORT_MAX_AGE,
        min_hits      = config.SORT_MIN_HITS,
        iou_threshold = config.SORT_IOU_THRESHOLD
    )

    # ── 3. Initialise our vehicle state manager ───────────────────────────────
    vehicle_tracker = VehicleTracker(
        line1_y           = config.LINE_1_Y,
        line2_y           = config.LINE_2_Y,
        real_distance_m   = config.REAL_DISTANCE_METERS,
        fps               = 30.0  # Temporary — will be overwritten once video opens
    )

    # ── 4. Open video source ──────────────────────────────────────────────────
    print(f"[Main] Opening video: {config.VIDEO_PATH}")
    cap = cv2.VideoCapture(config.VIDEO_PATH)
    if not cap.isOpened():
        print(f"[Main] ERROR: Could not open video at {config.VIDEO_PATH}")
        print("       Make sure the file exists in the data/ folder.")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Main] Video: {width}x{height} @ {fps:.1f} FPS")

    # Re-initialise VehicleTracker now that we have the real FPS value.
    vehicle_tracker = VehicleTracker(
        line1_y         = config.LINE_1_Y,
        line2_y         = config.LINE_2_Y,
        real_distance_m = config.REAL_DISTANCE_METERS,
        fps             = fps
    )

    # ── Optional: Set up video writer ─────────────────────────────────────────
    video_writer = None
    if config.SAVE_VIDEO:
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(
            config.OUTPUT_VIDEO, fourcc, fps, (width, height)
        )
        print(f"[Main] Saving annotated video to {config.OUTPUT_VIDEO}")

    # ── 5. Main processing loop ───────────────────────────────────────────────
    frame_num = 0
    print("[Main] Starting processing loop. Press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Main] End of video reached.")
            break

        frame_num += 1

        frame = cv2.resize(frame, (1920, 1080))

        # ── 5a. Detect vehicles with YOLOv8 ──────────────────────────────────
        # verbose=False suppresses YOLOv8's per-frame console output.
        vehicle_results = vehicle_model(frame, conf=config.VEHICLE_CONF, verbose=False, device  = config.DEVICE, half=True)[0]

        # Build two lists from the detections:
        #   detections_for_sort: numpy array that SORT expects [x1, y1, x2, y2, conf]
        #   vehicle_boxes:       plain list of (x1, y1, x2, y2) for our own use
        detections_for_sort = []
        for box in vehicle_results.boxes:
            cls_id = int(box.cls[0])
            if cls_id in config.VEHICLE_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                detections_for_sort.append([x1, y1, x2, y2, conf])

        # SORT requires a numpy array; if no vehicles detected, pass empty array.
        if detections_for_sort:
            dets_np = np.array(detections_for_sort, dtype=np.float32)
        else:
            dets_np = np.empty((0, 5), dtype=np.float32)

        # ── 5b. Update SORT tracker ───────────────────────────────────────────
        # tracked_objects is a numpy array of shape (N, 5).
        # Each row is [x1, y1, x2, y2, track_id].
        # The track_id is SORT's persistent integer — the same physical car
        # keeps the same ID across frames.
        tracked_objects = sort_tracker.update(dets_np)

        # ── 5c & 5d. Process each tracked vehicle ─────────────────────────────
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = map(int, obj)
            bbox     = (x1, y1, x2, y2)
            track_id = int(track_id)

            plate_text = None

            # Detect license plate within this vehicle's bounding box crop.
            # We crop first so the plate model only searches within the vehicle
            # region — this is both faster and more accurate than searching the
            # whole frame.
            if plate_model is not None:
                vehicle_crop = frame[y1:y2, x1:x2]

                if vehicle_crop.size > 0:
                    plate_results = plate_model(
                        vehicle_crop, conf=config.PLATE_CONF, verbose=False, device=config.DEVICE, half=True
                    )[0]

                    for plate_box in plate_results.boxes:
                        px1, py1, px2, py2 = map(int, plate_box.xyxy[0])
                        plate_crop = vehicle_crop[py1:py2, px1:px2]

                        if plate_crop.size > 0:
                            plate_text = plate_reader.read(plate_crop)

                            if plate_text:
                                draw_plate_box(
                                    frame,
                                    plate_bbox  = (px1, py1, px2, py2),
                                    full_offset = (x1, y1),
                                    plate_text  = plate_text
                                )
                        # Use only the highest-confidence plate detection per vehicle.
                        break

            # Update the vehicle's state in our tracker (crossing detection,
            # speed calculation when both lines are crossed).
            vehicle_tracker.update(
                track_id    = track_id,
                bbox        = bbox,
                current_frame = frame_num,
                plate_text  = plate_text
            )

            # Retrieve the best plate and speed seen so far for this vehicle,
            # so we can display them on the frame even after the crossing is done.
            best_plate = vehicle_tracker.get_plate_for(track_id) or plate_text
            speed      = None  # Speed only becomes available after both crossings

            # Check completed records — if this vehicle just finished, grab its speed
            for record in vehicle_tracker.completed_records:
                if record['track_id'] == track_id:
                    speed = record['speed_kmph']
                    break

            draw_vehicle_box(frame, bbox, track_id, best_plate, speed)

        # ── 5e. Draw HUD elements ─────────────────────────────────────────────
        draw_reference_lines(frame)
        draw_frame_info(frame, frame_num, len(vehicle_tracker.get_active_ids()))

        # ── 5f. Save completed records ────────────────────────────────────────
        # We save incrementally — every time new records are completed, write
        # them to disk immediately. This way, if the program crashes, we don't
        # lose all the data from the session.
        if vehicle_tracker.completed_records:
            save_records_to_csv(vehicle_tracker.completed_records)
            vehicle_tracker.completed_records.clear()

        # ── 5g. Display and optional video save ───────────────────────────────
        if config.SHOW_VIDEO:
            display_frame = cv2.resize(frame, (1280, 720))
            cv2.imshow("ANPR System — press Q to quit", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[Main] Quit signal received.")
                break

        if video_writer:
            video_writer.write(frame)

    # ── 6. Save any remaining records and clean up ────────────────────────────
    if vehicle_tracker.completed_records:
        save_records_to_csv(vehicle_tracker.completed_records)

    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

    print(f"\n[Main] Done. Results saved to {config.OUTPUT_CSV}")


if __name__ == "__main__":
    run()