# main.py  — headless pipeline entry point (no GUI)
# ─────────────────────────────────────────────────────────────────────────────
# Run this directly if you don't need the GUI:
#   python main.py
#
# Before running, calibrate the ROI with:
#   python find_calibration.py
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
import numpy as np
import cv2
import config

SORT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sort')
sys.path.insert(0, SORT_PATH)

from sort import Sort
from ultralytics import YOLO

from src.plate_reader      import PlateReader
from src.vehicle_tracker   import VehicleTracker
from src.perspective_speed import PerspectiveSpeedCalculator
from src.utils import (
    draw_vehicle_box, draw_plate_box,
    draw_frame_info, save_records_to_csv
)


def run():
    # ── Models ────────────────────────────────────────────────────────────────
    print("[Main] Loading vehicle detection model...")
    vehicle_model = YOLO(config.VEHICLE_MODEL_PATH)

    print("[Main] Loading plate detection model...")
    if not os.path.exists(config.PLATE_MODEL_PATH):
        print(f"[Main] WARNING: Plate model not found at {config.PLATE_MODEL_PATH}")
        plate_model = None
    else:
        plate_model = YOLO(config.PLATE_MODEL_PATH)

    plate_reader = PlateReader()

    sort_tracker = Sort(
        max_age       = config.SORT_MAX_AGE,
        min_hits      = config.SORT_MIN_HITS,
        iou_threshold = config.SORT_IOU_THRESHOLD,
    )

    # ── Video ─────────────────────────────────────────────────────────────────
    print(f"[Main] Opening video: {config.VIDEO_PATH}")
    cap = cv2.VideoCapture(config.VIDEO_PATH)
    if not cap.isOpened():
        print(f"[Main] ERROR: Could not open video at {config.VIDEO_PATH}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Main] Video: {width}×{height} @ {fps:.1f} FPS")

    if not os.path.exists(config.CALIBRATION_FILE):
        print("[Main] WARNING: No calibration.json found — using default ROI.")
        print("       Run  python find_calibration.py  to calibrate.")

    # ── Perspective calculator ────────────────────────────────────────────────
    perspective_calc = PerspectiveSpeedCalculator(
        roi_points    = config.ROI_POINTS,
        real_width_m  = config.ROI_REAL_WIDTH_M,
        real_height_m = config.ROI_REAL_HEIGHT_M,
        fps           = fps,
        tolerance     = config.LINE_TOLERANCE,
    )
    print(f"[Main] ROI: {config.ROI_POINTS}")
    print(f"[Main] Real-world: {config.ROI_REAL_WIDTH_M}m × {config.ROI_REAL_HEIGHT_M}m")

    vehicle_tracker = VehicleTracker(perspective_calc)

    # ── Optional video writer ─────────────────────────────────────────────────
    video_writer = None
    if config.SAVE_VIDEO:
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        fourcc       = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(
            config.OUTPUT_VIDEO, fourcc, fps, (width, height)
        )

    # ── Main loop ─────────────────────────────────────────────────────────────
    frame_num = 0
    print("[Main] Starting — press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Main] End of video.")
            break

        frame_num += 1
        frame = cv2.resize(frame, (1920, 1080))

        if frame_num % config.FRAME_SKIP != 0:
            continue

        # Vehicle detection
        vehicle_results = vehicle_model(
            frame, conf=config.VEHICLE_CONF, verbose=False,
            device=config.DEVICE, half=True,
        )[0]

        dets = []
        for box in vehicle_results.boxes:
            if int(box.cls[0]) in config.VEHICLE_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                dets.append([x1, y1, x2, y2, float(box.conf[0])])

        dets_np         = np.array(dets, dtype=np.float32) if dets else np.empty((0, 5))
        tracked_objects = sort_tracker.update(dets_np)

        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = map(int, obj)
            bbox       = (x1, y1, x2, y2)
            track_id   = int(track_id)
            plate_text = None

            if plate_model is not None:
                vehicle_crop = frame[y1:y2, x1:x2]
                if vehicle_crop.size > 0:
                    plate_results = plate_model(
                        vehicle_crop, conf=config.PLATE_CONF, verbose=False,
                        device=config.DEVICE, half=True,
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
                                    plate_text  = plate_text,
                                )
                        break

            vehicle_tracker.update(
                track_id      = track_id,
                bbox          = bbox,
                current_frame = frame_num,
                plate_text    = plate_text,
            )

            best_plate = vehicle_tracker.get_plate_for(track_id) or plate_text
            speed      = None
            for record in vehicle_tracker.completed_records:
                if record['track_id'] == track_id:
                    speed = record['speed_kmph']
                    break

            draw_vehicle_box(frame, bbox, track_id, best_plate, speed)

            # Draw ground contact point
            gx, gy = (x1 + x2) // 2, y2
            cv2.circle(frame, (gx, gy), 4, (0, 255, 255), -1)

        # Draw ROI overlay instead of two horizontal lines
        perspective_calc.draw_roi(frame)
        draw_frame_info(frame, frame_num, len(vehicle_tracker.get_active_ids()))

        if vehicle_tracker.completed_records:
            save_records_to_csv(vehicle_tracker.completed_records)
            vehicle_tracker.completed_records.clear()

        if config.SHOW_VIDEO:
            display = cv2.resize(frame, (1280, 720))
            cv2.imshow("ANPR System — press Q to quit", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[Main] Quit.")
                break

        if video_writer:
            video_writer.write(frame)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if vehicle_tracker.completed_records:
        save_records_to_csv(vehicle_tracker.completed_records)

    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    print(f"\n[Main] Done. Results saved to {config.OUTPUT_CSV}")


if __name__ == "__main__":
    run()