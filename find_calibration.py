#!/usr/bin/env python
# find_calibration.py
# ─────────────────────────────────────────────────────────────────────────────
# Interactive tool that lets you mark four road points on a video frame,
# enter the real-world dimensions, and save the result to calibration.json.
#
# Usage:
#   python find_calibration.py
#   python find_calibration.py --video path/to/video.mp4
#   python find_calibration.py --frame 150        # use frame #150 instead of #1
#
# Click the FOUR CORNERS of a rectangular road section in this order:
#   1. Top-left     (far side of road, left)
#   2. Top-right    (far side of road, right)
#   3. Bottom-right (near side of road, right)
#   4. Bottom-left  (near side of road, left)
#
# "Far" = the edge closer to the horizon / further from the camera.
# "Near" = the edge closest to the camera lens.
#
# Good calibration targets:
#   - White lane markings at two different distances
#   - Road studs / cat's eyes forming a known rectangle
#   - Lane width (standard UK lane ≈ 3.65 m) × a measured road segment
#   - Two pedestrian crossing bars (measure the distance between them)
#
# Keyboard shortcuts during clicking:
#   z   — undo last point
#   r   — reset all points
#   q   — quit without saving
#   Enter / s — accept the 4 points and proceed to enter dimensions
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
import json
import argparse
import textwrap
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config   # noqa: E402  (needs sys.path set first)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

CORNER_LABELS = [
    "1 — Top-Left  (far-left)",
    "2 — Top-Right (far-right)",
    "3 — Bottom-Right (near-right)",
    "4 — Bottom-Left  (near-left)",
]
COLORS = [
    (0,   200, 255),   # cyan
    (0,   255,   0),   # green
    (255, 128,   0),   # orange
    (200,   0, 255),   # purple
]
LINE_COLOR   = (0, 0, 255)
LINE_A_COLOR = (0, 180, 255)
LINE_B_COLOR = (0, 180, 255)


def draw_state(base_frame: np.ndarray, points: list, target_size=(1280, 720)) -> np.ndarray:
    """
    Render the calibration UI overlay on a display-sized copy of the frame.

    Returns the annotated frame — does NOT modify base_frame in-place so we
    can cheaply re-draw each time the user moves the mouse or adds a point.
    """
    # Scale to display size while keeping aspect ratio
    h, w = base_frame.shape[:2]
    scale = min(target_size[0] / w, target_size[1] / h)
    dw, dh = int(w * scale), int(h * scale)
    frame  = cv2.resize(base_frame, (dw, dh))

    n = len(points)

    # ── Draw existing points ──────────────────────────────────────────────────
    for i, (px, py) in enumerate(points):
        spx, spy = int(px * scale), int(py * scale)
        cv2.circle(frame, (spx, spy), 8, COLORS[i], -1)
        cv2.circle(frame, (spx, spy), 8, (255, 255, 255), 1)
        cv2.putText(frame, f"  {i+1}", (spx + 8, spy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[i], 2)

    # ── Draw ROI polygon when all 4 points are set ────────────────────────────
    if n == 4:
        scaled_pts = np.array(
            [[int(px * scale), int(py * scale)] for px, py in points],
            dtype=np.int32
        ).reshape(-1, 1, 2)

        # Semi-transparent fill
        overlay = frame.copy()
        cv2.fillPoly(overlay, [scaled_pts], (0, 0, 255))
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)

        cv2.polylines(frame, [scaled_pts], isClosed=True, color=LINE_COLOR, thickness=2)

        # Line A (far edge)
        a0 = (int(points[0][0] * scale), int(points[0][1] * scale))
        a1 = (int(points[1][0] * scale), int(points[1][1] * scale))
        cv2.line(frame, a0, a1, LINE_A_COLOR, 2)
        cv2.putText(frame, "Line A (far)",  (a0[0]+6, a0[1]-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, LINE_A_COLOR, 1)

        # Line B (near edge)
        b0 = (int(points[3][0] * scale), int(points[3][1] * scale))
        b1 = (int(points[2][0] * scale), int(points[2][1] * scale))
        cv2.line(frame, b0, b1, LINE_B_COLOR, 2)
        cv2.putText(frame, "Line B (near)", (b0[0]+6, b0[1]+16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, LINE_B_COLOR, 1)

    # ── Instruction overlay ───────────────────────────────────────────────────
    if n < 4:
        next_label = CORNER_LABELS[n]
        instruction = f"Click point {n+1}: {next_label}"
        cv2.rectangle(frame, (0, dh - 36), (dw, dh), (20, 20, 20), -1)
        cv2.putText(frame, instruction, (10, dh - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS[n], 2)
    else:
        cv2.rectangle(frame, (0, dh - 36), (dw, dh), (20, 20, 20), -1)
        cv2.putText(frame, "All 4 points set.  Press Enter/S to confirm, Z to undo, R to reset.",
                    (10, dh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 2)

    # ── Top status bar ────────────────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (dw, 28), (20, 20, 20), -1)
    status = f"ANPR Calibration  |  {n}/4 points  |  Z=undo  R=reset  Q=quit  Enter=confirm"
    cv2.putText(frame, status, (8, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    return frame


def ask_dimensions() -> tuple[float, float] | None:
    """
    Prompt the user for real-world ROI dimensions in the terminal.
    Returns (real_width_m, real_height_m) or None if the user cancels.
    """
    print()
    print("─" * 60)
    print("  Enter real-world dimensions of the marked road section.")
    print()
    print("  ROI_REAL_WIDTH_M   = horizontal distance (left→right) in metres.")
    print("    → e.g. one lane = 3.65 m, dual carriageway = 7.3 m")
    print()
    print("  ROI_REAL_HEIGHT_M  = depth (far→near, Line A→Line B) in metres.")
    print("    → Estimate from road markings, kerb stones, or known objects.")
    print("       A standard car ≈ 4.5 m long.  A road junction ≈ 10–25 m.")
    print("─" * 60)

    while True:
        try:
            w = float(input("  Width  (ROI_REAL_WIDTH_M,  metres) [e.g. 7.3]: ").strip())
            h = float(input("  Height (ROI_REAL_HEIGHT_M, metres) [e.g. 20.0]: ").strip())
            if w <= 0 or h <= 0:
                print("  Both values must be positive.")
                continue
            return w, h
        except ValueError:
            print("  Please enter a number (e.g. 7.3)")
        except (EOFError, KeyboardInterrupt):
            return None


def save_calibration(roi_points: list, real_width_m: float,
                     real_height_m: float, path: str):
    """Write calibration data to JSON and print confirmation."""
    data = {
        'roi_points':    roi_points,
        'real_width_m':  real_width_m,
        'real_height_m': real_height_m,
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

    print()
    print("─" * 60)
    print(f"  ✓ Saved to {path}")
    print()
    print("  Paste these into config.py if you want hardcoded defaults:")
    print()
    print(f"  ROI_POINTS = {roi_points}")
    print(f"  ROI_REAL_WIDTH_M  = {real_width_m}")
    print(f"  ROI_REAL_HEIGHT_M = {real_height_m}")
    print("─" * 60)
    print()
    print("  Run  python gui.py  or  python main.py  to start the pipeline.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Interactive 4-point road calibration for the ANPR system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Click the four corners of a known road rectangle in order:
              1  Top-Left     (far-left)
              2  Top-Right    (far-right)
              3  Bottom-Right (near-right)
              4  Bottom-Left  (near-left)
        """),
    )
    parser.add_argument('--video',  default=config.VIDEO_PATH,
                        help="Path to video file (default: config.VIDEO_PATH)")
    parser.add_argument('--frame',  type=int, default=1,
                        help="Which frame number to use for calibration (default: 1)")
    parser.add_argument('--output', default=config.CALIBRATION_FILE,
                        help="Where to write calibration.json")
    args = parser.parse_args()

    # ── Load the calibration frame ─────────────────────────────────────────────
    if not os.path.exists(args.video):
        print(f"ERROR: video not found at '{args.video}'")
        print("       Pass --video <path> or update VIDEO_PATH in config.py")
        sys.exit(1)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: cannot open video '{args.video}'")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame - 1)
    ret, raw_frame = cap.read()
    cap.release()

    if not ret:
        print(f"ERROR: could not read frame {args.frame} from video.")
        sys.exit(1)

    # Downscale 4K → 1080p (matching what the pipeline sees)
    fh, fw = raw_frame.shape[:2]
    if fw > 1920:
        raw_frame = cv2.resize(raw_frame, (1920, 1080))
        fh, fw = 1080, 1920

    print(f"\nFrame resolution (pipeline space): {fw}×{fh}")
    print(f"Calibration frame: #{args.frame}")
    print()
    print(textwrap.dedent("""\
        ┌──────────────────────────────────────────────────────┐
        │  ANPR — Road Calibration                             │
        │                                                      │
        │  Click 4 corners of a rectangular road section:      │
        │    1. Top-Left      (far  side, left)                │
        │    2. Top-Right     (far  side, right)               │
        │    3. Bottom-Right  (near side, right)               │
        │    4. Bottom-Left   (near side, left)                │
        │                                                      │
        │  Keys: Z = undo  |  R = reset  |  Q = quit          │
        │        Enter / S = confirm & continue                │
        └──────────────────────────────────────────────────────┘
    """))

    # ── OpenCV click loop ──────────────────────────────────────────────────────
    points    = []
    WINDOW    = "ANPR Calibration — click 4 road corners"
    confirmed = False

    # Compute display scale (for converting click coords back to 1080p)
    TARGET_W, TARGET_H = 1280, 720
    scale = min(TARGET_W / fw, TARGET_H / fh)

    def on_mouse(event, x, y, flags, param):
        nonlocal points
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            # Convert display-space click back to 1080p image space
            orig_x = int(round(x / scale))
            orig_y = int(round(y / scale))
            orig_x = max(0, min(fw - 1, orig_x))
            orig_y = max(0, min(fh - 1, orig_y))
            points.append([orig_x, orig_y])
            print(f"  Point {len(points)}: ({orig_x}, {orig_y})"
                  f"  ← {CORNER_LABELS[len(points)-1]}")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, TARGET_W, TARGET_H)
    cv2.setMouseCallback(WINDOW, on_mouse)

    while True:
        display = draw_state(raw_frame, points, (TARGET_W, TARGET_H))
        cv2.imshow(WINDOW, display)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord('q'), 27):          # Q or Esc
            print("\nCalibration cancelled.")
            cv2.destroyAllWindows()
            sys.exit(0)

        elif key == ord('z') and points:   # undo
            removed = points.pop()
            print(f"  Undo: removed point {removed}")

        elif key == ord('r'):              # reset
            points.clear()
            print("  Reset: all points cleared.")

        elif key in (13, ord('s')) and len(points) == 4:   # Enter or S
            confirmed = True
            break

    cv2.destroyAllWindows()

    if not confirmed or len(points) != 4:
        print("Calibration cancelled.")
        sys.exit(0)

    # ── Collect real-world dimensions ──────────────────────────────────────────
    dims = ask_dimensions()
    if dims is None:
        print("Calibration cancelled.")
        sys.exit(0)

    real_width_m, real_height_m = dims

    # ── Optional: show bird's-eye preview ─────────────────────────────────────
    try:
        from src.perspective_speed import PerspectiveSpeedCalculator
        calc = PerspectiveSpeedCalculator(
            roi_points    = points,
            real_width_m  = real_width_m,
            real_height_m = real_height_m,
            fps           = 30,
        )
        bev_frame = cv2.warpPerspective(
            raw_frame, calc.H,
            (int(real_width_m * calc._SCALE),
             int(real_height_m * calc._SCALE))
        )
        annotated = raw_frame.copy()
        calc.draw_roi(annotated)
        preview = np.hstack([
            cv2.resize(annotated, (640, 360)),
            cv2.resize(bev_frame, (640, 360)),
        ])
        cv2.imshow("Left: annotated frame   |   Right: bird's-eye view", preview)
        print("\n  Showing preview. Press any key to close and save.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"  (Preview skipped: {e})")

    # ── Save ───────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    save_calibration(points, real_width_m, real_height_m, args.output)


if __name__ == '__main__':
    main()