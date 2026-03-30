# gui.py
# ─────────────────────────────────────────────────────────────────────────────
# PyQt6 GUI for the ANPR system — perspective-based speed detection.
#
# What's new in this version:
#   • CALIBRATE button opens CalibrationDialog — a Qt window showing the first
#     video frame where the user clicks 4 road corners and enters real-world
#     dimensions.  The result is saved to calibration.json; clicking START
#     afterwards picks it up automatically via config.py.
#   • The pipeline worker now builds a PerspectiveSpeedCalculator from the
#     ROI config and passes it to VehicleTracker — no more LINE_1_Y / LINE_2_Y.
#   • The live video overlay shows the ROI polygon + Line A / Line B instead
#     of two horizontal lines.
#
# Threading model (unchanged):
#   Main thread  → Qt event loop (draws UI, handles clicks)
#   Worker thread → video → YOLO → OCR → tracker → emits signals
# ─────────────────────────────────────────────────────────────────────────────

import torch          # must be imported before any PyQt6 import — WinError 1114 fix
import torchvision    # noqa: F401
import sys
import os
import json
import time
import importlib
import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sort'))

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QFileDialog, QFrame, QSizePolicy, QSplitter, QProgressBar,
    QGridLayout, QMessageBox, QDialog, QFormLayout, QDoubleSpinBox,
    QDialogButtonBox, QScrollArea,
)
from PyQt6.QtCore  import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui   import (
    QImage, QPixmap, QFont, QColor, QPalette, QCursor,
)

import config
from src.plate_reader      import PlateReader
from src.vehicle_tracker   import VehicleTracker
from src.perspective_speed import PerspectiveSpeedCalculator
from src.utils             import save_records_to_csv


# ─────────────────────────────────────────────────────────────────────────────
#  WORKER THREAD
# ─────────────────────────────────────────────────────────────────────────────

class PipelineWorker(QThread):
    """Runs the full CV pipeline on a background thread, emitting Qt signals."""

    frame_ready     = pyqtSignal(np.ndarray)
    detection_ready = pyqtSignal(dict)
    stats_updated   = pyqtSignal(dict)
    finished        = pyqtSignal(str)
    error           = pyqtSignal(str)

    def __init__(self, video_path: str):
        super().__init__()
        self.video_path   = video_path
        self._running     = False
        self._paused      = False
        self.total_frames = 0
        self.processed    = 0

    def stop(self):   self._running = False
    def pause(self):  self._paused  = True
    def resume(self): self._paused  = False

    def run(self):
        self._running = True
        try:
            from ultralytics import YOLO
            from sort import Sort

            # Reload config so any calibration saved since program start is picked up
            importlib.reload(config)

            vehicle_model = YOLO(config.VEHICLE_MODEL_PATH)
            plate_model   = YOLO(config.PLATE_MODEL_PATH) \
                            if os.path.exists(config.PLATE_MODEL_PATH) else None
            plate_reader  = PlateReader()
            sort_tracker  = Sort(
                max_age       = config.SORT_MAX_AGE,
                min_hits      = config.SORT_MIN_HITS,
                iou_threshold = config.SORT_IOU_THRESHOLD,
            )

            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.error.emit(f"Cannot open video: {self.video_path}")
                return

            fps               = cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Build perspective speed calculator from current config
            perspective_calc = PerspectiveSpeedCalculator(
                roi_points    = config.ROI_POINTS,
                real_width_m  = config.ROI_REAL_WIDTH_M,
                real_height_m = config.ROI_REAL_HEIGHT_M,
                fps           = fps,
                tolerance     = config.LINE_TOLERANCE,
            )
            vehicle_tracker = VehicleTracker(perspective_calc)

        except Exception as e:
            self.error.emit(f"Initialisation error: {e}")
            return

        frame_num        = 0
        total_detections = 0
        speed_readings   = []

        while self._running:
            while self._paused and self._running:
                time.sleep(0.05)

            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1

            # Downscale 4K → 1080p
            h, w = frame.shape[:2]
            if w > 1920:
                frame = cv2.resize(frame, (1920, 1080))

            if frame_num % config.FRAME_SKIP != 0:
                continue

            self.processed += 1

            # ── Vehicle detection ──────────────────────────────────────────
            results = vehicle_model(
                frame, conf=config.VEHICLE_CONF, verbose=False,
                device=config.DEVICE, half=True,
            )[0]

            dets = []
            for box in results.boxes:
                if int(box.cls[0]) in config.VEHICLE_CLASSES:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    dets.append([x1, y1, x2, y2, float(box.conf[0])])

            dets_np      = np.array(dets, dtype=np.float32) if dets else np.empty((0, 5))
            tracked_objs = sort_tracker.update(dets_np)

            # ── Per-vehicle plate detection & OCR ──────────────────────────
            for obj in tracked_objs:
                x1, y1, x2, y2, track_id = map(int, obj)
                bbox       = (x1, y1, x2, y2)
                track_id   = int(track_id)
                plate_text = None

                if plate_model is not None:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        pr = plate_model(
                            crop, conf=config.PLATE_CONF, verbose=False,
                            device=config.DEVICE, half=True,
                        )[0]
                        for pb in pr.boxes:
                            px1, py1, px2, py2 = map(int, pb.xyxy[0])
                            pc = crop[py1:py2, px1:px2]
                            if pc.size > 0:
                                plate_text = plate_reader.read(pc)
                                if plate_text:
                                    cv2.rectangle(
                                        frame,
                                        (x1 + px1, y1 + py1),
                                        (x1 + px2, y1 + py2),
                                        config.COLOR_PLATE, 2,
                                    )
                            break

                vehicle_tracker.update(track_id, bbox, frame_num, plate_text)

                best  = vehicle_tracker.get_plate_for(track_id) or plate_text or ''
                label = f"ID:{track_id}"
                if best:
                    label += f"  {best}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), config.COLOR_VEHICLE, 2)
                cv2.putText(frame, label, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            config.COLOR_VEHICLE, 2)

                # Draw ground-contact dot (what the tracker actually uses)
                gx, gy = (x1 + x2) // 2, y2
                cv2.circle(frame, (gx, gy), 4, (0, 255, 255), -1)

            # ── Draw ROI overlay ───────────────────────────────────────────
            perspective_calc.draw_roi(frame)

            # ── Emit completed records ─────────────────────────────────────
            for record in vehicle_tracker.completed_records:
                total_detections += 1
                if record['speed_kmph'] > 0:
                    speed_readings.append(record['speed_kmph'])
                save_records_to_csv([record])
                self.detection_ready.emit(record)

            vehicle_tracker.completed_records.clear()

            # ── Emit frame + stats ─────────────────────────────────────────
            self.frame_ready.emit(frame.copy())
            self.stats_updated.emit({
                'total':     total_detections,
                'avg_speed': round(sum(speed_readings) / len(speed_readings), 1)
                             if speed_readings else 0,
                'max_speed': round(max(speed_readings), 1) if speed_readings else 0,
                'active':    len(vehicle_tracker.get_active_ids()),
                'progress':  int(frame_num / max(self.total_frames, 1) * 100),
            })

        cap.release()
        self.finished.emit(config.OUTPUT_CSV)


# ─────────────────────────────────────────────────────────────────────────────
#  CALIBRATION DIALOG
# ─────────────────────────────────────────────────────────────────────────────

class ClickableImageLabel(QLabel):
    """
    QLabel subclass that emits (x, y) in original image coordinates whenever
    the user left-clicks.

    The label displays the image scaled-to-fit.  We track the scale factor and
    offset so click coordinates can be mapped back to original (1080p) space.
    """

    clicked = pyqtSignal(int, int)   # original image coordinates

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img_w = 1   # original image width
        self._img_h = 1   # original image height
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor))

    def set_frame(self, frame_bgr: np.ndarray):
        """Display a BGR numpy frame (preserves aspect ratio, fills label)."""
        self._img_h, self._img_w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        lw   = max(self.width(),  1)
        lh   = max(self.height(), 1)
        sc   = min(lw / w, lh / h)
        nw, nh = int(w * sc), int(h * sc)
        disp = cv2.resize(rgb, (nw, nh))
        img  = QImage(disp.data, nw, nh, nw * 3, QImage.Format.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(img))

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        px = self.pixmap()
        if px is None:
            return

        # The pixmap is centred inside the label — compute its top-left offset
        lw, lh   = self.width(), self.height()
        pw, ph   = px.width(), px.height()
        off_x    = (lw - pw) // 2
        off_y    = (lh - ph) // 2

        # Click position relative to the pixmap
        rel_x = event.position().x() - off_x
        rel_y = event.position().y() - off_y

        if rel_x < 0 or rel_y < 0 or rel_x >= pw or rel_y >= ph:
            return  # click outside image area

        # Scale back to original image coordinates
        orig_x = int(round(rel_x * self._img_w / pw))
        orig_y = int(round(rel_y * self._img_h / ph))
        orig_x = max(0, min(self._img_w - 1, orig_x))
        orig_y = max(0, min(self._img_h - 1, orig_y))
        self.clicked.emit(orig_x, orig_y)


class CalibrationDialog(QDialog):
    """
    Modal dialog for interactive 4-point road calibration.

    The user sees the first frame of the selected video.  They click four
    corners of a rectangular road section in order:
      1. Top-Left      (far-left)
      2. Top-Right     (far-right)
      3. Bottom-Right  (near-right)
      4. Bottom-Left   (near-left)

    Then they enter the real-world width and height of that rectangle.
    Clicking Save writes calibration.json; the pipeline picks it up on
    the next START.
    """

    CORNER_LABELS = [
        "1 — Top-Left (far-left)",
        "2 — Top-Right (far-right)",
        "3 — Bottom-Right (near-right)",
        "4 — Bottom-Left (near-left)",
    ]
    CORNER_COLORS_BGR = [
        (0,   200, 255),
        (0,   255,   0),
        (255, 128,   0),
        (200,   0, 255),
    ]
    CORNER_COLORS_QT = [
        "#00C8FF", "#00FF00", "#FF8000", "#C800FF",
    ]

    # ── Dark palette constants ─────────────────────────────────────────────────
    DARK_BG  = "#0D1117"
    PANEL_BG = "#141820"
    BORDER   = "#2A2F3D"
    ACCENT   = "#00D4FF"
    SUCCESS  = "#00E676"
    WARNING  = "#FFD600"
    TEXT_PRI = "#E8EAF0"
    TEXT_SEC = "#6B7694"

    def __init__(self, video_path: str, parent=None):
        super().__init__(parent)
        self.video_path  = video_path
        self.points      = []          # list of [x, y] in 1080p coords
        self._base_frame = None        # the raw 1080p frame for annotation

        self.setWindowTitle("ANPR — Road Calibration")
        self.setMinimumSize(1100, 700)
        self.resize(1200, 750)
        self.setModal(True)

        self._apply_style()
        self._build_ui()
        self._load_first_frame()

    def _apply_style(self):
        self.setStyleSheet(f"""
            QDialog, QWidget {{
                background: {self.DARK_BG};
                color: {self.TEXT_PRI};
                font-family: 'Segoe UI', sans-serif;
                font-size: 13px;
            }}
            QLabel {{ color: {self.TEXT_PRI}; }}
            QDoubleSpinBox {{
                background: #1E2330;
                color: {self.TEXT_PRI};
                border: 1px solid {self.BORDER};
                border-radius: 4px;
                padding: 4px 8px;
                font-family: 'Consolas';
                font-size: 14px;
            }}
            QPushButton {{
                background: transparent;
                border: 1px solid {self.BORDER};
                border-radius: 4px;
                padding: 6px 14px;
                color: {self.TEXT_PRI};
                font-weight: 600;
            }}
            QPushButton:hover {{ background: #1E2330; }}
        """)

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        # ── Left: image panel ─────────────────────────────────────────────────
        left = QVBoxLayout()
        left.setSpacing(8)

        self.title_label = QLabel("Click the 4 road corners in order")
        self.title_label.setStyleSheet(
            f"color: {self.ACCENT}; font-weight: 700; font-size: 14px; "
            "letter-spacing: 1px;"
        )

        self.img_label = ClickableImageLabel()
        self.img_label.setMinimumSize(700, 440)
        self.img_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.img_label.setStyleSheet(
            f"background: #080B10; border: 1px solid {self.BORDER}; border-radius: 4px;"
        )
        self.img_label.clicked.connect(self._on_image_click)

        left.addWidget(self.title_label)
        left.addWidget(self.img_label, stretch=1)
        root.addLayout(left, stretch=3)

        # ── Right: controls panel ─────────────────────────────────────────────
        right = QVBoxLayout()
        right.setSpacing(16)

        # Instructions
        instr = QLabel(
            "<b>How to calibrate:</b><br><br>"
            "Mark 4 corners of a <b>rectangular</b> road section "
            "whose real-world size you know.<br><br>"
            "<b>Ideal targets:</b><br>"
            "• Two parallel lane markings<br>"
            "• Road studs / cat's eyes<br>"
            "• Stop-line to crossing bar<br>"
            "• Two consecutive junction lines<br><br>"
            "<b>Click order:</b><br>"
        )
        instr.setWordWrap(True)
        instr.setStyleSheet(f"color: {self.TEXT_SEC}; font-size: 12px;")

        # Corner status indicators
        self.corner_labels = []
        corner_frame = QFrame()
        corner_frame.setStyleSheet(
            f"background: {self.PANEL_BG}; border: 1px solid {self.BORDER}; border-radius: 6px;"
        )
        cf_layout = QVBoxLayout(corner_frame)
        cf_layout.setSpacing(6)
        cf_layout.setContentsMargins(10, 10, 10, 10)

        for i, text in enumerate(self.CORNER_LABELS):
            lbl = QLabel(f"○  {text}")
            lbl.setStyleSheet(
                f"color: {self.TEXT_SEC}; font-size: 12px; "
                f"font-family: 'Consolas'; border: none; background: transparent;"
            )
            self.corner_labels.append(lbl)
            cf_layout.addWidget(lbl)

        # Undo / reset
        btn_row = QHBoxLayout()
        self.btn_undo  = QPushButton("↩  Undo")
        self.btn_reset = QPushButton("✕  Reset")
        self.btn_undo.setStyleSheet(
            f"color: {self.WARNING}; border-color: {self.WARNING};"
        )
        self.btn_reset.setStyleSheet(
            f"color: #FF6B35; border-color: #FF6B35;"
        )
        self.btn_undo.clicked.connect(self._undo)
        self.btn_reset.clicked.connect(self._reset)
        btn_row.addWidget(self.btn_undo)
        btn_row.addWidget(self.btn_reset)

        # Real-world dimensions
        dim_frame = QFrame()
        dim_frame.setStyleSheet(
            f"background: {self.PANEL_BG}; border: 1px solid {self.BORDER}; border-radius: 6px;"
        )
        df_layout = QFormLayout(dim_frame)
        df_layout.setContentsMargins(12, 12, 12, 12)
        df_layout.setSpacing(10)

        dim_header = QLabel("REAL-WORLD DIMENSIONS")
        dim_header.setStyleSheet(
            f"color: {self.TEXT_SEC}; font-size: 10px; font-weight: 700; "
            "letter-spacing: 1.5px; border: none; background: transparent;"
        )
        df_layout.addRow(dim_header)

        self.spin_width = QDoubleSpinBox()
        self.spin_width.setRange(0.5, 500.0)
        self.spin_width.setValue(config.ROI_REAL_WIDTH_M)
        self.spin_width.setSuffix(" m")
        self.spin_width.setDecimals(1)
        self.spin_width.setSingleStep(0.5)

        self.spin_height = QDoubleSpinBox()
        self.spin_height.setRange(0.5, 500.0)
        self.spin_height.setValue(config.ROI_REAL_HEIGHT_M)
        self.spin_height.setSuffix(" m")
        self.spin_height.setDecimals(1)
        self.spin_height.setSingleStep(0.5)

        width_hint  = QLabel("Width (left→right)")
        height_hint = QLabel("Depth (far→near, Line A→B)")
        for lbl in (width_hint, height_hint):
            lbl.setStyleSheet(
                f"color: {self.TEXT_PRI}; border: none; background: transparent;"
            )

        df_layout.addRow(width_hint,  self.spin_width)
        df_layout.addRow(height_hint, self.spin_height)

        hint = QLabel(
            "Standard lane ≈ 3.65 m wide\n"
            "Dual carriageway ≈ 7.3 m wide\n"
            "Car length ≈ 4.5 m"
        )
        hint.setStyleSheet(
            f"color: {self.TEXT_SEC}; font-size: 11px; border: none; background: transparent;"
        )
        df_layout.addRow(hint)

        # Dialog buttons
        self.btn_save = QPushButton("💾  Save Calibration")
        self.btn_save.setEnabled(False)
        self.btn_save.setFixedHeight(38)
        self.btn_save.setStyleSheet(f"""
            QPushButton {{
                color: {self.SUCCESS};
                border: 1px solid {self.SUCCESS};
                border-radius: 4px;
                font-weight: 700;
                font-size: 12px;
                letter-spacing: 1px;
            }}
            QPushButton:hover  {{ background: {self.SUCCESS}22; }}
            QPushButton:disabled {{ color: #3A4055; border-color: #2A2F3D; }}
        """)
        self.btn_save.clicked.connect(self._save)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.setFixedHeight(38)
        btn_cancel.clicked.connect(self.reject)

        right.addWidget(instr)
        right.addWidget(corner_frame)
        right.addLayout(btn_row)
        right.addWidget(dim_frame)
        right.addStretch()
        right.addWidget(self.btn_save)
        right.addWidget(btn_cancel)

        root.addLayout(right, stretch=1)

    # ── Frame loading ─────────────────────────────────────────────────────────

    def _load_first_frame(self):
        if not self.video_path or not os.path.exists(self.video_path):
            self.img_label.setText("No video loaded.")
            return

        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            self.img_label.setText("Could not read video frame.")
            return

        # Downscale 4K → 1080p (same as pipeline)
        h, w = frame.shape[:2]
        if w > 1920:
            frame = cv2.resize(frame, (1920, 1080))

        self._base_frame = frame.copy()
        self._refresh_image()

    # ── Click handling ────────────────────────────────────────────────────────

    def _on_image_click(self, x: int, y: int):
        if len(self.points) >= 4:
            return

        self.points.append([x, y])
        self._refresh_corner_labels()
        self._refresh_image()

        if len(self.points) == 4:
            self.title_label.setText(
                "All 4 points set — enter dimensions and click Save"
            )
            self.btn_save.setEnabled(True)

    def _undo(self):
        if self.points:
            self.points.pop()
            self._refresh_corner_labels()
            self._refresh_image()
            self.btn_save.setEnabled(len(self.points) == 4)
            n = len(self.points)
            if n < 4:
                self.title_label.setText("Click the 4 road corners in order")

    def _reset(self):
        self.points.clear()
        self._refresh_corner_labels()
        self._refresh_image()
        self.btn_save.setEnabled(False)
        self.title_label.setText("Click the 4 road corners in order")

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _refresh_corner_labels(self):
        n = len(self.points)
        for i, lbl in enumerate(self.corner_labels):
            if i < n:
                pt  = self.points[i]
                col = self.CORNER_COLORS_QT[i]
                lbl.setText(f"● {self.CORNER_LABELS[i]}  ({pt[0]}, {pt[1]})")
                lbl.setStyleSheet(
                    f"color: {col}; font-size: 12px; "
                    "font-family: 'Consolas'; border: none; background: transparent;"
                )
            elif i == n:
                lbl.setText(f"◎  {self.CORNER_LABELS[i]}  ← click next")
                lbl.setStyleSheet(
                    f"color: {self.WARNING}; font-size: 12px; "
                    "font-family: 'Consolas'; border: none; background: transparent;"
                )
            else:
                lbl.setText(f"○  {self.CORNER_LABELS[i]}")
                lbl.setStyleSheet(
                    f"color: {self.TEXT_SEC}; font-size: 12px; "
                    "font-family: 'Consolas'; border: none; background: transparent;"
                )

    def _refresh_image(self):
        if self._base_frame is None:
            return

        frame = self._base_frame.copy()
        n     = len(self.points)

        # Draw existing points
        for i, (px, py) in enumerate(self.points):
            col = self.CORNER_COLORS_BGR[i]
            cv2.circle(frame, (px, py), 9, col, -1)
            cv2.circle(frame, (px, py), 9, (255, 255, 255), 1)
            cv2.putText(frame, str(i + 1), (px + 12, py + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

        # Draw ROI polygon and lines when all 4 are placed
        if n == 4:
            pts = np.array(self.points, dtype=np.int32).reshape(-1, 1, 2)

            # Semi-transparent fill
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], (0, 0, 200))
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

            cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

            # Line A (far edge — points 0→1)
            cv2.line(frame,
                     tuple(self.points[0]), tuple(self.points[1]),
                     (0, 180, 255), 2)
            cv2.putText(frame, "Line A (far)",
                        (self.points[0][0] + 6, self.points[0][1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1)

            # Line B (near edge — points 3→2)
            cv2.line(frame,
                     tuple(self.points[3]), tuple(self.points[2]),
                     (0, 180, 255), 2)
            cv2.putText(frame, "Line B (near)",
                        (self.points[3][0] + 6, self.points[3][1] + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1)

            # Dimension annotations
            mx = (self.points[0][0] + self.points[1][0]) // 2
            my = (self.points[0][1] + self.points[1][1]) // 2
            cv2.putText(frame,
                        f"{self.spin_width.value():.1f}m wide",
                        (mx - 60, my - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            mx2 = (self.points[0][0] + self.points[3][0]) // 2
            my2 = (self.points[0][1] + self.points[3][1]) // 2
            cv2.putText(frame,
                        f"{self.spin_height.value():.1f}m deep",
                        (mx2 - 60, my2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Show "click here next" crosshair hint when < 4 points
        if 0 < n < 4:
            last = self.points[-1]
            next_col = self.CORNER_COLORS_BGR[n]
            cv2.putText(frame,
                        f"Next: click point {n + 1}",
                        (last[0] + 14, last[1] + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, next_col, 1)

        self.img_label.set_frame(frame)

    # ── Saving ────────────────────────────────────────────────────────────────

    def _save(self):
        if len(self.points) != 4:
            QMessageBox.warning(self, "Incomplete",
                                "Please click all 4 road corners first.")
            return

        cal = {
            'roi_points':    self.points,
            'real_width_m':  self.spin_width.value(),
            'real_height_m': self.spin_height.value(),
        }
        try:
            with open(config.CALIBRATION_FILE, 'w') as f:
                json.dump(cal, f, indent=2)
        except Exception as e:
            QMessageBox.critical(self, "Save Error",
                                 f"Could not write calibration.json:\n{e}")
            return

        QMessageBox.information(
            self, "Calibration Saved",
            f"Calibration saved to:\n{config.CALIBRATION_FILE}\n\n"
            f"ROI points:  {self.points}\n"
            f"Width:       {cal['real_width_m']} m\n"
            f"Depth:       {cal['real_height_m']} m\n\n"
            "Click START to run the pipeline with the new calibration."
        )
        self.accept()

    def resizeEvent(self, event):
        """Re-render the image when the dialog is resized."""
        super().resizeEvent(event)
        self._refresh_image()


# ─────────────────────────────────────────────────────────────────────────────
#  STAT CARD
# ─────────────────────────────────────────────────────────────────────────────

class StatCard(QFrame):
    def __init__(self, label: str, value: str = "—", accent: str = "#00D4FF"):
        super().__init__()
        self.setFixedHeight(90)
        self.setStyleSheet(f"""
            QFrame {{
                background: #141820;
                border: 1px solid #2A2F3D;
                border-top: 3px solid {accent};
                border-radius: 8px;
            }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(4)

        self.label_w = QLabel(label.upper())
        self.label_w.setStyleSheet(
            "color: #6B7694; font-size: 10px; font-weight: 700; "
            "letter-spacing: 1.5px; border: none;"
        )
        self.value_w = QLabel(value)
        self.value_w.setStyleSheet(
            f"color: #E8EAF0; font-size: 24px; font-weight: 700; "
            "font-family: 'Consolas'; border: none;"
        )
        layout.addWidget(self.label_w)
        layout.addWidget(self.value_w)

    def set_value(self, v: str):
        self.value_w.setText(v)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN WINDOW
# ─────────────────────────────────────────────────────────────────────────────

class ANPRWindow(QMainWindow):

    DARK_BG  = "#0D1117"
    PANEL_BG = "#141820"
    BORDER   = "#2A2F3D"
    ACCENT   = "#00D4FF"
    ACCENT2  = "#FF6B35"
    TEXT_PRI = "#E8EAF0"
    TEXT_SEC = "#6B7694"
    SUCCESS  = "#00E676"
    WARNING  = "#FFD600"

    def __init__(self):
        super().__init__()
        self.worker     = None
        self.video_path = config.VIDEO_PATH
        self._setup_window()
        self._apply_global_style()
        self._build_ui()

    def _setup_window(self):
        self.setWindowTitle("ANPR System  •  Speed Detection  •  Perspective Calibration")
        self.setMinimumSize(1400, 820)
        self.resize(1550, 920)

    def _apply_global_style(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background: {self.DARK_BG};
                color: {self.TEXT_PRI};
                font-family: 'Segoe UI', sans-serif;
                font-size: 13px;
            }}
            QSplitter::handle {{ background: {self.BORDER}; width: 1px; }}
            QScrollBar:vertical {{
                background: {self.DARK_BG}; width: 6px; border-radius: 3px;
            }}
            QScrollBar::handle:vertical {{ background: #3A4055; border-radius: 3px; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
        """)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_header())

        content = QWidget()
        cl = QHBoxLayout(content)
        cl.setContentsMargins(16, 12, 16, 12)
        cl.setSpacing(12)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.addWidget(self._build_video_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([960, 520])

        cl.addWidget(splitter)
        root.addWidget(content, stretch=1)
        root.addWidget(self._build_footer())

    # ── Header ────────────────────────────────────────────────────────────────

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setFixedHeight(56)
        header.setStyleSheet(f"""
            QFrame {{
                background: {self.PANEL_BG};
                border-bottom: 1px solid {self.BORDER};
            }}
        """)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 0, 20, 0)

        title = QLabel("◈  ANPR  •  SPEED DETECTION")
        title.setStyleSheet(f"""
            color: {self.ACCENT}; font-size: 14px; font-weight: 700;
            letter-spacing: 3px; font-family: 'Consolas', monospace;
        """)

        self.status_label = QLabel("● IDLE")
        self.status_label.setStyleSheet(f"""
            color: {self.TEXT_SEC}; font-size: 11px; font-weight: 600;
            letter-spacing: 1px; padding: 4px 12px;
            background: #1E2330; border-radius: 10px;
        """)

        self.btn_open      = self._make_button("OPEN VIDEO",   self.ACCENT,  self._open_video)
        self.btn_calibrate = self._make_button("⬡  CALIBRATE", "#9C27B0",    self._calibrate)
        self.btn_start     = self._make_button("▶  START",     self.SUCCESS, self._start,  enabled=True)
        self.btn_pause     = self._make_button("⏸  PAUSE",    self.WARNING, self._pause,  enabled=False)
        self.btn_stop      = self._make_button("■  STOP",      self.ACCENT2, self._stop,   enabled=False)
        self.btn_export    = self._make_button("↓  EXPORT",   "#00BCD4",    self._export, enabled=True)

        layout.addWidget(title)
        layout.addStretch()
        layout.addWidget(self.status_label)
        layout.addSpacing(16)
        for btn in (self.btn_open, self.btn_calibrate,
                    self.btn_start, self.btn_pause,
                    self.btn_stop, self.btn_export):
            layout.addWidget(btn)

        return header

    def _make_button(self, text, color, slot, enabled=True) -> QPushButton:
        btn = QPushButton(text)
        btn.setFixedHeight(34)
        btn.setMinimumWidth(110)
        btn.setEnabled(enabled)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {color};
                border: 1px solid {color}; border-radius: 4px;
                font-size: 11px; font-weight: 700;
                letter-spacing: 1px; padding: 0 14px;
            }}
            QPushButton:hover   {{ background: {color}22; }}
            QPushButton:pressed {{ background: {color}44; }}
            QPushButton:disabled {{ color: #3A4055; border-color: #2A2F3D; }}
        """)
        btn.clicked.connect(slot)
        return btn

    # ── Video panel ───────────────────────────────────────────────────────────

    def _build_video_panel(self) -> QWidget:
        panel = QFrame()
        panel.setStyleSheet(f"""
            QFrame {{
                background: {self.PANEL_BG};
                border: 1px solid {self.BORDER};
                border-radius: 8px;
            }}
        """)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Sub-header with calibration status
        header_row = QHBoxLayout()
        title = QLabel("LIVE FEED")
        title.setStyleSheet(
            f"color: {self.TEXT_SEC}; font-size: 10px; font-weight: 700; letter-spacing: 2px;"
        )
        self.cal_status = QLabel()
        self._refresh_cal_status()

        header_row.addWidget(title)
        header_row.addStretch()
        header_row.addWidget(self.cal_status)
        layout.addLayout(header_row)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 400)
        self.video_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.video_label.setStyleSheet(f"""
            background: #080B10; border: 1px solid {self.BORDER};
            border-radius: 4px; color: {self.TEXT_SEC}; font-size: 13px;
        """)
        self.video_label.setText(
            "No video loaded\nClick OPEN VIDEO to begin, then CALIBRATE to mark the road"
        )

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background: #1E2330; border: none; border-radius: 2px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {self.ACCENT}, stop:1 {self.SUCCESS});
                border-radius: 2px;
            }}
        """)

        layout.addWidget(self.video_label, stretch=1)
        layout.addWidget(self.progress_bar)
        return panel

    def _refresh_cal_status(self):
        """Update the calibration status badge in the video panel header."""
        if os.path.exists(config.CALIBRATION_FILE):
            self.cal_status.setText("⬡ CALIBRATED")
            self.cal_status.setStyleSheet(
                "color: #00E676; font-size: 11px; font-weight: 600; "
                "letter-spacing: 1px; padding: 3px 10px; "
                "background: #00E67618; border-radius: 8px;"
            )
        else:
            self.cal_status.setText("⬡ NOT CALIBRATED")
            self.cal_status.setStyleSheet(
                "color: #FFD600; font-size: 11px; font-weight: 600; "
                "letter-spacing: 1px; padding: 3px 10px; "
                "background: #FFD60018; border-radius: 8px;"
            )

    # ── Right panel ───────────────────────────────────────────────────────────

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        layout.addWidget(self._build_stats_row())
        layout.addWidget(self._build_detections_table(), stretch=1)
        return panel

    def _build_stats_row(self) -> QWidget:
        row = QWidget()
        layout = QGridLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.card_total   = StatCard("Plates Detected",   "0",  self.ACCENT)
        self.card_active  = StatCard("Currently Tracked", "0",  self.WARNING)
        self.card_avg_spd = StatCard("Avg Speed (km/h)",  "—",  self.SUCCESS)
        self.card_max_spd = StatCard("Max Speed (km/h)",  "—",  self.ACCENT2)

        layout.addWidget(self.card_total,   0, 0)
        layout.addWidget(self.card_active,  0, 1)
        layout.addWidget(self.card_avg_spd, 1, 0)
        layout.addWidget(self.card_max_spd, 1, 1)
        return row

    def _build_detections_table(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet(f"""
            QFrame {{
                background: {self.PANEL_BG};
                border: 1px solid {self.BORDER};
                border-radius: 8px;
            }}
        """)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        header_row = QHBoxLayout()
        title = QLabel("DETECTIONS")
        title.setStyleSheet(
            f"color: {self.TEXT_SEC}; font-size: 10px; font-weight: 700; letter-spacing: 2px;"
        )
        self.row_count_label = QLabel("0 records")
        self.row_count_label.setStyleSheet(f"color: {self.TEXT_SEC}; font-size: 11px;")
        header_row.addWidget(title)
        header_row.addStretch()
        header_row.addWidget(self.row_count_label)
        layout.addLayout(header_row)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["ID", "PLATE", "SPEED", "DIST", "DIR", "TIME"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(0, 48)
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.table.setStyleSheet(f"""
            QTableWidget {{
                background: transparent; border: none;
                font-family: 'Consolas', monospace; font-size: 12px;
                gridline-color: transparent;
            }}
            QTableWidget::item {{
                padding: 6px 8px; color: {self.TEXT_PRI}; border: none;
            }}
            QTableWidget::item:alternate {{ background: #0D1117; }}
            QTableWidget::item:selected {{
                background: {self.ACCENT}33; color: {self.TEXT_PRI};
            }}
            QHeaderView::section {{
                background: #1A1F2E; color: {self.TEXT_SEC};
                font-size: 10px; font-weight: 700; letter-spacing: 1.5px;
                padding: 6px 8px; border: none;
                border-bottom: 1px solid {self.BORDER};
            }}
        """)
        layout.addWidget(self.table)
        return frame

    # ── Footer ────────────────────────────────────────────────────────────────

    def _build_footer(self) -> QWidget:
        footer = QFrame()
        footer.setFixedHeight(32)
        footer.setStyleSheet(f"""
            QFrame {{
                background: {self.PANEL_BG};
                border-top: 1px solid {self.BORDER};
            }}
        """)
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(20, 0, 20, 0)

        self.footer_msg = QLabel(f"Output: {config.OUTPUT_CSV}")
        self.footer_msg.setStyleSheet(
            f"color: {self.TEXT_SEC}; font-size: 11px; font-family: 'Consolas';"
        )
        gpu_label = QLabel(f"Device: {config.DEVICE.upper()}")
        gpu_label.setStyleSheet(
            f"color: {self.SUCCESS if config.DEVICE == 'cuda' else self.TEXT_SEC}; font-size: 11px;"
        )
        layout.addWidget(self.footer_msg)
        layout.addStretch()
        layout.addWidget(gpu_label)
        return footer

    # ── Control slots ─────────────────────────────────────────────────────────

    def _open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*)"
        )
        if path:
            self.video_path = path
            self.footer_msg.setText(f"Video: {os.path.basename(path)}")
            cap = cv2.VideoCapture(path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                if frame.shape[1] > 1920:
                    frame = cv2.resize(frame, (1920, 1080))
                self._display_frame(frame)

    def _calibrate(self):
        if not self.video_path or not os.path.exists(self.video_path):
            QMessageBox.warning(self, "No Video",
                "Please open a video first (OPEN VIDEO), then calibrate.")
            return

        dlg = CalibrationDialog(self.video_path, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            # Reload config so the new calibration.json values are active
            importlib.reload(config)
            self._refresh_cal_status()
            self.footer_msg.setText(
                f"Calibration saved — ROI: {config.ROI_POINTS[0]}…  "
                f"{config.ROI_REAL_WIDTH_M}m × {config.ROI_REAL_HEIGHT_M}m"
            )

    def _start(self):
        if not self.video_path or not os.path.exists(self.video_path):
            QMessageBox.warning(self, "No Video",
                "Please select a video file first using OPEN VIDEO.")
            return

        if not os.path.exists(config.CALIBRATION_FILE):
            reply = QMessageBox.question(
                self, "No Calibration",
                "No calibration.json found — the default ROI will be used "
                "and speeds may be inaccurate.\n\n"
                "Run CALIBRATE first for accurate readings.\n\n"
                "Continue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                return

        self.table.setRowCount(0)
        self.row_count_label.setText("0 records")
        self.progress_bar.setValue(0)

        self.worker = PipelineWorker(self.video_path)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.detection_ready.connect(self._on_detection)
        self.worker.stats_updated.connect(self._on_stats)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

        self._set_status("PROCESSING", self.SUCCESS)
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)

    def _pause(self):
        if self.worker:
            if self.worker._paused:
                self.worker.resume()
                self.btn_pause.setText("⏸  PAUSE")
                self._set_status("PROCESSING", self.SUCCESS)
            else:
                self.worker.pause()
                self.btn_pause.setText("▶  RESUME")
                self._set_status("PAUSED", self.WARNING)

    def _stop(self):
        if self.worker:
            self.worker.stop()
        self._set_controls_idle()

    def _export(self):
        if not os.path.exists(config.OUTPUT_CSV):
            QMessageBox.information(self, "No Data",
                "No detections saved yet. Run the pipeline first.")
            return
        dest, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "detections_export.csv",
            "CSV Files (*.csv)"
        )
        if dest:
            import shutil
            shutil.copy(config.OUTPUT_CSV, dest)
            QMessageBox.information(self, "Exported",
                f"Detections saved to:\n{dest}")

    # ── Signal handlers (main thread) ─────────────────────────────────────────

    def _on_frame(self, frame: np.ndarray):
        self._display_frame(frame)

    def _display_frame(self, frame: np.ndarray):
        h, w   = frame.shape[:2]
        lw     = self.video_label.width()
        lh     = self.video_label.height()
        scale  = min(lw / w, lh / h)
        nw, nh = int(w * scale), int(h * scale)
        disp   = cv2.resize(frame, (nw, nh))
        rgb    = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        img    = QImage(rgb.data, nw, nh, nw * 3, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(img))

    def _on_detection(self, record: dict):
        row = self.table.rowCount()
        self.table.insertRow(row)

        speed_str = f"{record['speed_kmph']:.1f} km/h"
        dist_str  = f"{record.get('dist_m', 0):.1f}m"
        time_str  = record['timestamp'].split(' ')[-1]

        values = [
            str(record['track_id']),
            record['plate'],
            speed_str,
            dist_str,
            "↓" if "Bottom" in record['direction'] else "↑",
            time_str,
        ]

        for col, val in enumerate(values):
            item = QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            if col == 1:   # plate
                color = self.TEXT_SEC if val == 'Not Detected' else self.ACCENT
                item.setForeground(QColor(color))

            if col == 2:   # speed
                kmh = record['speed_kmph']
                color = (self.ACCENT2 if kmh > 80 else
                         self.WARNING if kmh > 50 else self.SUCCESS)
                item.setForeground(QColor(color))

            self.table.setItem(row, col, item)

        self.table.scrollToBottom()
        self.row_count_label.setText(f"{row + 1} records")

    def _on_stats(self, stats: dict):
        self.card_total.set_value(str(stats['total']))
        self.card_active.set_value(str(stats['active']))
        self.card_avg_spd.set_value(str(stats['avg_speed']) if stats['avg_speed'] else "—")
        self.card_max_spd.set_value(str(stats['max_speed']) if stats['max_speed'] else "—")
        self.progress_bar.setValue(stats['progress'])

    def _on_finished(self, csv_path: str):
        self._set_status("COMPLETE", self.ACCENT)
        self._set_controls_idle()
        self.footer_msg.setText(f"Saved → {csv_path}")
        self.progress_bar.setValue(100)

    def _on_error(self, msg: str):
        self._set_status("ERROR", self.ACCENT2)
        self._set_controls_idle()
        QMessageBox.critical(self, "Pipeline Error", msg)

    def _set_status(self, text: str, color: str):
        self.status_label.setText(f"● {text}")
        self.status_label.setStyleSheet(f"""
            color: {color}; font-size: 11px; font-weight: 600;
            letter-spacing: 1px; padding: 4px 12px;
            background: {color}18; border-radius: 10px;
            border: 1px solid {color}44;
        """)

    def _set_controls_idle(self):
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("⏸  PAUSE")
        self.btn_stop.setEnabled(False)
        self._set_status("IDLE", self.TEXT_SEC)

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)
        event.accept()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window,        QColor("#0D1117"))
    palette.setColor(QPalette.ColorRole.WindowText,    QColor("#E8EAF0"))
    palette.setColor(QPalette.ColorRole.Base,          QColor("#141820"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#0D1117"))
    palette.setColor(QPalette.ColorRole.Text,          QColor("#E8EAF0"))
    palette.setColor(QPalette.ColorRole.Button,        QColor("#141820"))
    palette.setColor(QPalette.ColorRole.ButtonText,    QColor("#E8EAF0"))
    palette.setColor(QPalette.ColorRole.Highlight,     QColor("#00D4FF"))
    app.setPalette(palette)

    window = ANPRWindow()
    window.show()
    sys.exit(app.exec())