# gui.py
# ─────────────────────────────────────────────────────────────────────────────
# Modern PyQt6 GUI for the ANPR system.
# Run this instead of main.py when you want a visual interface.
#
# Architecture — why threading matters here:
#   PyQt6's golden rule: ONLY the main thread can touch the UI.
#   Your YOLO + OCR pipeline is slow and blocking — if it ran on the main
#   thread, the entire window would freeze every frame. The fix is to run
#   the whole CV pipeline on a QThread (worker), and communicate results
#   back to the UI using Qt signals. Signals are thread-safe by design.
#
#   Main thread:  Qt event loop → draws UI, handles button clicks
#   Worker thread: reads video → YOLO → OCR → tracker → emits signals
#   Signals carry: annotated frames, new detection records, stats updates
# ─────────────────────────────────────────────────────────────────────────────
import torch
import torchvision
import sys
import os
import time
import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sort'))

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QFileDialog, QFrame, QSizePolicy, QSplitter, QProgressBar,
    QGridLayout, QSlider, QMessageBox
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QSize
)
from PyQt6.QtGui import (
    QImage, QPixmap, QFont, QColor, QPalette, QIcon, QLinearGradient,
    QPainter, QBrush
)

import config
from src.plate_reader    import PlateReader
from src.vehicle_tracker import VehicleTracker
from src.utils           import save_records_to_csv


# ─────────────────────────────────────────────────────────────────────────────
# WORKER THREAD
# Runs the entire CV pipeline. Emits signals to update the UI.
# ─────────────────────────────────────────────────────────────────────────────

class PipelineWorker(QThread):
    """
    Runs video reading, YOLO detection, OCR, and tracking on a background thread.
    Communicates with the GUI exclusively through Qt signals.
    """

    # Signal carrying an annotated BGR frame as numpy array
    frame_ready     = pyqtSignal(np.ndarray)
    # Signal carrying a new completed detection record (dict)
    detection_ready = pyqtSignal(dict)
    # Signal carrying updated stats dict
    stats_updated   = pyqtSignal(dict)
    # Signal when processing finishes or is stopped
    finished        = pyqtSignal(str)
    # Signal for error messages
    error           = pyqtSignal(str)

    def __init__(self, video_path: str):
        super().__init__()
        self.video_path   = video_path
        self._running     = False
        self._paused      = False
        self.total_frames = 0
        self.processed    = 0

    def stop(self):
        self._running = False

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def run(self):
        """Main pipeline loop — runs entirely on the worker thread."""
        self._running = True

        try:
            from ultralytics import YOLO
            from sort import Sort

            vehicle_model = YOLO(config.VEHICLE_MODEL_PATH)
            plate_model   = YOLO(config.PLATE_MODEL_PATH) if os.path.exists(config.PLATE_MODEL_PATH) else None
            plate_reader  = PlateReader()
            sort_tracker  = Sort(
                max_age       = config.SORT_MAX_AGE,
                min_hits      = config.SORT_MIN_HITS,
                iou_threshold = config.SORT_IOU_THRESHOLD
            )

            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.error.emit(f"Cannot open video: {self.video_path}")
                return

            fps               = cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            vehicle_tracker = VehicleTracker(
                line1_y         = config.LINE_1_Y,
                line2_y         = config.LINE_2_Y,
                real_distance_m = config.REAL_DISTANCE_METERS,
                fps             = fps
            )

            frame_num        = 0
            total_detections = 0
            speed_readings   = []

        except Exception as e:
            self.error.emit(f"Initialisation error: {e}")
            return

        while self._running:
            # Handle pause
            while self._paused and self._running:
                time.sleep(0.05)

            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1

            # Downscale 4K to 1080p for processing
            h, w = frame.shape[:2]
            if w > 1920:
                frame = cv2.resize(frame, (1920, 1080))

            if frame_num % config.FRAME_SKIP != 0:
                continue

            self.processed += 1

            # ── Vehicle detection ──────────────────────────────────────────
            results = vehicle_model(
                frame, conf=config.VEHICLE_CONF, verbose=False,
                device=config.DEVICE
            )[0]

            dets = []
            for box in results.boxes:
                if int(box.cls[0]) in config.VEHICLE_CLASSES:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    dets.append([x1, y1, x2, y2, float(box.conf[0])])

            dets_np      = np.array(dets, dtype=np.float32) if dets else np.empty((0, 5))
            tracked_objs = sort_tracker.update(dets_np)

            # ── Per-vehicle plate detection and OCR ────────────────────────
            for obj in tracked_objs:
                x1, y1, x2, y2, track_id = map(int, obj)
                bbox     = (x1, y1, x2, y2)
                track_id = int(track_id)
                plate_text = None

                if plate_model is not None:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        pr = plate_model(crop, conf=config.PLATE_CONF, verbose=False,
                                         device=config.DEVICE)[0]
                        for pb in pr.boxes:
                            px1, py1, px2, py2 = map(int, pb.xyxy[0])
                            pc = crop[py1:py2, px1:px2]
                            if pc.size > 0:
                                plate_text = plate_reader.read(pc)
                                if plate_text:
                                    # Draw green plate box
                                    cv2.rectangle(frame,
                                        (x1+px1, y1+py1), (x1+px2, y1+py2),
                                        config.COLOR_PLATE, 2)
                            break

                vehicle_tracker.update(track_id, bbox, frame_num, plate_text)

                # Draw vehicle box
                best = vehicle_tracker.get_plate_for(track_id) or plate_text or ''
                label = f"ID:{track_id}"
                if best:
                    label += f"  {best}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), config.COLOR_VEHICLE, 2)
                cv2.putText(frame, label, (x1, y1-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, config.COLOR_VEHICLE, 2)

            # ── Reference lines ────────────────────────────────────────────
            fh, fw = frame.shape[:2]
            cv2.line(frame, (0, config.LINE_1_Y), (fw, config.LINE_1_Y), config.COLOR_LINE, 2)
            cv2.line(frame, (0, config.LINE_2_Y), (fw, config.LINE_2_Y), config.COLOR_LINE, 2)
            cv2.putText(frame, "Line 1", (10, config.LINE_1_Y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COLOR_LINE, 1)
            cv2.putText(frame, "Line 2", (10, config.LINE_2_Y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COLOR_LINE, 1)

            # ── Emit completed records ─────────────────────────────────────
            for record in vehicle_tracker.completed_records:
                total_detections += 1
                if record['speed_kmph'] > 0:
                    speed_readings.append(record['speed_kmph'])
                save_records_to_csv([record])
                self.detection_ready.emit(record)

            vehicle_tracker.completed_records.clear()

            # ── Emit frame and stats ───────────────────────────────────────
            self.frame_ready.emit(frame.copy())
            self.stats_updated.emit({
                'total':    total_detections,
                'avg_speed': round(sum(speed_readings)/len(speed_readings), 1) if speed_readings else 0,
                'max_speed': round(max(speed_readings), 1) if speed_readings else 0,
                'active':   len(vehicle_tracker.get_active_ids()),
                'progress': int(frame_num / max(self.total_frames, 1) * 100),
            })

        cap.release()
        self.finished.emit(config.OUTPUT_CSV)


# ─────────────────────────────────────────────────────────────────────────────
# STAT CARD WIDGET
# ─────────────────────────────────────────────────────────────────────────────

class StatCard(QFrame):
    """A styled card showing a single metric with a label and large value."""

    def __init__(self, label: str, value: str = "—", accent: str = "#00D4FF"):
        super().__init__()
        self.accent = accent
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
        self.label_w.setStyleSheet("color: #6B7694; font-size: 10px; font-weight: 700; letter-spacing: 1.5px; border: none;")

        self.value_w = QLabel(value)
        self.value_w.setStyleSheet(f"color: #E8EAF0; font-size: 24px; font-weight: 700; font-family: 'Consolas'; border: none;")

        layout.addWidget(self.label_w)
        layout.addWidget(self.value_w)

    def set_value(self, v: str):
        self.value_w.setText(v)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN WINDOW
# ─────────────────────────────────────────────────────────────────────────────

class ANPRWindow(QMainWindow):

    DARK_BG    = "#0D1117"
    PANEL_BG   = "#141820"
    BORDER     = "#2A2F3D"
    ACCENT     = "#00D4FF"
    ACCENT2    = "#FF6B35"
    TEXT_PRI   = "#E8EAF0"
    TEXT_SEC   = "#6B7694"
    SUCCESS    = "#00E676"
    WARNING    = "#FFD600"

    def __init__(self):
        super().__init__()
        self.worker     = None
        self.video_path = config.VIDEO_PATH
        self._setup_window()
        self._apply_global_style()
        self._build_ui()

    def _setup_window(self):
        self.setWindowTitle("ANPR System  •  Speed Detection")
        self.setMinimumSize(1400, 820)
        self.resize(1500, 900)

    def _apply_global_style(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background: {self.DARK_BG};
                color: {self.TEXT_PRI};
                font-family: 'Segoe UI', sans-serif;
                font-size: 13px;
            }}
            QSplitter::handle {{
                background: {self.BORDER};
                width: 1px;
            }}
            QScrollBar:vertical {{
                background: {self.DARK_BG};
                width: 6px;
                border-radius: 3px;
            }}
            QScrollBar::handle:vertical {{
                background: #3A4055;
                border-radius: 3px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_header())

        # Main content area
        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(16, 12, 16, 12)
        content_layout.setSpacing(12)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)

        # Left panel: video
        splitter.addWidget(self._build_video_panel())
        # Right panel: detections + stats
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([900, 500])

        content_layout.addWidget(splitter)
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

        # Logo / title
        title = QLabel("◈  ANPR  •  SPEED DETECTION")
        title.setStyleSheet(f"""
            color: {self.ACCENT};
            font-size: 14px;
            font-weight: 700;
            letter-spacing: 3px;
            font-family: 'Consolas', monospace;
        """)

        # Status pill
        self.status_label = QLabel("● IDLE")
        self.status_label.setStyleSheet(f"""
            color: {self.TEXT_SEC};
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 1px;
            padding: 4px 12px;
            background: #1E2330;
            border-radius: 10px;
        """)

        # Control buttons
        self.btn_open    = self._make_button("OPEN VIDEO",  self.ACCENT,  self._open_video)
        self.btn_start   = self._make_button("▶  START",    self.SUCCESS, self._start,  enabled=True)
        self.btn_pause   = self._make_button("⏸  PAUSE",   self.WARNING, self._pause,  enabled=False)
        self.btn_stop    = self._make_button("■  STOP",     self.ACCENT2, self._stop,   enabled=False)
        self.btn_export  = self._make_button("↓  EXPORT",  "#9C27B0",    self._export, enabled=True)

        layout.addWidget(title)
        layout.addStretch()
        layout.addWidget(self.status_label)
        layout.addSpacing(16)
        layout.addWidget(self.btn_open)
        layout.addWidget(self.btn_start)
        layout.addWidget(self.btn_pause)
        layout.addWidget(self.btn_stop)
        layout.addWidget(self.btn_export)

        return header

    def _make_button(self, text: str, color: str, slot, enabled: bool = True) -> QPushButton:
        btn = QPushButton(text)
        btn.setFixedHeight(34)
        btn.setMinimumWidth(110)
        btn.setEnabled(enabled)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {color};
                border: 1px solid {color};
                border-radius: 4px;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 1px;
                padding: 0 14px;
            }}
            QPushButton:hover {{
                background: {color}22;
            }}
            QPushButton:pressed {{
                background: {color}44;
            }}
            QPushButton:disabled {{
                color: #3A4055;
                border-color: #2A2F3D;
            }}
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

        # Panel title
        title = QLabel("LIVE FEED")
        title.setStyleSheet(f"color: {self.TEXT_SEC}; font-size: 10px; font-weight: 700; letter-spacing: 2px;")

        # Video display label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 400)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setStyleSheet(f"""
            background: #080B10;
            border: 1px solid {self.BORDER};
            border-radius: 4px;
            color: {self.TEXT_SEC};
            font-size: 13px;
        """)
        self.video_label.setText("No video loaded\nClick OPEN VIDEO to begin")

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background: #1E2330;
                border: none;
                border-radius: 2px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {self.ACCENT}, stop:1 {self.SUCCESS});
                border-radius: 2px;
            }}
        """)

        layout.addWidget(title)
        layout.addWidget(self.video_label, stretch=1)
        layout.addWidget(self.progress_bar)

        return panel

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

        self.card_total   = StatCard("Plates Detected",  "0",    self.ACCENT)
        self.card_active  = StatCard("Currently Tracked","0",    self.WARNING)
        self.card_avg_spd = StatCard("Avg Speed (km/h)", "—",    self.SUCCESS)
        self.card_max_spd = StatCard("Max Speed (km/h)", "—",    self.ACCENT2)

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
        title.setStyleSheet(f"color: {self.TEXT_SEC}; font-size: 10px; font-weight: 700; letter-spacing: 2px;")

        self.row_count_label = QLabel("0 records")
        self.row_count_label.setStyleSheet(f"color: {self.TEXT_SEC}; font-size: 11px;")

        header_row.addWidget(title)
        header_row.addStretch()
        header_row.addWidget(self.row_count_label)
        layout.addLayout(header_row)

        # Table
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["ID", "PLATE", "SPEED", "DIR", "TIME"])
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
                background: transparent;
                border: none;
                font-family: 'Consolas', monospace;
                font-size: 12px;
                gridline-color: transparent;
            }}
            QTableWidget::item {{
                padding: 6px 8px;
                color: {self.TEXT_PRI};
                border: none;
            }}
            QTableWidget::item:alternate {{
                background: #0D1117;
            }}
            QTableWidget::item:selected {{
                background: {self.ACCENT}33;
                color: {self.TEXT_PRI};
            }}
            QHeaderView::section {{
                background: #1A1F2E;
                color: {self.TEXT_SEC};
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 1.5px;
                padding: 6px 8px;
                border: none;
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
        self.footer_msg.setStyleSheet(f"color: {self.TEXT_SEC}; font-size: 11px; font-family: 'Consolas';")

        gpu_label = QLabel(f"Device: {config.DEVICE.upper()}")
        gpu_label.setStyleSheet(f"color: {self.SUCCESS if config.DEVICE == 'cuda' else self.TEXT_SEC}; font-size: 11px;")

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
            # Show first frame as preview
            cap = cv2.VideoCapture(path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                self._display_frame(frame)

    def _start(self):
        if not self.video_path or not os.path.exists(self.video_path):
            QMessageBox.warning(self, "No Video",
                "Please select a video file first using OPEN VIDEO.")
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
                "No detections have been saved yet.\nRun the pipeline first.")
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

    # ── Signal handlers (run on main thread) ──────────────────────────────────

    def _on_frame(self, frame: np.ndarray):
        """Convert BGR numpy frame → QPixmap and display it."""
        self._display_frame(frame)

    def _display_frame(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        lw = self.video_label.width()
        lh = self.video_label.height()

        # Scale to fit label while preserving aspect ratio
        scale = min(lw / w, lh / h)
        nw, nh = int(w * scale), int(h * scale)
        display = cv2.resize(frame, (nw, nh))

        # BGR → RGB → QImage → QPixmap
        rgb  = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        img  = QImage(rgb.data, nw, nh, nw * 3, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(img))

    def _on_detection(self, record: dict):
        """Add a new detection record to the table."""
        row = self.table.rowCount()
        self.table.insertRow(row)

        speed_str = f"{record['speed_kmph']:.1f} km/h"
        time_str  = record['timestamp'].split(' ')[-1]  # just HH:MM:SS

        values = [
            str(record['track_id']),
            record['plate'],
            speed_str,
            "↓" if "Bottom" in record['direction'] else "↑",
            time_str,
        ]

        for col, val in enumerate(values):
            item = QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            # Colour the plate cell based on whether it was detected
            if col == 1:
                if val == 'Not Detected':
                    item.setForeground(QColor(self.TEXT_SEC))
                else:
                    item.setForeground(QColor(self.ACCENT))

            # Colour speed based on threshold
            if col == 2:
                kmh = record['speed_kmph']
                if kmh > 80:
                    item.setForeground(QColor(self.ACCENT2))
                elif kmh > 50:
                    item.setForeground(QColor(self.WARNING))
                else:
                    item.setForeground(QColor(self.SUCCESS))

            self.table.setItem(row, col, item)

        self.table.scrollToBottom()
        self.row_count_label.setText(f"{row+1} records")

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
            color: {color};
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 1px;
            padding: 4px 12px;
            background: {color}18;
            border-radius: 10px;
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

    # Dark palette fallback (QSS handles most of it but Fusion needs this too)
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