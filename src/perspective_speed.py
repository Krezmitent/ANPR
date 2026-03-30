# src/perspective_speed.py
# ─────────────────────────────────────────────────────────────────────────────
# Responsibility: All perspective-transform maths.
#
# Why perspective matters:
#   A standard camera looking at a road at an angle suffers from foreshortening
#   — objects near the camera appear much larger than identical objects far away.
#   If you naively measure distances in image pixels, a vehicle travelling 10m
#   near the camera covers far more pixels than the same 10m far away.
#
#   The fix is a perspective transform (homography). The user marks four points
#   on the road that form a known real-world rectangle (e.g. two lane markings
#   20m apart). OpenCV uses those four point-pairs to compute a 3×3 matrix H
#   that warps the image into a "bird's-eye view" where pixels represent metres
#   uniformly across the entire frame.
#
#   Speed is then measured by:
#     1. Projecting each vehicle's ground contact point through H → world (metres)
#     2. Timing two crossing events (near edge and far edge of the ROI)
#     3. speed = world_distance / elapsed_time
#
# ROI point order (IMPORTANT — must match what the calibration tool saves):
#   [0] = top-left  (far-left  corner of road section)
#   [1] = top-right (far-right corner of road section)
#   [2] = bottom-right (near-right corner)
#   [3] = bottom-left  (near-left  corner)
#
# "Top" / "far" = the edge closer to the horizon (higher Y if looking down a road).
# "Bottom" / "near" = the edge closest to the camera.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import cv2
import numpy as np


class PerspectiveSpeedCalculator:
    """
    Converts image-space pixel coordinates to real-world metres using a
    homography derived from four user-marked road points, then computes
    vehicle speed from two timed line-crossing events.
    """

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(
        self,
        roi_points: list,          # 4 × [x, y] in 1080p image pixels
        real_width_m: float,       # road width covered by the ROI (metres)
        real_height_m: float,      # road depth  covered by the ROI (metres)
        fps: float,
        tolerance: int = 30,       # px tolerance for line-crossing detection
    ):
        if len(roi_points) != 4:
            raise ValueError("roi_points must contain exactly 4 [x, y] pairs")
        if fps <= 0:
            raise ValueError(f"FPS must be positive, got {fps}")
        if real_width_m <= 0 or real_height_m <= 0:
            raise ValueError("Real-world dimensions must be positive")

        self.fps           = fps
        self.real_width_m  = real_width_m
        self.real_height_m = real_height_m
        self.tolerance     = tolerance
        self.roi_pts       = np.array(roi_points, dtype=np.float32)  # shape (4, 2)

        # ── Build homography (image pixels → bird's-eye pixels) ───────────────
        # We represent the real-world plane at 100 px/metre so maths stays
        # numerically stable and the bird's-eye image is a reasonable size.
        self._SCALE = 100  # px per metre in the bird's-eye plane
        bev_w = real_width_m  * self._SCALE   # bird's-eye width  (px)
        bev_h = real_height_m * self._SCALE   # bird's-eye height (px)

        dst = np.float32([
            [0,     0    ],   # TL → world far-left
            [bev_w, 0    ],   # TR → world far-right
            [bev_w, bev_h],   # BR → world near-right
            [0,     bev_h],   # BL → world near-left
        ])

        self.H = cv2.getPerspectiveTransform(self.roi_pts, dst)

        # ── Trigger lines ──────────────────────────────────────────────────────
        # Line A = far  edge: roi_pts[0] → roi_pts[1]
        # Line B = near edge: roi_pts[3] → roi_pts[2]
        #
        # A vehicle travelling top→bottom crosses Line A first, then Line B.
        # A vehicle travelling bottom→top crosses Line B first, then Line A.
        # The tracker records whichever pair of crossings occurs.
        self.line_a = (tuple(self.roi_pts[0].astype(int)),
                       tuple(self.roi_pts[1].astype(int)))
        self.line_b = (tuple(self.roi_pts[3].astype(int)),
                       tuple(self.roi_pts[2].astype(int)))

    # ── Coordinate projection ─────────────────────────────────────────────────

    def image_to_world(self, image_pt) -> np.ndarray:
        """
        Project an image-space point (x, y in pixels, 1080p) to world
        coordinates (x, y in metres).

        The world origin is the far-left corner of the ROI (roi_pts[0]).
        Y increases from far → near (top of frame → bottom of frame).

        Returns np.array([x_m, y_m]).
        """
        pt = np.float32([[image_pt]]).reshape(1, 1, 2)
        bev_px = cv2.perspectiveTransform(pt, self.H)[0][0]
        return bev_px / self._SCALE   # convert px → metres

    # ── Line-crossing geometry ────────────────────────────────────────────────

    def _dist_point_to_segment(self, point, seg_p1, seg_p2) -> float:
        """
        Perpendicular distance (pixels) from 'point' to the finite segment
        p1→p2.  Points beyond the segment ends clamp to the nearest endpoint.
        """
        p1 = np.array(seg_p1, dtype=float)
        p2 = np.array(seg_p2, dtype=float)
        pt = np.array(point,  dtype=float)
        seg = p2 - p1
        seg_len_sq = float(np.dot(seg, seg))
        if seg_len_sq < 1e-9:
            return float(np.linalg.norm(pt - p1))
        t = float(np.dot(pt - p1, seg)) / seg_len_sq
        t = max(0.0, min(1.0, t))
        closest = p1 + t * seg
        return float(np.linalg.norm(pt - closest))

    def is_crossing_line_a(self, image_pt) -> bool:
        """True if the image point is within `tolerance` pixels of Line A (far edge)."""
        return self._dist_point_to_segment(image_pt, *self.line_a) <= self.tolerance

    def is_crossing_line_b(self, image_pt) -> bool:
        """True if the image point is within `tolerance` pixels of Line B (near edge)."""
        return self._dist_point_to_segment(image_pt, *self.line_b) <= self.tolerance

    def is_inside_roi(self, image_pt) -> bool:
        """True if image_pt lies inside (or on the boundary of) the ROI polygon."""
        pt = (float(image_pt[0]), float(image_pt[1]))
        return cv2.pointPolygonTest(self.roi_pts, pt, measureDist=False) >= 0

    # ── Speed calculation ─────────────────────────────────────────────────────

    def calculate(
        self,
        world_pos_a: np.ndarray,  # world coords (metres) at first crossing
        world_pos_b: np.ndarray,  # world coords (metres) at second crossing
        frame_a: int,             # frame number of first  crossing
        frame_b: int,             # frame number of second crossing
    ) -> dict:
        """
        Compute speed from the two timed crossing events.

        Because positions are in real-world metres (after perspective correction),
        the speed is accurate regardless of where in the frame the crossings occur.

        Returns {'speed_mps': float, 'speed_kmph': float}
        """
        frame_diff = abs(frame_b - frame_a)
        if frame_diff == 0:
            return {'speed_mps': 0.0, 'speed_kmph': 0.0}

        time_s     = frame_diff / self.fps
        dist_m     = float(np.linalg.norm(
            np.array(world_pos_b, dtype=float) -
            np.array(world_pos_a, dtype=float)
        ))

        if dist_m < 0.1:
            # Implausibly short — probably a spurious double-crossing on the
            # same line rather than a genuine A→B traversal.
            return {'speed_mps': 0.0, 'speed_kmph': 0.0}

        speed_mps = dist_m / time_s
        return {
            'speed_mps':  round(speed_mps, 2),
            'speed_kmph': round(speed_mps * 3.6, 2),
        }

    def determine_direction(self, frame_a: int, frame_b: int) -> str:
        """
        Direction is inferred from which trigger line was crossed first.
        Line A = far/top → Line B = near/bottom.
        """
        if frame_a <= frame_b:
            return "Top → Bottom"
        else:
            return "Bottom → Top"

    # ── Drawing ───────────────────────────────────────────────────────────────

    def draw_roi(self, frame, color_roi=(0, 0, 255), color_lines=(0, 180, 255)):
        """
        Overlay the calibration quadrilateral and the two trigger lines on `frame`.

        color_roi:   BGR colour for the outer polygon.
        color_lines: BGR colour for Line A / Line B highlights.
        """
        pts = self.roi_pts.astype(np.int32).reshape(-1, 1, 2)

        # Filled semi-transparent overlay for the ROI interior
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 0, 255))
        cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)

        # Outer polygon border
        cv2.polylines(frame, [pts], isClosed=True, color=color_roi, thickness=2)

        # Line A (far / top)
        a0, a1 = self.line_a
        cv2.line(frame, a0, a1, color_lines, 2)
        cv2.putText(frame, "Line A (far)",
                    (a0[0] + 6, a0[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_lines, 1)

        # Line B (near / bottom)
        b0, b1 = self.line_b
        cv2.line(frame, b0, b1, color_lines, 2)
        cv2.putText(frame, "Line B (near)",
                    (b0[0] + 6, b0[1] + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_lines, 1)

        # Numbered corner markers
        corner_labels = ["1 TL", "2 TR", "3 BR", "4 BL"]
        for i, pt in enumerate(self.roi_pts):
            px, py = int(pt[0]), int(pt[1])
            cv2.circle(frame, (px, py), 7, color_roi, -1)
            cv2.putText(frame, corner_labels[i], (px + 10, py + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color_roi, 1)

    def get_config_dict(self) -> dict:
        """
        Return a dict ready to be serialised as calibration.json.
        """
        return {
            'roi_points':    self.roi_pts.tolist(),
            'real_width_m':  self.real_width_m,
            'real_height_m': self.real_height_m,
        }