# src/speed_calculator.py
# ─────────────────────────────────────────────────────────────────────────────
# Responsibility: Pure speed calculation logic.
# This module contains a single class that has no knowledge of YOLO, OCR, or
# OpenCV. Its only job is: given frame numbers and calibration data, compute
# a speed. This makes it very easy to unit-test independently.
#
# The Physics:
#   speed = distance / time
#   distance = real_world_meters between the two lines (measured physically)
#   time = (frame_line2 - frame_line1) / fps   [converts frames → seconds]
# ─────────────────────────────────────────────────────────────────────────────


class SpeedCalculator:
    """
    Computes vehicle speed from two frame-crossing timestamps and calibration data.

    The key insight behind this approach is that we don't need GPS or any
    external sensor. We just need:
      1. Two known reference lines in the video (LINE_1_Y and LINE_2_Y in config)
      2. The real-world distance between those lines (measured physically)
      3. The video's frame rate (obtained from OpenCV when opening the video)

    Everything else is just arithmetic.
    """

    def __init__(self, fps: float, real_distance_meters: float):
        """
        fps:                   Frames per second of the video source.
                               Get this with cap.get(cv2.CAP_PROP_FPS) in main.py.
        real_distance_meters:  Physical distance between LINE_1 and LINE_2 in meters.
                               This is your calibration constant — the accuracy of
                               your speed readings depends entirely on this value
                               being correct.
        """
        if fps <= 0:
            raise ValueError(f"FPS must be positive, got {fps}")
        if real_distance_meters <= 0:
            raise ValueError(f"real_distance_meters must be positive, got {real_distance_meters}")

        self.fps = fps
        self.real_distance_meters = real_distance_meters

    def calculate(self, frame_line1: int, frame_line2: int) -> dict:
        """
        Given the frame numbers at which a vehicle crossed each line,
        return a dictionary with speed in both m/s and km/h.

        frame_line1: Frame number when the vehicle crossed the first reference line.
        frame_line2: Frame number when the vehicle crossed the second reference line.

        Returns a dict: { 'speed_mps': float, 'speed_kmph': float }

        Why return a dict instead of just a number?
        Because returning structured data makes the calling code more readable:
          result['speed_kmph']   is clearer than
          result[1]              or result[0]
        And in the future you can add more fields (e.g. acceleration) without
        changing the function signature.
        """

        frame_diff = abs(frame_line2 - frame_line1)

        # Guard against division by zero — can happen if a fast car crosses
        # both lines in the same frame (very unlikely but worth handling).
        if frame_diff == 0:
            return {'speed_mps': 0.0, 'speed_kmph': 0.0}

        # Convert frame difference to real time
        # e.g. 60 frames at 30fps = 2.0 seconds
        time_seconds = frame_diff / self.fps

        # Core calculation
        speed_mps  = self.real_distance_meters / time_seconds
        speed_kmph = speed_mps * 3.6   # 1 m/s = 3.6 km/h

        return {
            'speed_mps':  round(speed_mps,  2),
            'speed_kmph': round(speed_kmph, 2)
        }

    def determine_direction(self, frame_line1: int, frame_line2: int) -> str:
        """
        Determine which direction the car was travelling based on which
        reference line it crossed first.

        If line 1 (higher up in the frame, smaller Y) was crossed before line 2
        (lower in the frame, larger Y), the car is moving downward (top to bottom).
        This handles cars coming from both directions automatically.
        """
        if frame_line1 < frame_line2:
            return "Top → Bottom"
        else:
            return "Bottom → Top"