# src/vehicle_tracker.py
# ─────────────────────────────────────────────────────────────────────────────
# Responsibility: Per-vehicle state management, line crossing detection,
# and plate text voting.
#
# Simpler than the previous version because format detection and OCR correction
# now happen in plate_reader.py BEFORE strings enter the vote pool. Every
# entry in plate_votes is already a corrected, strictly valid UK plate string.
#
# The voter therefore only needs to resolve disagreements between near-correct
# readings (e.g. 'EE15NER' vs 'EE15NEB' — same plate, last char misread).
# Character-level voting handles this cleanly without needing correction maps.
# ─────────────────────────────────────────────────────────────────────────────

import time
import config
from collections import Counter
from src.speed_calculator import SpeedCalculator


class VehicleTracker:

    def __init__(self, line1_y: int, line2_y: int,
                 real_distance_m: float, fps: float):
        self.line1_y        = line1_y
        self.line2_y        = line2_y
        self.speed_calc     = SpeedCalculator(fps, real_distance_m)
        self._active        = {}
        self.completed_records = []
        self.LINE_TOLERANCE = config.LINE_TOLERANCE

    def _make_state(self) -> dict:
        return {
            'line1_frame': None,
            'line2_frame': None,
            'plate_votes': [],   # only corrected, valid UK plates enter here
        }

    def _is_crossing(self, center_y: int, line_y: int) -> bool:
        return abs(center_y - line_y) <= self.LINE_TOLERANCE

    def _character_level_vote(self, votes: list) -> str:
        """
        Pick the best plate string by voting on each character position
        independently.

        Because every string in votes has already been corrected and validated
        by plate_reader.py, we know:
          - All strings are the same length (7 for standard UK formats)
          - Every character is the right type for its position

        So this voter only needs to resolve residual disagreements like
        'EE15NER' vs 'EE15NEB' — both are valid, the correct one just needs
        to win on position 6 (R vs B).

        Algorithm:
          1. Group votes by length, use the most common length
          2. For each character position, pick the character that appeared most
          3. Join the per-position winners into the final plate string

        No correction maps needed here — that work is already done upstream.
        """
        if not votes:
            return 'Not Detected'

        # Use the most common length to align reads.
        # In practice all reads should be length 7, but this guards against
        # any edge case where a shorter read slips through.
        target_length = Counter(len(v) for v in votes).most_common(1)[0][0]
        aligned       = [v for v in votes if len(v) == target_length]

        if not aligned:
            return 'Not Detected'

        # Vote position by position
        result = []
        for i in range(target_length):
            chars_at_pos = [v[i] for v in aligned]
            winner       = Counter(chars_at_pos).most_common(1)[0][0]
            result.append(winner)

        final = ''.join(result)
        print(
            f"  [Vote] {len(aligned)} clean reads → '{final}' "
            f"(pool: {dict(Counter(votes))})"
        )
        return final

    def update(self, track_id: int, bbox: tuple,
               current_frame: int, plate_text: str | None = None):
        """
        Called once per processed frame for each SORT-tracked vehicle.
        plate_text is already corrected and validated — or None if plate_reader
        found nothing worth keeping this frame.
        """
        if track_id not in self._active:
            self._active[track_id] = self._make_state()

        state = self._active[track_id]

        # Only corrected, validated strings reach this point — no filtering needed
        if plate_text:
            state['plate_votes'].append(plate_text)

        x1, y1, x2, y2 = bbox
        center_y = (y1 + y2) // 2

        if state['line1_frame'] is None and self._is_crossing(center_y, self.line1_y):
            state['line1_frame'] = current_frame

        if state['line2_frame'] is None and self._is_crossing(center_y, self.line2_y):
            state['line2_frame'] = current_frame

        if state['line1_frame'] is not None and state['line2_frame'] is not None:
            self._finalise(track_id, state)

    def _finalise(self, track_id: int, state: dict):
        speed_data  = self.speed_calc.calculate(
            state['line1_frame'], state['line2_frame']
        )
        direction   = self.speed_calc.determine_direction(
            state['line1_frame'], state['line2_frame']
        )
        final_plate = self._character_level_vote(state['plate_votes'])

        record = {
            'track_id':   track_id,
            'plate':      final_plate,
            'speed_kmph': speed_data['speed_kmph'],
            'speed_mps':  speed_data['speed_mps'],
            'direction':  direction,
            'timestamp':  time.strftime('%Y-%m-%d %H:%M:%S'),
            'ocr_reads':  len(state['plate_votes']),
        }

        self.completed_records.append(record)
        print(
            f"[RECORD] ID:{track_id:>3} | "
            f"Plate: {final_plate:<12} | "
            f"Speed: {speed_data['speed_kmph']:>6.1f} km/h | "
            f"Reads: {len(state['plate_votes'])}"
        )
        del self._active[track_id]

    def get_active_ids(self) -> list[int]:
        return list(self._active.keys())

    def get_plate_for(self, track_id: int) -> str | None:
        """Live display best guess using current vote pool."""
        if track_id in self._active:
            votes = self._active[track_id]['plate_votes']
            if votes:
                return self._character_level_vote(votes)
        return None