# src/vehicle_tracker.py
# ─────────────────────────────────────────────────────────────────────────────
# Responsibility: Per-vehicle state management, ROI line crossing detection,
# and plate text voting.
#
# Upgrade from the two-line version:
#   Previously the tracker checked whether a vehicle's Y coordinate was within
#   LINE_TOLERANCE pixels of a fixed horizontal line (LINE_1_Y / LINE_2_Y).
#
#   Now it asks PerspectiveSpeedCalculator.is_crossing_line_a/b() which
#   measures the perpendicular pixel distance from the vehicle's ground contact
#   point to the ROI's far or near edge — the two edges of the calibration
#   quadrilateral drawn on the road by the user.
#
#   The crossing positions are also projected into real-world metres before
#   being handed to the speed calculation, so the result is accurate
#   regardless of camera angle or zoom level.
#
# Ground contact point:
#   We use the bottom-centre of the bounding box (cx, y2) as the vehicle's
#   ground contact point. This is more geometrically meaningful than the box
#   centre — it's the point closest to where the tyres touch the road, which
#   is what the perspective transform was calibrated for.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import time
from collections import Counter

import numpy as np

from src.perspective_speed import PerspectiveSpeedCalculator


class VehicleTracker:

    def __init__(self, perspective_calc: PerspectiveSpeedCalculator):
        """
        perspective_calc: a fully initialised PerspectiveSpeedCalculator.
                          It carries fps, real-world dimensions, and the
                          homography — everything needed to compute speed.
        """
        self.perspective_calc  = perspective_calc
        self._active           = {}   # track_id → state dict
        self.completed_records = []   # filled by _finalise(), drained by caller

    # ── State template ────────────────────────────────────────────────────────

    def _make_state(self) -> dict:
        return {
            # Crossing events — whichever pair fires first wins
            'line_a_frame': None,     # frame number when ground pt hit Line A
            'line_b_frame': None,     # frame number when ground pt hit Line B
            'world_pos_a':  None,     # np.array([x_m, y_m]) at Line A crossing
            'world_pos_b':  None,     # np.array([x_m, y_m]) at Line B crossing

            # OCR accumulation (only corrected, validated UK plates enter here)
            'plate_votes':  [],
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        track_id:      int,
        bbox:          tuple,         # (x1, y1, x2, y2) in 1080p pixels
        current_frame: int,
        plate_text:    str | None = None,
    ):
        """
        Call once per processed frame for each SORT-tracked vehicle.

        plate_text must already be corrected and validated (or None).
        """
        if track_id not in self._active:
            self._active[track_id] = self._make_state()

        state = self._active[track_id]

        # Accumulate validated plate reads
        if plate_text:
            state['plate_votes'].append(plate_text)

        # ── Ground contact point ───────────────────────────────────────────────
        x1, y1, x2, y2 = bbox
        ground_pt = ((x1 + x2) // 2, y2)   # bottom-centre of bounding box

        # ── Line A crossing (far / top edge) ──────────────────────────────────
        if (state['line_a_frame'] is None and
                self.perspective_calc.is_crossing_line_a(ground_pt)):
            state['line_a_frame'] = current_frame
            state['world_pos_a']  = self.perspective_calc.image_to_world(ground_pt)
            print(f"  [Tracker] ID:{track_id} crossed Line A "
                  f"at frame {current_frame}, "
                  f"world=({state['world_pos_a'][0]:.1f}m, "
                  f"{state['world_pos_a'][1]:.1f}m)")

        # ── Line B crossing (near / bottom edge) ──────────────────────────────
        if (state['line_b_frame'] is None and
                self.perspective_calc.is_crossing_line_b(ground_pt)):
            state['line_b_frame'] = current_frame
            state['world_pos_b']  = self.perspective_calc.image_to_world(ground_pt)
            print(f"  [Tracker] ID:{track_id} crossed Line B "
                  f"at frame {current_frame}, "
                  f"world=({state['world_pos_b'][0]:.1f}m, "
                  f"{state['world_pos_b'][1]:.1f}m)")

        # ── Finalise when both crossings are recorded ─────────────────────────
        if (state['line_a_frame'] is not None and
                state['line_b_frame'] is not None):
            self._finalise(track_id, state)

    def get_active_ids(self) -> list[int]:
        return list(self._active.keys())

    def get_plate_for(self, track_id: int) -> str | None:
        """Best live plate guess using the votes collected so far."""
        if track_id in self._active:
            votes = self._active[track_id]['plate_votes']
            if votes:
                return self._character_level_vote(votes)
        return None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _finalise(self, track_id: int, state: dict):
        """
        Compute the final speed and plate, package into a record dict, and
        append to completed_records for the caller to drain.
        """
        speed_data = self.perspective_calc.calculate(
            world_pos_a = state['world_pos_a'],
            world_pos_b = state['world_pos_b'],
            frame_a     = state['line_a_frame'],
            frame_b     = state['line_b_frame'],
        )
        direction   = self.perspective_calc.determine_direction(
            state['line_a_frame'], state['line_b_frame']
        )
        final_plate = self._character_level_vote(state['plate_votes'])

        # Compute the real-world distance for logging
        dist_m = float(np.linalg.norm(
            np.array(state['world_pos_b']) - np.array(state['world_pos_a'])
        ))

        record = {
            'track_id':   track_id,
            'plate':      final_plate,
            'speed_kmph': speed_data['speed_kmph'],
            'speed_mps':  speed_data['speed_mps'],
            'direction':  direction,
            'timestamp':  time.strftime('%Y-%m-%d %H:%M:%S'),
            'ocr_reads':  len(state['plate_votes']),
            'dist_m':     round(dist_m, 2),
        }

        self.completed_records.append(record)
        print(
            f"[RECORD] ID:{track_id:>3} | "
            f"Plate: {final_plate:<12} | "
            f"Speed: {speed_data['speed_kmph']:>6.1f} km/h | "
            f"Dist: {dist_m:.1f}m | "
            f"Reads: {len(state['plate_votes'])}"
        )
        del self._active[track_id]

    def _character_level_vote(self, votes: list) -> str:
        """
        Resolve disagreements between near-correct OCR readings by voting on
        each character position independently.

        Every string in 'votes' has already been corrected and validated by
        plate_reader.py, so:
          - All strings are 7 characters for standard UK formats.
          - Every character is the correct type (letter/digit) for its position.

        This voter only resolves residual misreads like 'EE15NER' vs 'EE15NEB'.

        Algorithm:
          1. Find the most common string length (guards against any edge cases).
          2. At each character position, pick the character with the most votes.
          3. Join per-position winners → final plate string.
        """
        if not votes:
            return 'Not Detected'

        target_length = Counter(len(v) for v in votes).most_common(1)[0][0]
        aligned       = [v for v in votes if len(v) == target_length]

        if not aligned:
            return 'Not Detected'

        result = []
        for i in range(target_length):
            chars   = [v[i] for v in aligned]
            winner  = Counter(chars).most_common(1)[0][0]
            result.append(winner)

        final = ''.join(result)
        print(
            f"  [Vote] {len(aligned)} clean reads → '{final}' "
            f"(pool: {dict(Counter(votes))})"
        )
        return final