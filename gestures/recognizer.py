"""
Deterministic Gesture Engine
Human-stable, XR-style logic
No ML. No magic.
"""

import time

from gestures.finger_state import (
    get_finger_states,
    count_extended_fingers,
    distance_3d,
    THUMB_TIP,
    INDEX_TIP,
    MIDDLE_TIP,
    RING_TIP,
    PINKY_TIP
)


class GestureRecognizer:
    def __init__(self):
        # Thresholds
        self.pinch_threshold = 0.045
        self.hold_time = 0.35  # seconds

        # State memory for HOLD gestures
        self.last_gesture = None
        self.gesture_start_time = 0.0

    # -------------------------------------------------
    # CORE HOLD LOGIC
    # -------------------------------------------------

    def _is_held(self, gesture_name):
        now = time.time()

        if gesture_name != self.last_gesture:
            self.last_gesture = gesture_name
            self.gesture_start_time = now
            return False

        return (now - self.gesture_start_time) >= self.hold_time

    # -------------------------------------------------
    # BASIC DETECTORS
    # -------------------------------------------------

    def detect_open_palm(self, landmarks):
        return count_extended_fingers(landmarks) >= 4

    def detect_fist(self, landmarks):
        return count_extended_fingers(landmarks) <= 1

    def detect_pinch(self, landmarks):
        d = distance_3d(landmarks[THUMB_TIP], landmarks[INDEX_TIP])
        return d < self.pinch_threshold

    # -------------------------------------------------
    # 🔑 FIXED INDEX POINTING (DOMINANCE-BASED)
    # -------------------------------------------------

    def detect_index_point(self, landmarks):
        s = get_finger_states(landmarks)

        # Index MUST be extended
        if not s["INDEX"]:
            return False

        # Other fingers must NOT dominate
        noise_count = sum([
            s["MIDDLE"],
            s["RING"],
            s["PINKY"]
        ])

        # Allow one noisy finger (human realistic)
        return noise_count <= 1

    # -------------------------------------------------
    # ADVANCED POSES
    # -------------------------------------------------

    def detect_three_finger(self, landmarks):
        s = get_finger_states(landmarks)
        return (
            s["INDEX"] and
            s["MIDDLE"] and
            s["RING"] and
            not s["PINKY"]
        )

    def detect_four_finger(self, landmarks):
        s = get_finger_states(landmarks)
        return (
            s["INDEX"] and
            s["MIDDLE"] and
            s["RING"] and
            s["PINKY"]
        )

    def detect_precision_mode(self, landmarks):
        """Also called 'pointer' or 'gun' gesture - thumb + index extended."""
        s = get_finger_states(landmarks)
        return (
            s["THUMB"] and
            s["INDEX"] and
            not s["MIDDLE"] and
            not s["RING"] and
            not s["PINKY"]
        )
    
    def detect_pointer(self, landmarks):
        """Alias for precision_mode - thumb + index up (gun gesture)."""
        return self.detect_precision_mode(landmarks)

    # -------------------------------------------------
    # HOLD GESTURES
    # -------------------------------------------------

    def detect_pinch_hold(self, landmarks):
        if not self.detect_pinch(landmarks):
            return False
        return self._is_held("PINCH")

    def detect_fist_hold(self, landmarks):
        if not self.detect_fist(landmarks):
            return False
        return self._is_held("FIST")

    # -------------------------------------------------
    # SINGLE HAND GESTURE OS (PRIORITY FIXED)
    # -------------------------------------------------

    def recognize_single_hand(self, landmarks):
        if landmarks is None or len(landmarks) < 21:
            return "NONE"

        # -------- HOLD (highest priority) --------
        if self.detect_pinch_hold(landmarks):
            return "GRAB_DRAG"

        if self.detect_fist_hold(landmarks):
            return "ERASE_CONTINUOUS"

        # -------- DRAW (POINTER GESTURE - thumb + index) --------
        if self.detect_pointer(landmarks):
            return "pointer"  # For voxel drawing

        # -------- LEGACY index point (lower priority) --------
        if self.detect_index_point(landmarks):
            return "index_point"

        # -------- CAMERA CONTROL --------
        if self.detect_three_finger(landmarks):
            return "MOVE_CAMERA"

        if self.detect_four_finger(landmarks):
            return "ROTATE_CAMERA"

        # -------- MODE / ACTION --------
        if self.detect_pinch(landmarks):
            return "pinch"  # For erasing

        if self.detect_fist(landmarks):
            return "fist"  # For hold mode

        if self.detect_open_palm(landmarks):
            return "open_palm"  # For rotation

        return "UNKNOWN"

    # -------------------------------------------------
    # TWO HAND GESTURES
    # -------------------------------------------------

    def recognize_two_hands(self, left, right):
        if left is None or right is None:
            return None

        if self.detect_open_palm(left) and self.detect_open_palm(right):
            return "ZOOM"

        if self.detect_pinch(left) and self.detect_pinch(right):
            return "SCALE_OBJECT"

        return None
