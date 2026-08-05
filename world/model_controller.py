"""
ModelController — translates hand gestures into transforms on a ModelObject.

Mirrors VoxelEditor's interface so main.py integration is minimal.

Gesture → Model action mapping
────────────────────────────────────────────────────────────────
open_palm   →  MODEL_ROTATE   (palm X/Y movement → Y/X axis rotation)
pinch_hold  →  MODEL_GRAB     (drag index tip → translate model)
ZOOM        →  MODEL_SCALE    (two-hand spread → uniform scale)
fist        →  HOLD           (freeze everything)
────────────────────────────────────────────────────────────────
"""

import math
import time
import numpy as np

from vision.depth_mapper import map_depth_to_world


class ModelController:
    """
    Manages gesture-driven transforms for a ModelObject.

    Parameters
    ----------
    model_obj : ModelObject — the model to control
    """

    ROTATION_SENSITIVITY = 2.5   # radians per normalised-coord unit
    TRANSLATE_SENSITIVITY = 8.0  # world units per normalised-coord unit
    SCALE_DEADZONE        = 0.01 # minimum ratio-change to apply scale

    def __init__(self, model_obj):
        self.model    = model_obj
        self.mode     = "IDLE"

        # Rotation state
        self._rot_base_x  = None
        self._rot_base_y  = None

        # Translation state
        self._grab_start  = None   # np.array [wx, wy, wz]

        # Scale state
        self._scale_ref_dist = 0.0

    # ─────────────────────── mode update ─────────────────────────────

    def update_mode(self, gesture):
        """
        Called every frame with the stable gesture from GestureStateMachine.
        Updates self.mode and resets stale state on transition.
        """
        prev = self.mode

        if gesture == "open_palm":
            self.mode = "MODEL_ROTATE"
        elif gesture in ("pinch_hold", "GRAB_DRAG", "grab"):
            self.mode = "MODEL_GRAB"
        elif gesture in ("ZOOM", "SCALE_OBJECT"):
            self.mode = "MODEL_SCALE"
        elif gesture == "fist":
            self.mode = "HOLD"
        else:
            self.mode = "IDLE"

        # Clear stale per-session state on mode change
        if self.mode != prev:
            self._rot_base_x     = None
            self._rot_base_y     = None
            self._grab_start     = None
            self._scale_ref_dist = 0.0

    # ─────────────────────── rotation ─────────────────────────────────

    def update_rotation(self, hand_x: float, hand_y: float):
        """
        Rotate the model based on open-palm movement.

        Parameters
        ----------
        hand_x, hand_y : normalised [0, 1] hand position (from palm landmark)
        """
        if self.mode != "MODEL_ROTATE":
            return

        if self._rot_base_x is None:
            self._rot_base_x = hand_x
            self._rot_base_y = hand_y
            return

        dx = (hand_x - self._rot_base_x) * self.ROTATION_SENSITIVITY
        dy = (hand_y - self._rot_base_y) * self.ROTATION_SENSITIVITY

        # Horizontal hand move → rotate around Y; vertical → around X
        if abs(dx) > 0.001:
            self.model.rotate('y', dx)
        if abs(dy) > 0.001:
            self.model.rotate('x', dy)

        # Update base so rotation feels relative
        self._rot_base_x = hand_x
        self._rot_base_y = hand_y

    def reset_rotation(self):
        """Call when hand leaves frame or mode changes."""
        self._rot_base_x = None
        self._rot_base_y = None

    # ─────────────────────── translation ──────────────────────────────

    def update_grab(self, hand_x: float, hand_y: float, hand_z: float):
        """
        Translate the model by tracking the index-tip position.

        Parameters
        ----------
        hand_x, hand_y : normalised position
        hand_z         : MediaPipe z depth value
        """
        if self.mode != "MODEL_GRAB":
            return

        # Convert to approximate world coords
        wx = (hand_x - 0.5) * self.TRANSLATE_SENSITIVITY
        wy = -(hand_y - 0.5) * self.TRANSLATE_SENSITIVITY
        wz = map_depth_to_world(hand_z, min_depth=-3, max_depth=3)
        current = np.array([wx, wy, wz], dtype=np.float32)

        if self._grab_start is None:
            self._grab_start = current
            return

        delta = current - self._grab_start

        # Dead-zone to prevent micro-jitter
        if np.linalg.norm(delta) > 0.05:
            self.model.translate(float(delta[0]),
                                 float(delta[1]),
                                 float(delta[2]))
            self._grab_start = current

    def reset_grab(self):
        self._grab_start = None

    # ─────────────────────── scale ────────────────────────────────────

    def update_scale(self, landmarks_list):
        """
        Scale the model using two-hand pinch distance.

        Parameters
        ----------
        landmarks_list : list of hand landmark arrays (expects 2 hands)
        """
        if self.mode != "MODEL_SCALE":
            return
        if len(landmarks_list) < 2:
            return

        p1 = np.array(landmarks_list[0][0])   # wrist 1
        p2 = np.array(landmarks_list[1][0])   # wrist 2
        dist = float(np.linalg.norm(p1 - p2))

        if self._scale_ref_dist < 1e-4:
            self._scale_ref_dist = dist
            return

        ratio = dist / self._scale_ref_dist
        if abs(ratio - 1.0) > self.SCALE_DEADZONE:
            # Clamp to avoid explosive scaling
            ratio = max(0.92, min(1.08, ratio))
            self.model.scale(ratio)
            self._scale_ref_dist = dist

    # ─────────────────────── combined update ─────────────────────────

    def process(self, gesture, landmarks_list, w, h):
        """
        Full per-frame update — call once per frame from main loop.

        Parameters
        ----------
        gesture        : str — stable gesture from GestureStateMachine
        landmarks_list : list of hand landmark lists
        w, h           : frame width/height (unused but kept for API symmetry)
        """
        self.update_mode(gesture)

        if not landmarks_list:
            self.reset_rotation()
            self.reset_grab()
            return

        lm0 = landmarks_list[0]   # primary hand

        if self.mode == "MODEL_ROTATE":
            palm = lm0[0]          # wrist = landmark 0
            self.update_rotation(palm[0], palm[1])

        elif self.mode == "MODEL_GRAB":
            idx = lm0[8]           # index tip = landmark 8
            self.update_grab(idx[0], idx[1], idx[2])

        elif self.mode == "MODEL_SCALE":
            self.update_scale(landmarks_list)

        else:
            self.reset_rotation()
            self.reset_grab()
