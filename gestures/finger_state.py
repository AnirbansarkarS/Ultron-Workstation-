"""
Finger state detection using geometric thresholds.
Pure deterministic logic (NO ML).
"""

import numpy as np

# MediaPipe landmark indices
WRIST = 0

THUMB_TIP = 4
THUMB_IP = 3
THUMB_MCP = 2

INDEX_TIP = 8
INDEX_PIP = 6
INDEX_MCP = 5

MIDDLE_TIP = 12
MIDDLE_PIP = 10
MIDDLE_MCP = 9

RING_TIP = 16
RING_PIP = 14
RING_MCP = 13

PINKY_TIP = 20
PINKY_PIP = 18
PINKY_MCP = 17


def distance_3d(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def is_thumb_extended(landmarks):
    thumb_tip = landmarks[THUMB_TIP]
    thumb_mcp = landmarks[THUMB_MCP]
    wrist = landmarks[WRIST]

    tip_to_mcp = distance_3d(thumb_tip, thumb_mcp)
    mcp_to_wrist = distance_3d(thumb_mcp, wrist)

    return tip_to_mcp > mcp_to_wrist * 0.6


def is_finger_extended(landmarks, tip, pip, mcp):
    tip_to_mcp = distance_3d(landmarks[tip], landmarks[mcp])
    pip_to_mcp = distance_3d(landmarks[pip], landmarks[mcp])

    return tip_to_mcp > pip_to_mcp * 1.1


def get_finger_states(landmarks):
    return {
        "THUMB": is_thumb_extended(landmarks),
        "INDEX": is_finger_extended(landmarks, INDEX_TIP, INDEX_PIP, INDEX_MCP),
        "MIDDLE": is_finger_extended(landmarks, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
        "RING": is_finger_extended(landmarks, RING_TIP, RING_PIP, RING_MCP),
        "PINKY": is_finger_extended(landmarks, PINKY_TIP, PINKY_PIP, PINKY_MCP),
    }


def count_extended_fingers(landmarks):
    return sum(get_finger_states(landmarks).values())
