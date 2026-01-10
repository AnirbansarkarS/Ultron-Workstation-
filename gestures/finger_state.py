"""
Finger state detection using geometric thresholds.
No ML - pure deterministic logic.
"""
import numpy as np

# MediaPipe hand landmark indices
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
    """Calculate Euclidean distance between two 3D points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

def is_thumb_extended(landmarks):
    """Thumb is extended if tip is far from palm base."""
    thumb_tip = landmarks[THUMB_TIP]
    thumb_mcp = landmarks[THUMB_MCP]
    wrist = landmarks[WRIST]
    
    # Distance from tip to MCP (metacarpal base)
    tip_to_mcp = distance_3d(thumb_tip, thumb_mcp)
    # Distance from MCP to wrist
    mcp_to_wrist = distance_3d(thumb_mcp, wrist)
    
    # Thumb is extended if tip is far from base
    return tip_to_mcp > mcp_to_wrist * 0.6

def is_finger_extended(landmarks, finger_name):
    """
    Detect if a finger is extended.
    
    Args:
        landmarks: List of (x, y, z) tuples
        finger_name: 'INDEX', 'MIDDLE', 'RING', or 'PINKY'
    
    Returns:
        bool: True if finger is extended
    """
    finger_map = {
        'INDEX': (INDEX_TIP, INDEX_PIP, INDEX_MCP),
        'MIDDLE': (MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
        'RING': (RING_TIP, RING_PIP, RING_MCP),
        'PINKY': (PINKY_TIP, PINKY_PIP, PINKY_MCP)
    }
    
    if finger_name not in finger_map:
        return False
    
    tip_idx, pip_idx, mcp_idx = finger_map[finger_name]
    
    tip = landmarks[tip_idx]
    pip = landmarks[pip_idx]
    mcp = landmarks[mcp_idx]
    
    # Finger is extended if tip is above PIP (in y-axis, normalized coords)
    # and tip-to-mcp distance is larger than pip-to-mcp
    tip_to_mcp = distance_3d(tip, mcp)
    pip_to_mcp = distance_3d(pip, mcp)
    
    return tip_to_mcp > pip_to_mcp * 1.1

def get_finger_states(landmarks):
    """
    Get state of all fingers.
    
    Returns:
        dict: {'THUMB': bool, 'INDEX': bool, 'MIDDLE': bool, 'RING': bool, 'PINKY': bool}
    """
    return {
        'THUMB': is_thumb_extended(landmarks),
        'INDEX': is_finger_extended(landmarks, 'INDEX'),
        'MIDDLE': is_finger_extended(landmarks, 'MIDDLE'),
        'RING': is_finger_extended(landmarks, 'RING'),
        'PINKY': is_finger_extended(landmarks, 'PINKY')
    }

def count_extended_fingers(landmarks):
    """Count how many fingers are extended."""
    states = get_finger_states(landmarks)
    return sum(states.values())
