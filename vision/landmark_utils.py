import numpy as np

def calculate_distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

def normalize_landmarks(landmarks, reference_point_idx=0):
    """
    Normalize landmarks relative to a reference point (e.g., wrist).
    """
    if not landmarks:
        return []
    ref = np.array(landmarks[reference_point_idx])
    return [np.array(lm) - ref for lm in landmarks]

def denormalize_point(point, width, height):
    """
    Convert normalized [0, 1] coordinates to pixel coordinates.
    """
    return (int(point[0] * width), int(point[1] * height))

def normalize_point(point, width, height):
    """
    Convert pixel coordinates to normalized [0, 1] coordinates.
    """
    return (point[0] / width, point[1] / height)
