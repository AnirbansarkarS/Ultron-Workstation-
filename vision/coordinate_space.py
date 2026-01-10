import numpy as np

class CoordinateSpace:
    """
    Converts MediaPipe normalized landmarks
    into a consistent right-handed space
    """

    def __init__(self):
        self.origin = None

    def normalize(self, landmarks):
        """
        Input: list of (x, y, z) in MediaPipe space
        Output: list of centered, scaled 3D points
        """

        pts = np.array(landmarks)

        # Use wrist (landmark 0) as origin
        wrist = pts[0]

        if self.origin is None:
            self.origin = wrist

        # Translate
        pts = pts - wrist

        # Flip Y for natural math (up is positive)
        pts[:, 1] *= -1

        # Scale normalization
        scale = np.linalg.norm(pts[9]) + 1e-6  # middle finger MCP
        pts /= scale

        return pts
