"""
Gesture engine to resolve active gesture.
"""

class GestureEngine:
    def __init__(self):
        self.gestures = []

    def update(self, landmarks):
        for gesture in self.gestures:
            if gesture.detect(landmarks):
                return gesture
        return None
