"""
Open palm detection logic.
"""
from .base import Gesture

class OpenPalmGesture(Gesture):
    def detect(self, landmarks):
        # Implementation to check if palm is flat
        return False
