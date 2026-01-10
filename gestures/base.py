"""
Gesture base class.
"""

class Gesture:
    def detect(self, landmarks):
        raise NotImplementedError
