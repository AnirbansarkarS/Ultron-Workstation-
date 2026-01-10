from gestures import (
    detect_open_palm,
    detect_fist,
    detect_index_only,
    detect_pinch
)

class GestureEngine:
    def __init__(self):
        self.current = "UNKNOWN"

    def resolve(self, fingers, landmarks):
        if detect_pinch(landmarks):
            self.current = "PINCH"
        elif detect_fist(fingers):
            self.current = "ERASE"
        elif detect_index_only(fingers):
            self.current = "DRAW"
        elif detect_open_palm(fingers):
            self.current = "IDLE"
        else:
            self.current = "UNKNOWN"

        return self.current
