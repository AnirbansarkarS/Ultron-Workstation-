import cv2

class Camera:
    def __init__(self, index=0, width=1280, height=720):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        success, frame = self.cap.read()
        if not success:
            return None
        return cv2.flip(frame, 1)  # mirror for natural interaction

    def release(self):
        self.cap.release()
