import cv2
from vision.camera import Camera
from vision.hand_tracker import HandTracker
import time
import numpy as np

# -------- ANTIGRAVITY PROMPT --------
ANTIGRAVITY_PROMPT = "ULTRON"

# Hand connections for drawing skeleton manually
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

def draw_hand(frame, landmarks):
    h, w, _ = frame.shape
    # Convert normalized to pixel coordinates
    points = []
    for lm in landmarks:
        points.append((int(lm[0] * w), int(lm[1] * h)))

    # Draw connections
    for connection in HAND_CONNECTIONS:
        pt1 = points[connection[0]]
        pt2 = points[connection[1]]
        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    # Draw landmarks
    for pt in points:
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)

def main():
    cam = Camera()
    tracker = HandTracker()

    window_name = "Ultron Workstation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    prev_time = 0

    while True:
        frame = cam.read()
        if frame is None:
            break

        all_landmarks, _ = tracker.process(frame)

        if all_landmarks:
            for landmarks in all_landmarks:
                draw_hand(frame, landmarks)

        # FPS counter
        curr_time = time.time()
        fps = int(1 / (curr_time - prev_time)) if prev_time else 0
        prev_time = curr_time

        cv2.putText(frame, f"FPS: {fps}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Antigravity Prompt (mental anchor)
        cv2.putText(frame, ANTIGRAVITY_PROMPT, (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 180, 0), 2)

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    tracker.close()
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
