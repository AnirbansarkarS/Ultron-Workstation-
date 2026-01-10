import cv2
from vision.camera import Camera
from vision.hand_tracker import HandTracker
from gestures import GestureRecognizer, GestureStateMachine
import time
import numpy as np

# -------- ANTIGRAVITY PROMPT --------
ANTIGRAVITY_PROMPT = "ULTRON"

from vision.landmark_utils import denormalize_point

# Hand connections for drawing skeleton manually
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),             # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),             # Index
    (5, 9), (9, 10), (10, 11), (11, 12),        # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),      # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
]

def draw_hand(frame, landmarks):
    h, w, _ = frame.shape
    # Convert normalized to pixel coordinates using utility
    points = [denormalize_point(lm, w, h) for lm in landmarks]

    # Draw connections with slightly transparent effect (simulated with thin lines)
    for connection in HAND_CONNECTIONS:
        pt1 = points[connection[0]]
        pt2 = points[connection[1]]
        cv2.line(frame, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)

    # Draw landmarks as clean circles
    for pt in points:
        cv2.circle(frame, pt, 4, (0, 0, 255), -1, cv2.LINE_AA)

def main():
    cam = Camera()
    tracker = HandTracker()
    
    # Gesture recognition setup
    recognizer = GestureRecognizer()
    state_machines = [GestureStateMachine(stability_frames=2) for _ in range(2)]

    window_name = "Ultron Workstation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    prev_time = 0

    while True:
        frame = cam.read()
        if frame is None:
            break

        all_landmarks, _ = tracker.process(frame)

        gestures = []
        
        if all_landmarks:
            # Draw hands and recognize gestures
            for i, landmarks in enumerate(all_landmarks):
                draw_hand(frame, landmarks)
                
                # Recognize gesture
                gesture = recognizer.recognize_single_hand(landmarks)
                
                # Apply state machine for stability
                if i < len(state_machines):
                    stable_gesture = state_machines[i].update(gesture)
                    gestures.append(stable_gesture)
            
            # Check for two-hand gestures
            if len(all_landmarks) == 2:
                two_hand = recognizer.recognize_two_hands(all_landmarks[0], all_landmarks[1])
                if two_hand:
                    gestures = [two_hand, two_hand]  # Override with two-hand gesture

        # FPS counter
        curr_time = time.time()
        fps = int(1 / (curr_time - prev_time)) if prev_time else 0
        prev_time = curr_time

        cv2.putText(frame, f"FPS: {fps}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display gestures
        gesture_text = " | ".join([f"Hand {i+1}: {g}" for i, g in enumerate(gestures)]) if gestures else "No hands detected"
        cv2.putText(frame, gesture_text, (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

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
