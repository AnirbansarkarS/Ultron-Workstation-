import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import os
from utils.filters import OneEuroFilter

class HandTracker:
    def __init__(self, model_path="vision/hand_landmarker.task"):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.8,
            min_hand_presence_confidence=0.8,
            min_tracking_confidence=0.8
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # filters mapped to handedness to prevent index swapping jitter
        # Tuned parameters: min_cutoff=0.5 (less static jitter), beta=0.005 (responsive)
        self.filters = {
            "Left": [OneEuroFilter(min_cutoff=0.5, beta=0.005) for _ in range(21)],
            "Right": [OneEuroFilter(min_cutoff=0.5, beta=0.005) for _ in range(21)]
        }

    def process(self, frame):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect hand landmarks
        detection_result = self.detector.detect(mp_image)

        all_landmarks = []
        if detection_result.hand_landmarks:
            for i, hand_landmarks in enumerate(detection_result.hand_landmarks):
                # Get handedness label (Left or Right)
                handedness = detection_result.handedness[i][0].category_name
                
                if handedness in self.filters:
                    filtered_hand = []
                    for j, lm in enumerate(hand_landmarks):
                        # Use the persistent filter for this specific hand side
                        smoothed = self.filters[handedness][j].smooth(
                            (lm.x, lm.y, lm.z)
                        )
                        filtered_hand.append(smoothed)
                    all_landmarks.append(filtered_hand)
                
            return all_landmarks, detection_result.hand_landmarks
        
        return [], None

    def close(self):
        self.detector.close()
