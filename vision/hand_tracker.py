import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import os
from utils.smoothing import EMASmoother

class HandTracker:
    def __init__(self, model_path="vision/hand_landmarker.task"):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        # 2 sets of smoothers, one for each hand slot
        self.smoothers = [[EMASmoother() for _ in range(21)] for _ in range(2)]

    def process(self, frame):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect hand landmarks
        detection_result = self.detector.detect(mp_image)

        all_landmarks = []
        if detection_result.hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
                if hand_idx >= 2: break # Safety cap
                
                smoothed_hand = []
                for i, lm in enumerate(hand_landmarks):
                    # Smooth the landmarks (x, y, z are normalized)
                    smoothed = self.smoothers[hand_idx][i].smooth(
                        (lm.x, lm.y, lm.z)
                    )
                    smoothed_hand.append(smoothed)
                all_landmarks.append(smoothed_hand)
                
            return all_landmarks, detection_result.hand_landmarks
        
        return [], None

    def close(self):
        self.detector.close()
