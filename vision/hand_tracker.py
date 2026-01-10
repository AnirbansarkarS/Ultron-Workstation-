import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils.filters import OneEuroFilter


class HandTracker:
    def __init__(self, model_path="vision/hand_landmarker.task"):
        cv2.setUseOptimized(True)

        # -------- MediaPipe setup --------
        base_options = python.BaseOptions(model_asset_path=model_path)

        self.latest_result = None
        self.timestamp = 0

        def result_callback(result, output_image, timestamp_ms):
            self.latest_result = result

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            result_callback=result_callback,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.detector = vision.HandLandmarker.create_from_options(options)

        # -------- One Euro Filters (IRONMAN TUNING) --------
        self.filters = {
            "Left": [
                OneEuroFilter(min_cutoff=1.2, beta=0.02) for _ in range(21)
            ],
            "Right": [
                OneEuroFilter(min_cutoff=1.2, beta=0.02) for _ in range(21)
            ]
        }

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        self.timestamp += 1
        self.detector.detect_async(mp_image, self.timestamp)

        if self.latest_result is None:
            return [], None

        result = self.latest_result
        all_hands = []

        if not result.hand_landmarks:
            return [], None

        for i, hand_landmarks in enumerate(result.hand_landmarks):
            handedness = result.handedness[i][0].category_name

            if handedness not in self.filters:
                continue

            filtered_hand = []
            for j, lm in enumerate(hand_landmarks):
                fx, fy, fz = self.filters[handedness][j].smooth(
                    (lm.x, lm.y, lm.z)
                )

                # 🔥 Z-axis: minimal filtering (CRITICAL)
                fz = lm.z * 0.7 + fz * 0.3

                filtered_hand.append((fx, fy, fz))

            all_hands.append(filtered_hand)

        return all_hands, result.hand_landmarks

    def close(self):
        self.detector.close()
