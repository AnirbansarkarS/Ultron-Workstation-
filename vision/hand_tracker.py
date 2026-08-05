import time
import cv2
import numpy as np
import threading
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
        self.last_timestamp = 0
        self.lock = threading.Lock()

        def result_callback(result, output_image, timestamp_ms):
            with self.lock:
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

        # -------- One Euro Filters --------
        self.filters = {
            "Left": [
                OneEuroFilter(min_cutoff=1.2, beta=0.02) for _ in range(21)
            ],
            "Right": [
                OneEuroFilter(min_cutoff=1.2, beta=0.02) for _ in range(21)
            ]
        }

    def process(self, frame):
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            # Monotonic real-time timestamp in milliseconds (required for LIVE_STREAM mode)
            now_ms = int(time.time() * 1000)
            if now_ms <= self.last_timestamp:
                now_ms = self.last_timestamp + 1
            self.last_timestamp = now_ms

            self.detector.detect_async(mp_image, self.last_timestamp)

            with self.lock:
                result = self.latest_result

            if result is None or not getattr(result, 'hand_landmarks', None):
                return [], None

            all_hands = []
            hand_landmarks_list = result.hand_landmarks
            handedness_list = getattr(result, 'handedness', [])

            for i, hand_landmarks in enumerate(hand_landmarks_list):
                # Safely extract handedness string ("Left" or "Right")
                handedness = "Right"
                if i < len(handedness_list) and handedness_list[i] and len(handedness_list[i]) > 0:
                    handedness = handedness_list[i][0].category_name

                # Fallback filter mapping if handedness string is unexpected
                if handedness not in self.filters:
                    handedness = "Right" if i == 0 else "Left"

                filter_set = self.filters[handedness]

                filtered_hand = []
                for j, lm in enumerate(hand_landmarks):
                    fx, fy, fz = filter_set[j].smooth(
                        (lm.x, lm.y, lm.z)
                    )
                    # Z-axis minimal filtering
                    fz = lm.z * 0.7 + fz * 0.3
                    filtered_hand.append((fx, fy, fz))

                all_hands.append(filtered_hand)

            return all_hands, hand_landmarks_list

        except Exception as e:
            print(f"[HandTracker] Error during frame processing: {e}")
            return [], None

    def close(self):
        try:
            self.detector.close()
        except Exception:
            pass

