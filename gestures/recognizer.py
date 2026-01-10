"""
Deterministic gesture recognizer.
Pure logic - no ML.
"""
from gestures.finger_state import (
    get_finger_states, 
    count_extended_fingers,
    distance_3d,
    THUMB_TIP,
    INDEX_TIP,
    MIDDLE_TIP,
    RING_TIP,
    PINKY_TIP
)

class GestureRecognizer:
    """Deterministic gesture recognition using geometric rules."""
    
    def __init__(self):
        self.pinch_threshold = 0.05  # Normalized distance for pinch
    
    def detect_open_palm(self, landmarks):
        """All 5 fingers extended."""
        extended = count_extended_fingers(landmarks)
        return extended >= 4  # Allow 4 or 5 for robustness
    
    def detect_fist(self, landmarks):
        """All fingers closed."""
        extended = count_extended_fingers(landmarks)
        return extended <= 1  # Allow thumb to be slightly open
    
    def detect_pinch(self, landmarks):
        """Thumb and index tips are close together."""
        thumb_tip = landmarks[THUMB_TIP]
        index_tip = landmarks[INDEX_TIP]
        
        dist = distance_3d(thumb_tip, index_tip)
        return dist < self.pinch_threshold
    
    def detect_index_point(self, landmarks):
        """Only index finger extended."""
        states = get_finger_states(landmarks)
        
        # Index must be extended
        if not states['INDEX']:
            return False
        
        # Other fingers should be closed
        others_closed = (
            not states['MIDDLE'] and 
            not states['RING'] and 
            not states['PINKY']
        )
        
        return others_closed
    
    def recognize_single_hand(self, landmarks):
        """
        Recognize gesture from single hand landmarks.
        Priority order prevents conflicts.
        
        Returns:
            str: Gesture name
        """
        if landmarks is None or len(landmarks) < 21:
            return "NONE"
        
        # Priority order (most specific first)
        if self.detect_pinch(landmarks):
            return "PINCH"
        
        if self.detect_index_point(landmarks):
            return "INDEX_POINT"
        
        if self.detect_fist(landmarks):
            return "FIST"
        
        if self.detect_open_palm(landmarks):
            return "OPEN_PALM"
        
        return "UNKNOWN"
    
    def recognize_two_hands(self, landmarks_left, landmarks_right):
        """
        Recognize two-hand gestures.
        
        Returns:
            str: Gesture name or None if no two-hand gesture detected
        """
        if landmarks_left is None or landmarks_right is None:
            return None
        
        # Both hands open and spread
        left_open = self.detect_open_palm(landmarks_left)
        right_open = self.detect_open_palm(landmarks_right)
        
        if left_open and right_open:
            return "TWO_HAND_SPREAD"
        
        return None
