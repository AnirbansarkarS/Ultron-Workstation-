"""Gesture state machine to prevent flickering."""

class GestureStateMachine:
    """
    Prevents gesture flickering using hysteresis.
    Gesture must be stable for N frames before switching.
    """
    
    def __init__(self, stability_frames=3):
        self.stability_frames = stability_frames
        self.current_gesture = "NONE"
        self.candidate_gesture = "NONE"
        self.candidate_count = 0
    
    def update(self, new_gesture):
        """
        Update state machine with new gesture detection.
        
        Args:
            new_gesture: Detected gesture name
            
        Returns:
            str: Stable gesture name (only changes after N frames)
        """
        if new_gesture == self.current_gesture:
            # Same gesture - reset candidate
            self.candidate_gesture = new_gesture
            self.candidate_count = 0
            return self.current_gesture
        
        if new_gesture == self.candidate_gesture:
            # Same candidate - increment counter
            self.candidate_count += 1
            
            if self.candidate_count >= self.stability_frames:
                # Candidate is stable - switch
                self.current_gesture = self.candidate_gesture
                self.candidate_count = 0
        else:
            # New candidate - reset
            self.candidate_gesture = new_gesture
            self.candidate_count = 1
        
        return self.current_gesture
    
    def reset(self):
        """Reset to initial state."""
        self.current_gesture = "NONE"
        self.candidate_gesture = "NONE"
        self.candidate_count = 0
