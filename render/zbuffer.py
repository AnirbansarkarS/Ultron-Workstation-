"""
Simple Z-buffer for depth testing.
"""
import numpy as np

class ZBuffer:
    def __init__(self, width, height):
        """
        Initialize Z-buffer with given dimensions.
        
        Args:
            width: Screen width in pixels
            height: Screen height in pixels
        """
        self.width = width
        self.height = height
        self.clear()
    
    def clear(self):
        """Reset Z-buffer to infinity (far plane)."""
        self.buffer = np.full((self.height, self.width), np.inf, dtype=np.float32)
    
    def test_and_set(self, x, y, depth):
        """
        Test if pixel should be drawn and update depth if closer.
        
        Args:
            x: Screen X coordinate
            y: Screen Y coordinate
            depth: Depth value (smaller = closer)
        
        Returns:
            True if pixel should be drawn (passed depth test), False otherwise
        """
        # Bounds checking
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        
        # Depth test
        if depth < self.buffer[y, x]:
            self.buffer[y, x] = depth
            return True
        
        return False
    
    def get_depth(self, x, y):
        """Get current depth at pixel (for debugging)."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.buffer[y, x]
        return np.inf
    
    def resize(self, width, height):
        """Resize buffer (e.g., if window changes)."""
        self.width = width
        self.height = height
        self.clear()
