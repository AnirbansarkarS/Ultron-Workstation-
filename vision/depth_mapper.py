"""
Hand depth extraction and mapping to world space.
"""
import numpy as np

def extract_hand_depth(landmarks, method='average'):
    """
    Extract depth value from hand landmarks.
    
    Args:
        landmarks: List of hand landmarks (x, y, z) tuples
        method: 'average', 'wrist', 'index_tip', 'palm'
    
    Returns:
        Depth value (Z coordinate)
    """
    if not landmarks:
        return 0.0
    
    if method == 'wrist':
        # Use wrist (landmark 0) depth
        return landmarks[0][2]
    
    elif method == 'index_tip':
        # Use index finger tip (landmark 8) depth
        if len(landmarks) > 8:
            return landmarks[8][2]
        return landmarks[0][2]
    
    elif method == 'palm':
        # Average of palm landmarks (0, 5, 9, 13, 17)
        palm_indices = [0, 5, 9, 13, 17]
        palm_depths = [landmarks[i][2] for i in palm_indices if i < len(landmarks)]
        return np.mean(palm_depths) if palm_depths else 0.0
    
    else:  # 'average'
        # Average of all landmarks
        depths = [lm[2] for lm in landmarks]
        return np.mean(depths)

def map_depth_to_world(depth_normalized, min_depth=0, max_depth=10, 
                       input_range=(-0.15, 0.05)):
    """
    Map MediaPipe normalized depth to world space depth.
    
    MediaPipe Z values are relative to wrist and typically range from
    -0.1 (hand far from camera) to +0.1 (hand close to camera).
    
    Args:
        depth_normalized: MediaPipe Z value
        min_depth: Minimum world depth (far)
        max_depth: Maximum world depth (close)
        input_range: Expected range of MediaPipe Z values (min, max)
    
    Returns:
        World space depth value
    """
    input_min, input_max = input_range
    
    # Clamp input
    depth_clamped = np.clip(depth_normalized, input_min, input_max)
    
    # Normalize to [0, 1]
    normalized = (depth_clamped - input_min) / (input_max - input_min)
    
    # Invert (MediaPipe negative Z = far, we want far = low depth)
    normalized = 1.0 - normalized
    
    # Map to world range
    world_depth = min_depth + normalized * (max_depth - min_depth)
    
    return world_depth

def visualize_depth(frame, landmarks, depth_world, screen_pos=None):
    """
    Draw depth visualization on frame.
    
    Args:
        frame: OpenCV image to draw on
        landmarks: Hand landmarks (for positioning)
        depth_world: World space depth value
        screen_pos: Optional screen position for text (x, y)
    
    Returns:
        Modified frame
    """
    import cv2
    
    if not landmarks:
        return frame
    
    # Get wrist position for visualization anchor
    h, w, _ = frame.shape
    from vision.landmark_utils import denormalize_point
    wrist_screen = denormalize_point(landmarks[0], w, h)
    
    if screen_pos is None:
        screen_pos = (wrist_screen[0] + 50, wrist_screen[1] - 30)
    
    # Draw depth text
    depth_text = f"Depth: {depth_world:.2f}"
    cv2.putText(frame, depth_text, screen_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    
    # Draw depth bar (visual indicator)
    bar_x = screen_pos[0]
    bar_y = screen_pos[1] + 30
    bar_width = 100
    bar_height = 10
    
    # Background bar
    cv2.rectangle(frame, (bar_x, bar_y), 
                  (bar_x + bar_width, bar_y + bar_height),
                  (100, 100, 100), -1)
    
    # Filled portion (0-10 range)
    fill_width = int((depth_world / 10.0) * bar_width)
    fill_width = max(0, min(bar_width, fill_width))
    
    # Color gradient (close = red, far = blue)
    color_ratio = depth_world / 10.0
    color = (
        int(255 * (1 - color_ratio)),  # B
        int(128 * color_ratio),         # G
        int(255 * color_ratio)          # R
    )
    
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + fill_width, bar_y + bar_height),
                  color, -1)
    
    # Border
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_width, bar_y + bar_height),
                  (255, 255, 255), 1)
    
    return frame

def get_depth_color(depth_world, min_depth=0, max_depth=10):
    """
    Get color based on depth for visualization.
    
    Args:
        depth_world: World depth value
        min_depth: Minimum depth
        max_depth: Maximum depth
    
    Returns:
        (B, G, R) color tuple
    """
    # Normalize depth to [0, 1]
    normalized = (depth_world - min_depth) / (max_depth - min_depth)
    normalized = np.clip(normalized, 0, 1)
    
    # Color gradient: blue (far) -> cyan -> green -> yellow -> red (close)
    if normalized < 0.25:
        # Blue to Cyan
        t = normalized / 0.25
        return (int(255 * (1 - t)), int(255 * t), 0)
    elif normalized < 0.5:
        # Cyan to Green
        t = (normalized - 0.25) / 0.25
        return (int(255 * (1 - t)), 255, 0)
    elif normalized < 0.75:
        # Green to Yellow
        t = (normalized - 0.5) / 0.25
        return (0, 255, int(255 * t))
    else:
        # Yellow to Red
        t = (normalized - 0.75) / 0.25
        return (0, int(255 * (1 - t)), 255)
