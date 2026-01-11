import cv2
from vision.camera import Camera
from vision.hand_tracker import HandTracker
from gestures import GestureRecognizer, GestureStateMachine
import time
import numpy as np

# Phase 3: Pseudo-3D World
from render.camera3d import Camera3D
from render.zbuffer import ZBuffer
from render.pseudo3d import project_3d_to_2d
from world.voxel_grid import VoxelGrid
from world.voxel_ops import get_voxel_cube_vertices, sort_voxels_by_depth, draw_voxel
from world.voxel_editor import VoxelEditor
from vision.depth_mapper import extract_hand_depth, map_depth_to_world, visualize_depth

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

    # Phase 3: 3D World Setup
    camera_3d = Camera3D(position=(0, 0, -10), fov=60)  # Closer camera
    voxel_grid = VoxelGrid(create_sample=False)  # Start empty for editing
    voxel_editor = VoxelEditor(voxel_grid, camera_3d)  # Editor instance
    
    # PERFORMANCE: Disable Z-buffer for speed (rely on painter's algorithm only)
    use_zbuffer = False
    zbuffer = None
    
    # Camera rotation for floating effect (slower for less CPU)
    rotation_angle = 0.0
    auto_rotate = False  # Disable auto-rotate when editing

    window_name = "Ultron Workstation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    prev_time = 0

    while True:
        frame = cam.read()
        if frame is None:
            break
        
        h, w = frame.shape[:2]
        
        # PERFORMANCE: Z-buffer disabled for speed
        # if zbuffer is None:
        #     zbuffer = ZBuffer(w, h)

        all_landmarks, _ = tracker.process(frame)

        gestures = []
        hand_depth_world = 5.0  # Default depth
        
        if all_landmarks:
            # Use first hand for editing
            first_hand = all_landmarks[0]
            
            # Draw hands and recognize gestures
            for i, landmarks in enumerate(all_landmarks):
                draw_hand(frame, landmarks)
                
                # Recognize gesture
                gesture = recognizer.recognize_single_hand(landmarks)
                
                # Apply state machine for stability
                if i < len(state_machines):
                    stable_gesture = state_machines[i].update(gesture)
                    gestures.append(stable_gesture)
                
                # Extract hand depth (use first hand)
                if i == 0:
                    hand_depth_raw = extract_hand_depth(landmarks, method='palm')
                    hand_depth_world = map_depth_to_world(hand_depth_raw, min_depth=0, max_depth=10)
                    
                    # Visualize hand depth
                    visualize_depth(frame, landmarks, hand_depth_world)
                    
                    # Update editor mode based on gesture
                    voxel_editor.update_mode(stable_gesture)
                    
                    # Get index finger tip for cursor (landmark 8)
                    index_tip = landmarks[8]
                    voxel_editor.cursor_pos = voxel_editor.hand_to_world(
                        index_tip[0], index_tip[1], index_tip[2], w, h
                    )
                    
                    # Handle editing actions
                    if voxel_editor.mode == "DRAW":
                        # Place voxel at cursor
                        voxel_editor.place_voxel(voxel_editor.cursor_pos)
                    
                    elif voxel_editor.mode == "ERASE":
                        # Erase nearest voxel
                        target = voxel_editor.find_nearest_voxel(voxel_editor.cursor_pos)
                        if target:
                            voxel_editor.erase_voxel(target)
                    
                    elif voxel_editor.mode == "ROTATE":
                        # Manual rotation control
                        auto_rotate = False
                        palm_center = landmarks[0]  # Use wrist for rotation
                        voxel_editor.update_rotation(palm_center[0], palm_center[1])
                    
                    else:  # IDLE or HOLD
                        voxel_editor.reset_rotation()
                        auto_rotate = True  # Resume auto-rotate
            
            # Check for two-hand gestures
            if len(all_landmarks) == 2:
                two_hand = recognizer.recognize_two_hands(all_landmarks[0], all_landmarks[1])
                if two_hand:
                    gestures = [two_hand, two_hand]  # Override with two-hand gesture
                    # Two-hand open palms: cycle color
                    if two_hand == "ZOOM":
                        voxel_editor.cycle_color()
        else:
            # No hands: resume auto-rotate
            auto_rotate = True
            voxel_editor.reset_rotation()

        # ======== Phase 3: Render 3D Voxels ========
        # PERFORMANCE: No Z-buffer clear needed (disabled)
        # if zbuffer:
        #     zbuffer.clear()
        
        # Auto-rotation (when not manually controlling)
        if auto_rotate:
            rotation_angle += 0.005  # Reduced from 0.01
            camera_3d.set_rotation(0, rotation_angle, 0)

        
        # Get all voxels and sort by depth
        voxels = list(voxel_grid.get_all_voxels())
        sorted_voxels = sort_voxels_by_depth(voxels, camera_3d)
        
        # Render each voxel
        voxels_drawn = 0
        for pos, color in sorted_voxels:
            # Get voxel cube vertices (LARGER voxels for better visibility)
            vertices_3d = get_voxel_cube_vertices(pos, size=1.5)
            
            # Project to 2D
            vertices_2d = []
            for vertex in vertices_3d:
                projected = project_3d_to_2d(vertex, camera_3d, w, h)
                vertices_2d.append(projected)
            
            # Draw voxel WITHOUT Z-buffer (painter's algorithm only)
            if draw_voxel(frame, vertices_2d, color, zbuffer=None):
                voxels_drawn += 1
        
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
        
        # Voxel count and editor info
        voxel_info = f"Voxels: {voxels_drawn}/{voxel_grid.count()} (Max: {voxel_editor.max_voxels})"
        cv2.putText(frame, voxel_info, (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2, cv2.LINE_AA)
        
        # Edit mode display
        mode_text = f"Mode: {voxel_editor.mode}"
        mode_color = {
            "DRAW": (0, 255, 0),
            "ERASE": (0, 0, 255),
            "ROTATE": (255, 255, 0),
            "HOLD": (128, 128, 128),
            "IDLE": (200, 200, 200)
        }.get(voxel_editor.mode, (255, 255, 255))
        cv2.putText(frame, mode_text, (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2, cv2.LINE_AA)
        
        # Current color preview
        if voxel_editor.mode == "DRAW":
            color = voxel_editor.get_current_color()
            cv2.rectangle(frame, (20, 220), (60, 260), color, -1)
            cv2.rectangle(frame, (20, 220), (60, 260), (255, 255, 255), 2)
            cv2.putText(frame, "Color", (70, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Cursor position (3D coordinates)
        cursor_text = f"Cursor: {voxel_editor.cursor_pos}"
        cv2.putText(frame, cursor_text, (20, 290),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 255), 1)

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
