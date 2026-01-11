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
    points = [denormalize_point(lm, w, h) for lm in landmarks]

    for connection in HAND_CONNECTIONS:
        pt1 = points[connection[0]]
        pt2 = points[connection[1]]
        cv2.line(frame, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)

    for pt in points:
        cv2.circle(frame, pt, 4, (0, 0, 255), -1, cv2.LINE_AA)

def draw_3d_cursor(frame, cursor_pos, camera_3d, w, h, color=(0, 255, 255), size=0.8):
    """Draw visible 3D cursor at given position."""
    cursor_vertices_3d = get_voxel_cube_vertices(cursor_pos, size=size)
    cursor_vertices_2d = []
    
    for vertex in cursor_vertices_3d:
        projected = project_3d_to_2d(vertex, camera_3d, w, h)
        cursor_vertices_2d.append(projected)
    
    valid_verts = [v for v in cursor_vertices_2d if v is not None]
    
    if len(valid_verts) >= 4:
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        
        for i, j in edges:
            if i < len(cursor_vertices_2d) and j < len(cursor_vertices_2d):
                v1 = cursor_vertices_2d[i]
                v2 = cursor_vertices_2d[j]
                if v1 is not None and v2 is not None:
                    pt1 = (int(v1[0]), int(v1[1]))
                    pt2 = (int(v2[0]), int(v2[1]))
                    cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)

def main():
    cam = Camera()
    tracker = HandTracker()
    
    recognizer = GestureRecognizer()
    state_machines = [GestureStateMachine(stability_frames=2) for _ in range(2)]

    # 3D World Setup - STABLE CAMERA
    camera_3d = Camera3D(position=(0, 0, -15), rotation=(0, 0, 0), fov=60)
    
    voxel_grid = VoxelGrid(create_sample=True)
    voxel_editor = VoxelEditor(voxel_grid, camera_3d)
    
    zbuffer = None

    window_name = "Ultron Workstation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    prev_time = 0
    
    print("=== ULTRON WORKSTATION STARTED ===")
    print(f"Initial voxels: {voxel_grid.count()}")
    print(f"Camera position: {camera_3d.position}")
    print(f"Camera rotation: {camera_3d.rotation}")
    
    # DEBUG: Test projection with a known point
    test_point = (0, 0, 0)  # Origin
    print(f"\n=== PROJECTION TEST ===")
    print(f"Test point (world): {test_point}")

    frame_count = 0

    while True:
        frame = cam.read()
        if frame is None:
            break
        
        h, w = frame.shape[:2]
        
        # DEBUG: Test projection on first frame
        if frame_count == 0:
            test_projected = project_3d_to_2d(test_point, camera_3d, w, h)
            print(f"Test point (screen): {test_projected}")
            print(f"Screen size: {w}x{h}")

        all_landmarks, _ = tracker.process(frame)

        gestures = []
        show_cursor = False
        cursor_color = (0, 255, 255)
        
        if all_landmarks:
            for i, landmarks in enumerate(all_landmarks):
                draw_hand(frame, landmarks)
                
                gesture = recognizer.recognize_single_hand(landmarks)
                
                if i < len(state_machines):
                    stable_gesture = state_machines[i].update(gesture)
                    gestures.append(stable_gesture)
                
                if i == 0:
                    voxel_editor.update_mode(stable_gesture)
                    
                    thumb_tip = landmarks[4]
                    index_tip = landmarks[8]
                    
                    mid_x = (thumb_tip[0] + index_tip[0]) / 2.0
                    mid_y = (thumb_tip[1] + index_tip[1]) / 2.0
                    mid_z = (thumb_tip[2] + index_tip[2]) / 2.0
                    
                    voxel_editor.cursor_pos = voxel_editor.hand_to_world(
                        mid_x, mid_y, mid_z, w, h
                    )
                    
                    show_cursor = True
                    
                    cursor_2d_x = int(mid_x * w)
                    cursor_2d_y = int(mid_y * h)
                    
                    cv2.line(frame, (cursor_2d_x - 20, cursor_2d_y), (cursor_2d_x + 20, cursor_2d_y), (0, 255, 255), 2)
                    cv2.line(frame, (cursor_2d_x, cursor_2d_y - 20), (cursor_2d_x, cursor_2d_y + 20), (0, 255, 255), 2)
                    cv2.circle(frame, (cursor_2d_x, cursor_2d_y), 8, (0, 255, 255), 2)
                    
                    if voxel_editor.mode == "DRAW":
                        cursor_color = (0, 255, 0)
                        placed = voxel_editor.place_voxel(voxel_editor.cursor_pos)
                        if placed:
                            print(f"✓ Voxel placed at {voxel_editor.cursor_pos} | Total: {voxel_grid.count()}")
                    
                    elif voxel_editor.mode == "ERASE":
                        cursor_color = (0, 0, 255)
                        target = voxel_editor.find_nearest_voxel(voxel_editor.cursor_pos)
                        if target:
                            erased = voxel_editor.erase_voxel(target)
                            if erased:
                                print(f"✗ Voxel erased at {target} | Total: {voxel_grid.count()}")
                    
                    elif voxel_editor.mode == "ROTATE":
                        cursor_color = (255, 255, 0)
                        palm_center = landmarks[0]
                        voxel_editor.update_rotation(palm_center[0], palm_center[1])
                    
                    else:
                        voxel_editor.reset_rotation()
            
            if len(all_landmarks) == 2:
                two_hand = recognizer.recognize_two_hands(all_landmarks[0], all_landmarks[1])
                if two_hand:
                    gestures = [two_hand, two_hand]
                    if two_hand == "ZOOM":
                        voxel_editor.cycle_color()
                        print(f"Color cycled to {voxel_editor.get_current_color()}")
        else:
            voxel_editor.reset_rotation()

        # ======== Render 3D Voxels ========
        voxels = list(voxel_grid.get_all_voxels())
        sorted_voxels = sort_voxels_by_depth(voxels, camera_3d)
        
        voxels_drawn = 0
        voxels_clipped = 0
        
        # DEBUG: Print first voxel projection details
        if frame_count == 0 and len(voxels) > 0:
            first_voxel_pos, first_voxel_color = voxels[0]
            print(f"\n=== FIRST VOXEL DEBUG ===")
            print(f"Voxel position: {first_voxel_pos}")
            print(f"Voxel color: {first_voxel_color}")
            
            vertices_3d = get_voxel_cube_vertices(first_voxel_pos, size=1.0)
            print(f"First vertex (3D): {vertices_3d[0]}")
            
            projected_first = project_3d_to_2d(vertices_3d[0], camera_3d, w, h)
            print(f"First vertex (2D): {projected_first}")
        
        for pos, color in sorted_voxels:
            vertices_3d = get_voxel_cube_vertices(pos, size=1.0)
            
            vertices_2d = []
            valid_count = 0
            for vertex in vertices_3d:
                projected = project_3d_to_2d(vertex, camera_3d, w, h)
                vertices_2d.append(projected)
                if projected is not None:
                    valid_count += 1
            
            if valid_count == 0:
                voxels_clipped += 1
            
            if draw_voxel(frame, vertices_2d, color, zbuffer=None):
                voxels_drawn += 1
        
        # Draw 3D cursor
        if show_cursor:
            draw_3d_cursor(frame, voxel_editor.cursor_pos, camera_3d, w, h, cursor_color, size=0.8)
        
        # FPS counter
        curr_time = time.time()
        fps = int(1 / (curr_time - prev_time)) if prev_time else 0
        prev_time = curr_time

        cv2.putText(frame, f"FPS: {fps}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        gesture_text = " | ".join([f"Hand {i+1}: {g}" for i, g in enumerate(gestures)]) if gestures else "No hands detected"
        cv2.putText(frame, gesture_text, (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
        
        # ENHANCED DEBUG INFO
        voxel_info = f"Voxels: {voxels_drawn} drawn / {voxels_clipped} clipped / {voxel_grid.count()} total"
        cv2.putText(frame, voxel_info, (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2, cv2.LINE_AA)
        
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
        
        if voxel_editor.mode == "DRAW":
            color = voxel_editor.get_current_color()
            cv2.rectangle(frame, (20, 220), (60, 260), color, -1)
            cv2.rectangle(frame, (20, 220), (60, 260), (255, 255, 255), 2)
            cv2.putText(frame, "Color", (70, 245),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if show_cursor:
            cursor_text = f"3D Cursor: {voxel_editor.cursor_pos}"
            cv2.putText(frame, cursor_text, (20, 290),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Camera debug info
        cam_text = f"Cam: pos{camera_3d.position.to_tuple()} rot{camera_3d.rotation}"
        cv2.putText(frame, cam_text, (20, 330),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv2.putText(frame, ANTIGRAVITY_PROMPT, (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 180, 0), 2)

        cv2.imshow(window_name, frame)
        
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == 27:
            break

    tracker.close()
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()