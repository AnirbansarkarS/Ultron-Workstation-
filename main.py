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
                    pt2 = (int(v2[0]), int(v2[1]))
                    cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)

def draw_frame_axes(frame, camera_3d, transform, w, h, length=2.0):
    """Draw X/Y/Z axes based on the transform."""
    # Origin
    origin_local = (0, 0, 0)
    tx, ty, tz, _ = transform.transform_point(origin_local)
    origin_2d = project_3d_to_2d((tx, ty, tz), camera_3d, w, h)
    
    if origin_2d is None: return

    # Axes endpoints
    axes = [
        ((length, 0, 0), (0, 0, 255)),   # X - Red (BGR) -> No, OpenCV is BGR, so (0,0,255) is Red
        ((0, length, 0), (0, 255, 0)),   # Y - Green
        ((0, 0, length), (255, 0, 0))    # Z - Blue
    ]
    
    for local_pt, color in axes:
        # Transform direction vector? No, transform point (origin + axis)
        pt_local = (local_pt[0], local_pt[1], local_pt[2])
        wx, wy, wz, _ = transform.transform_point(pt_local)
        
        pt_2d = project_3d_to_2d((wx, wy, wz), camera_3d, w, h)
        
        if pt_2d is not None:
            p1 = (int(origin_2d[0]), int(origin_2d[1]))
            p2 = (int(pt_2d[0]), int(pt_2d[1]))
            cv2.line(frame, p1, p2, color, 3, cv2.LINE_AA)

def main():
    cam = Camera()
    tracker = HandTracker()
    
    recognizer = GestureRecognizer()
    state_machines = [GestureStateMachine(stability_frames=2) for _ in range(2)]

    # 3D World Setup - STABLE CAMERA
    # Position at +15 Z looking at origin (0,0,0) so objects are in front
    camera_3d = Camera3D(position=(0, 0, 15), rotation=(0, 0, 0), fov=60)
    
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

    # Phase 6: UI Manager
    from ui.ui_manager import UIManager
    ui_manager = None

    # Debugging import
    import traceback

    while True:
        frame = cam.read()
        if frame is None:
            break
        
        h, w = frame.shape[:2]
        
        # Init UI Manager once we have dimensions
        if ui_manager is None:
            ui_manager = UIManager(w, h, voxel_editor.tool_manager)
        
        # DEBUG: Test projection on first frame
        if frame_count == 0:
            test_projected = project_3d_to_2d(test_point, camera_3d, w, h)
            print(f"Test point (screen): {test_projected}")
            print(f"Screen size: {w}x{h}")

        try:
            all_landmarks, _ = tracker.process(frame)
        except TypeError:
            print("HandTracker returned NoneType, skipping frame.")
            continue

        gestures = []
        show_cursor = False
        cursor_2d_pos = (0, 0) # For UI interaction
        cursor_color = (0, 255, 255)  # Default cyan cursor
        is_ui_interacting = False # Did UI capture input?
        
        if all_landmarks:
            for i, landmarks in enumerate(all_landmarks):
                draw_hand(frame, landmarks)
                
                gesture = recognizer.recognize_single_hand(landmarks)
                
                if i < len(state_machines):
                    stable_gesture = state_machines[i].update(gesture)
                    gestures.append(stable_gesture)
                
                if i == 0:
                    # --- calculate cursor position ---
                    thumb_tip = landmarks[4]
                    index_tip = landmarks[8]
                    mid_x = (thumb_tip[0] + index_tip[0]) / 2.0
                    mid_y = (thumb_tip[1] + index_tip[1]) / 2.0
                    mid_z = (thumb_tip[2] + index_tip[2]) / 2.0
                    
                    cursor_2d_x = int(mid_x * w)
                    cursor_2d_y = int(mid_y * h)
                    cursor_2d_pos = (cursor_2d_x, cursor_2d_y)
                    
                    # --- UI UPDATE ---
                    # Check "pinch" for click
                    is_pinching = (stable_gesture == "pinch")
                    is_ui_interacting = ui_manager.update(cursor_2d_x, cursor_2d_y, is_pinching)
                    
                    if not is_ui_interacting:
                        # Only update World Modes if UI didn't capture
                        voxel_editor.update_mode(stable_gesture)
                        voxel_editor.update_manipulation(all_landmarks, w, h)
                        
                        voxel_editor.cursor_pos = voxel_editor.hand_to_world(
                            mid_x, mid_y, mid_z, w, h
                        )
                        
                        show_cursor = True
                        
                        # Execute Tool Action (Only if not clicking UI)
                        result = voxel_editor.use_current_tool(stable_gesture)
                        if result:
                             print(f"Action: {result} | Total: {voxel_grid.count()}")
                        
                        # Handle Rotation
                        if voxel_editor.mode == "ROTATE_CAM":
                             cursor_color = (255, 255, 0)
                             palm_center = landmarks[0]
                             voxel_editor.update_rotation(palm_center[0], palm_center[1])
                        else:
                             voxel_editor.reset_rotation()

            if len(all_landmarks) == 2:
                two_hand = recognizer.recognize_two_hands(all_landmarks[0], all_landmarks[1])
                if two_hand:
                    gestures = [two_hand, two_hand]
                    if two_hand == "ZOOM" or two_hand == "SCALE_OBJECT":
                         if not is_ui_interacting:
                             voxel_editor.update_mode(two_hand)
                             voxel_editor.update_manipulation(all_landmarks, w, h)
        else:
            voxel_editor.reset_rotation()
            
        # Draw 2D Cursor for UI feedback even if no world cursor
        if is_ui_interacting:
             cv2.circle(frame, cursor_2d_pos, 5, (0, 0, 255), -1) 
        
        # Draw UI
        if ui_manager:
            ui_manager.draw(frame)
        
        # Draw Hand Cursor (World Context)
        if show_cursor and not is_ui_interacting:
            cx, cy = cursor_2d_pos
            cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (0, 255, 255), 2)
            cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (0, 255, 255), 2)
        else:
            voxel_editor.reset_rotation()

        # ======== Render 3D Voxels ========
        # 1. Get all voxels
        raw_voxels = list(voxel_grid.get_all_voxels())
        
        # 2. Transform centers for sorting
        transformed_voxels_for_sort = []
        for pos, color in raw_voxels:
            # Transform center
            tx, ty, tz, _ = voxel_grid.transform.transform_point(pos)
            transformed_voxels_for_sort.append(((tx, ty, tz), color, pos)) # Keep original pos for Vertex generation
            
        # 3. Sort based on transformed centers (Manual sort to preserve original pos)
        # sorted_pack = sort_voxels_by_depth(...) # REDUNDANT/REMOVED
        # Let's write a custom sort here or modify the input to sort_voxels.
        # sort_voxels_by_depth takes (pos, color). We gave it (transformed_pos, color).
        # It returns sorted (transformed_pos, color).
        # We lost the connection to 'original_pos' if we are not careful.
        # But wait! 'get_voxel_cube_vertices' generates vertices around the input pos.
        # If input pos is Transformed, the cube is axis-aligned at the new position.
        # We want the cube to be ROTATED.
        # So we MUST generate vertices from ORIGINAL pos, and then Transform Vertices.
        
        # HACK: We need to sort 'raw_voxels' but using 'transformed_pos' as key.
        def get_dist_sq(v_pack):
            t_pos = v_pack[0]
            cx, cy, cz = camera_3d.position.x, camera_3d.position.y, camera_3d.position.z
            return (t_pos[0]-cx)**2 + (t_pos[1]-cy)**2 + (t_pos[2]-cz)**2
            
        transformed_voxels_for_sort.sort(key=get_dist_sq, reverse=True)
        
        voxels_drawn = 0
        voxels_clipped = 0
        
        for t_pos, color, orig_pos in transformed_voxels_for_sort:
            # Generate vertices in LOCAL space
            vertices_local = get_voxel_cube_vertices(orig_pos, size=1.0)
            
            # Transform vertices to WORLD space
            vertices_world = []
            for v in vertices_local:
                tx, ty, tz, _ = voxel_grid.transform.transform_point(v)
                vertices_world.append((tx, ty, tz))
            
            vertices_2d = []
            valid_count = 0
            for vertex in vertices_world:
                projected = project_3d_to_2d(vertex, camera_3d, w, h)
                vertices_2d.append(projected)
                if projected is not None:
                    valid_count += 1
            
            if valid_count == 0:
                voxels_clipped += 1
            
            if draw_voxel(frame, vertices_2d, color, zbuffer=None):
                voxels_drawn += 1
        
        # Draw Object Gizmo (Axes)
        draw_frame_axes(frame, camera_3d, voxel_grid.transform, w, h, length=3.0)
        
        # Draw 3D cursor
        
        # Draw 3D cursor (Ensure it follows transform if needed, or visualizes properly)
        if show_cursor:
            # Cursor is in Grid Space (returned by hand_to_world)
            # We must transform it to World Space for drawing
            ctx, cty, ctz, _ = voxel_grid.transform.transform_point(voxel_editor.cursor_pos)
            draw_3d_cursor(frame, (ctx, cty, ctz), camera_3d, w, h, cursor_color, size=0.8)
        
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
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")