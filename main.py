import cv2
import sys
import argparse
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

# ── 3D Model import system ─────────────────────────────────────────────
from world.model_loader import smart_load
from world.model_object import ModelObject
from world.model_controller import ModelController
from render.model_renderer import render_model, draw_model_bbox

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


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Ultron Workstation")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to a 3D model file (.obj, .stl, .glb, .gltf)")
    return parser.parse_args()


def open_file_dialog():
    """Open a Tkinter file-picker and return the selected path (or None)."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        path = filedialog.askopenfilename(
            title="Import 3D Model",
            filetypes=[
                ("3D Models", "*.obj *.stl *.glb *.gltf *.ply"),
                ("OBJ files",  "*.obj"),
                ("STL files",  "*.stl"),
                ("GLB/GLTF",  "*.glb *.gltf"),
                ("PLY files",  "*.ply"),
                ("All files",  "*.*"),
            ]
        )
        root.destroy()
        return path if path else None
    except Exception as e:
        print(f"[FileDialog] Error: {e}")
        return None


def try_load_model(path):
    """Load a model file and return (ModelObject, ModelController) or (None, None)."""
    if not path:
        return None, None
    try:
        mesh       = smart_load(path)
        model_obj  = ModelObject(mesh)
        controller = ModelController(model_obj)
        print(f"[Main] Model imported: {mesh}")
        return model_obj, controller
    except Exception as e:
        print(f"[Main] Failed to load model '{path}': {e}")
        return None, None


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


def draw_frame_axes(frame, camera_3d, transform, w, h, length=2.0):
    """Draw X/Y/Z axes based on the transform."""
    origin_local = (0, 0, 0)
    tx, ty, tz, _ = transform.transform_point(origin_local)
    origin_2d = project_3d_to_2d((tx, ty, tz), camera_3d, w, h)

    if origin_2d is None:
        return

    axes = [
        ((length, 0, 0), (0, 0, 255)),   # X - Red
        ((0, length, 0), (0, 255, 0)),   # Y - Green
        ((0, 0, length), (255, 0, 0))    # Z - Blue
    ]

    for local_pt, color in axes:
        wx, wy, wz, _ = transform.transform_point(local_pt)
        pt_2d = project_3d_to_2d((wx, wy, wz), camera_3d, w, h)

        if pt_2d is not None:
            p1 = (int(origin_2d[0]), int(origin_2d[1]))
            p2 = (int(pt_2d[0]), int(pt_2d[1]))
            cv2.line(frame, p1, p2, color, 3, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════
#  HUD drawing
# ═══════════════════════════════════════════════════════════════════════

def draw_hud(frame, fps, gestures, voxel_editor, voxel_grid,
             voxels_drawn, voxels_clipped,
             show_cursor, camera_3d,
             workspace_mode, model_obj, model_controller):
    """Draw all HUD text and indicators."""
    h, w = frame.shape[:2]

    # ── FPS ──
    cv2.putText(frame, f"FPS: {fps}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # ── Gesture text ──
    gesture_text = (" | ".join([f"H{i+1}: {g}" for i, g in enumerate(gestures)])
                    if gestures else "No hands detected")
    cv2.putText(frame, gesture_text, (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

    # ── Workspace mode badge ─────────────────────────────────────────
    mode_badge_color = (0, 200, 255) if workspace_mode == "VOXEL" else (200, 80, 255)
    cv2.rectangle(frame, (20, 130), (220, 165), mode_badge_color, -1)
    cv2.rectangle(frame, (20, 130), (220, 165), (255, 255, 255), 1)
    cv2.putText(frame, f"TARGET: {workspace_mode}", (28, 156),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    # ── Voxel info (always shown) ─────────────────────────────────────
    voxel_info = (f"Voxels: {voxels_drawn} drawn / "
                  f"{voxels_clipped} clipped / {voxel_grid.count()} total")
    cv2.putText(frame, voxel_info, (20, 195),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2, cv2.LINE_AA)

    # ── Active-mode pill (VOXEL mode) ────────────────────────────────
    if workspace_mode == "VOXEL":
        mode_text  = f"Mode: {voxel_editor.mode}"
        mode_color = {
            "DRAW": (0, 255, 0), "ERASE": (0, 0, 255),
            "ROTATE": (255, 255, 0), "HOLD": (128, 128, 128),
            "IDLE": (200, 200, 200)
        }.get(voxel_editor.mode, (255, 255, 255))
        cv2.putText(frame, mode_text, (20, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2, cv2.LINE_AA)

        if voxel_editor.mode == "DRAW":
            col = voxel_editor.get_current_color()
            cv2.rectangle(frame, (20, 245), (55, 280), col, -1)
            cv2.rectangle(frame, (20, 245), (55, 280), (255, 255, 255), 2)
            cv2.putText(frame, "Color", (62, 268),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # ── Active-mode pill (MODEL mode) ────────────────────────────────
    if workspace_mode == "MODEL" and model_obj is not None:
        ctrl_mode  = model_controller.mode if model_controller else "N/A"
        ctrl_color = {
            "MODEL_ROTATE": (255, 100, 255),
            "MODEL_GRAB":   (100, 255, 200),
            "MODEL_SCALE":  (255, 200, 100),
            "HOLD":         (128, 128, 128),
        }.get(ctrl_mode, (200, 200, 200))
        cv2.putText(frame, f"Model: {ctrl_mode}", (20, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, ctrl_color, 2, cv2.LINE_AA)

        # Model info
        name_txt = (f"File: {model_obj.mesh.name}  "
                    f"({model_obj.mesh.face_count} faces)")
        cv2.putText(frame, name_txt, (20, 265),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 180, 255), 1, cv2.LINE_AA)

        wf_badge = "WIREFRAME" if model_obj.wireframe_only else "SOLID"
        cv2.putText(frame, wf_badge, (20, 288),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 150), 1)

    # ── MODEL mode: no model loaded notice ───────────────────────────
    if workspace_mode == "MODEL" and model_obj is None:
        cv2.putText(frame,
                    "No model loaded — press [O] to import",
                    (20, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 255), 2, cv2.LINE_AA)

    # ── Key hints (bottom-left) ───────────────────────────────────────
    hints = [
        "[M] Toggle VOXEL / MODEL",
        "[O] Import 3D Model",
        "[R] Reset Model Transform",
        "[ESC] Quit",
    ]
    for idx, hint in enumerate(hints):
        cv2.putText(frame, hint, (20, h - 100 + idx * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

    # ── Camera debug ──────────────────────────────────────────────────
    cam_txt = f"Cam: pos{camera_3d.position.to_tuple()} rot{camera_3d.rotation}"
    cv2.putText(frame, cam_txt, (20, h - 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1)

    # ── Watermark ─────────────────────────────────────────────────────
    cv2.putText(frame, ANTIGRAVITY_PROMPT,
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 180, 0), 2)


# ═══════════════════════════════════════════════════════════════════════
#  Main loop
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    cam     = Camera()
    tracker = HandTracker()

    recognizer     = GestureRecognizer()
    state_machines = [GestureStateMachine(stability_frames=2) for _ in range(2)]

    # 3D World Setup
    camera_3d   = Camera3D(position=(0, 0, 15), rotation=(0, 0, 0), fov=60)
    voxel_grid  = VoxelGrid(create_sample=True)
    voxel_editor = VoxelEditor(voxel_grid, camera_3d)
    zbuffer     = None

    # ── Model import ────────────────────────────────────────────────
    model_obj, model_controller = try_load_model(args.model)

    # ── Workspace mode: "VOXEL" | "MODEL" ────────────────────────────
    workspace_mode = "VOXEL"  # Start in voxel mode

    window_name = "Ultron Workstation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    prev_time  = 0
    frame_count = 0

    print("=== ULTRON WORKSTATION STARTED ===")
    print(f"Initial voxels : {voxel_grid.count()}")
    print(f"Camera position: {camera_3d.position}")
    print(f"Camera rotation: {camera_3d.rotation}")
    print("Keys: [M] toggle mode | [O] import model | [R] reset model | [ESC] quit")

    while True:
        frame = cam.read()
        if frame is None:
            break

        h, w = frame.shape[:2]

        # ── Keyboard input ────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == 27:   # ESC — quit
            break

        elif key == ord('m') or key == ord('M'):   # Toggle workspace mode
            if workspace_mode == "VOXEL":
                workspace_mode = "MODEL"
                print("[Main] Switched to MODEL mode")
            else:
                workspace_mode = "VOXEL"
                print("[Main] Switched to VOXEL mode")

        elif key == ord('o') or key == ord('O'):   # Open file dialog
            print("[Main] Opening model file picker...")
            path = open_file_dialog()
            if path:
                mo, mc = try_load_model(path)
                if mo is not None:
                    model_obj        = mo
                    model_controller = mc
                    workspace_mode   = "MODEL"  # Auto-switch to model mode
                    print(f"[Main] Auto-switched to MODEL mode for '{path}'")

        elif key == ord('r') or key == ord('R'):   # Reset model transform
            if model_obj is not None:
                model_obj.reset_transform()
                print("[Main] Model transform reset.")

        # ── Hand tracking ─────────────────────────────────────────────
        gestures    = []
        show_cursor = False
        cursor_color = (0, 255, 255)

        try:
            all_landmarks, _ = tracker.process(frame)

            if all_landmarks:
                for i, landmarks in enumerate(all_landmarks):
                    draw_hand(frame, landmarks)

                    gesture = recognizer.recognize_single_hand(landmarks)
                    stable_gesture = gesture

                    if i < len(state_machines):
                        stable_gesture = state_machines[i].update(gesture)
                        gestures.append(stable_gesture)

                    if i == 0:
                        # ── VOXEL mode: original behaviour ────────────────
                        if workspace_mode == "VOXEL":
                            voxel_editor.update_mode(stable_gesture)
                            voxel_editor.update_manipulation(all_landmarks, w, h)

                            thumb_tip  = landmarks[4]
                            index_tip  = landmarks[8]

                            mid_x = (thumb_tip[0] + index_tip[0]) / 2.0
                            mid_y = (thumb_tip[1] + index_tip[1]) / 2.0
                            mid_z = (thumb_tip[2] + index_tip[2]) / 2.0

                            voxel_editor.cursor_pos = voxel_editor.hand_to_world(
                                mid_x, mid_y, mid_z, w, h
                            )

                            show_cursor = True
                            cursor_2d_x = int(mid_x * w)
                            cursor_2d_y = int(mid_y * h)

                            cv2.line(frame, (cursor_2d_x - 20, cursor_2d_y),
                                     (cursor_2d_x + 20, cursor_2d_y), (0, 255, 255), 2)
                            cv2.line(frame, (cursor_2d_x, cursor_2d_y - 20),
                                     (cursor_2d_x, cursor_2d_y + 20), (0, 255, 255), 2)
                            cv2.circle(frame, (cursor_2d_x, cursor_2d_y), 8,
                                       (0, 255, 255), 2)

                            if voxel_editor.mode == "DRAW":
                                cursor_color = (0, 255, 0)
                                placed = voxel_editor.place_voxel(voxel_editor.cursor_pos)
                                if placed:
                                    print(f"✓ Voxel at {voxel_editor.cursor_pos} | "
                                          f"Total: {voxel_grid.count()}")

                            elif voxel_editor.mode == "ERASE":
                                cursor_color = (0, 0, 255)
                                target = voxel_editor.find_nearest_voxel(
                                    voxel_editor.cursor_pos)
                                if target:
                                    erased = voxel_editor.erase_voxel(target)
                                    if erased:
                                        print(f"✗ Voxel erased at {target} | "
                                              f"Total: {voxel_grid.count()}")

                            elif voxel_editor.mode == "ROTATE_CAM":
                                cursor_color = (255, 255, 0)
                                palm_center = landmarks[0]
                                voxel_editor.update_rotation(palm_center[0],
                                                             palm_center[1])
                            else:
                                voxel_editor.reset_rotation()

                        # ── MODEL mode: gesture → model transform ─────────
                        elif workspace_mode == "MODEL" and model_controller is not None:
                            model_controller.process(stable_gesture, all_landmarks, w, h)

                # Two-hand gestures (always processed)
                if len(all_landmarks) == 2:
                    two_hand = recognizer.recognize_two_hands(
                        all_landmarks[0], all_landmarks[1])
                    if two_hand:
                        gestures = [two_hand, two_hand]
                        if workspace_mode == "MODEL" and model_controller is not None:
                            model_controller.update_mode(two_hand)
                            model_controller.update_scale(all_landmarks)

            else:
                voxel_editor.reset_rotation()
                if model_controller is not None:
                    model_controller.reset_rotation()
                    model_controller.reset_grab()

        except Exception as err:
            print(f"[Main] Exception in hand processing loop: {err}")
            gestures = ["ERROR"]

        # ══════════════════════════════════════════════════════════════
        #  Render 3D Voxels
        # ══════════════════════════════════════════════════════════════
        raw_voxels = list(voxel_grid.get_all_voxels())

        transformed_voxels_for_sort = []
        for pos, color in raw_voxels:
            tx, ty, tz, _ = voxel_grid.transform.transform_point(pos)
            transformed_voxels_for_sort.append(((tx, ty, tz), color, pos))

        def get_dist_sq(v_pack):
            t_pos = v_pack[0]
            cx, cy, cz = (camera_3d.position.x,
                          camera_3d.position.y,
                          camera_3d.position.z)
            return (t_pos[0]-cx)**2 + (t_pos[1]-cy)**2 + (t_pos[2]-cz)**2

        transformed_voxels_for_sort.sort(key=get_dist_sq, reverse=True)

        voxels_drawn   = 0
        voxels_clipped = 0

        for t_pos, color, orig_pos in transformed_voxels_for_sort:
            vertices_local = get_voxel_cube_vertices(orig_pos, size=1.0)

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

        # Draw 3D cursor (VOXEL mode)
        if show_cursor and workspace_mode == "VOXEL":
            ctx, cty, ctz, _ = voxel_grid.transform.transform_point(
                voxel_editor.cursor_pos)
            draw_3d_cursor(frame, (ctx, cty, ctz), camera_3d, w, h,
                           cursor_color, size=0.8)

        # ══════════════════════════════════════════════════════════════
        #  Render Imported 3D Model
        # ══════════════════════════════════════════════════════════════
        if model_obj is not None and model_obj.visible:
            render_model(frame, model_obj, camera_3d)

            # Draw bounding box highlight when MODEL mode is active
            if workspace_mode == "MODEL":
                bbox_color = (200, 80, 255)   # Purple in BGR
                draw_model_bbox(frame, model_obj, camera_3d, bbox_color)

        # ══════════════════════════════════════════════════════════════
        #  HUD
        # ══════════════════════════════════════════════════════════════
        curr_time = time.time()
        fps = int(1 / (curr_time - prev_time)) if prev_time else 0
        prev_time = curr_time

        draw_hud(frame, fps, gestures,
                 voxel_editor, voxel_grid, voxels_drawn, voxels_clipped,
                 show_cursor, camera_3d,
                 workspace_mode, model_obj, model_controller)

        cv2.imshow(window_name, frame)
        frame_count += 1

    tracker.close()
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()