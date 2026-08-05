"""
Model Renderer — draws a ModelObject in an OpenCV frame.

Uses the existing project_3d_to_2d pipeline for consistent perspective
with the voxel world.  Two render paths:
  - Solid (face fill + wireframe edges)  — for models with ≤ FACE_THRESHOLD faces
  - Wireframe-only                        — for heavy meshes
"""

import cv2
import numpy as np
from render.pseudo3d import project_3d_to_2d


# ──────────────────────── batch projection ───────────────────────────

def project_vertices_batch(world_verts, camera, w, h):
    """
    Project all world-space vertices to screen space in one pass.

    Returns
    -------
    screen_pts : list of (sx, sy, depth) or None per vertex.
                 None means the vertex is behind the camera / clipped.
    """
    screen_pts = []
    for v in world_verts:
        result = project_3d_to_2d((float(v[0]), float(v[1]), float(v[2])),
                                  camera, w, h)
        screen_pts.append(result)   # None if clipped
    return screen_pts


# ────────────────────────── face rendering ───────────────────────────

def _draw_face(frame, p0, p1, p2, face_color, wire_color, face_alpha,
               overlay):
    """
    Draw a single filled + outlined triangle.
    Uses a pre-allocated overlay buffer to blend face fill.
    """
    pts = np.array([[int(p0[0]), int(p0[1])],
                    [int(p1[0]), int(p1[1])],
                    [int(p2[0]), int(p2[1])]], dtype=np.int32)

    # Fill on overlay, then blend
    cv2.fillPoly(overlay, [pts], face_color)

    # Wireframe on main frame
    cv2.polylines(frame, [pts], True, wire_color, 1, cv2.LINE_AA)


def _draw_edge(frame, p0, p1, wire_color):
    """Draw a single wireframe edge."""
    cv2.line(frame,
             (int(p0[0]), int(p0[1])),
             (int(p1[0]), int(p1[1])),
             wire_color, 1, cv2.LINE_AA)


# ────────────────────────── public entry point ───────────────────────

def render_model(frame, model_obj, camera_3d):
    """
    Render a ModelObject onto an OpenCV frame.

    Parameters
    ----------
    frame      : np.ndarray — BGR frame to draw onto (modified in place)
    model_obj  : ModelObject instance
    camera_3d  : Camera3D instance
    """
    if not model_obj.visible or model_obj.mesh is None:
        return

    h, w = frame.shape[:2]
    cam_pos = (camera_3d.position.x,
               camera_3d.position.y,
               camera_3d.position.z)

    # ── 1. Transform all vertices to world space (vectorised) ─────────
    world_verts = model_obj.get_world_vertices()   # np.ndarray (N, 3)

    # ── 2. Project all vertices to screen space ───────────────────────
    screen_pts = project_vertices_batch(world_verts, camera_3d, w, h)

    wire_color = model_obj.wire_color
    face_color = model_obj.face_color
    alpha      = model_obj.face_alpha

    # ── 3. Wireframe-only path ─────────────────────────────────────────
    if model_obj.wireframe_only:
        edges = model_obj.mesh.edges   # (K, 2)
        for ei, ej in edges:
            p0 = screen_pts[ei]
            p1 = screen_pts[ej]
            if p0 is not None and p1 is not None:
                _draw_edge(frame, p0, p1, wire_color)
        return

    # ── 4. Solid path: face fill + edges ─────────────────────────────
    # Sort faces back-to-front (Painter's algorithm)
    sorted_faces = model_obj.get_sorted_faces(world_verts, cam_pos)  # (M,3)

    # Overlay buffer for alpha-blended face fill
    overlay = frame.copy()

    drawn = 0
    for tri in sorted_faces:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        p0 = screen_pts[i0]
        p1 = screen_pts[i1]
        p2 = screen_pts[i2]

        # Skip if any vertex is clipped
        if p0 is None or p1 is None or p2 is None:
            continue

        _draw_face(frame, p0, p1, p2,
                   face_color, wire_color, alpha, overlay)
        drawn += 1

    # Blend the overlay (face fill) with the original frame
    if drawn > 0:
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

    return drawn


# ────────────────────── bounding box helper ──────────────────────────

def draw_model_bbox(frame, model_obj, camera_3d, color=(200, 200, 50)):
    """
    Draw the axis-aligned bounding box of the model in world space.
    Useful for debug / selection highlight.
    """
    h, w = frame.shape[:2]
    wv = model_obj.get_world_vertices()
    if len(wv) == 0:
        return

    mn = wv.min(axis=0)
    mx = wv.max(axis=0)

    # 8 corners of AABB
    corners = [
        (mn[0], mn[1], mn[2]), (mx[0], mn[1], mn[2]),
        (mx[0], mx[1], mn[2]), (mn[0], mx[1], mn[2]),
        (mn[0], mn[1], mx[2]), (mx[0], mn[1], mx[2]),
        (mx[0], mx[1], mx[2]), (mn[0], mx[1], mx[2]),
    ]
    edges_bb = [(0,1),(1,2),(2,3),(3,0),
                (4,5),(5,6),(6,7),(7,4),
                (0,4),(1,5),(2,6),(3,7)]

    pts2d = [project_3d_to_2d(c, camera_3d, w, h) for c in corners]

    for a, b in edges_bb:
        p0, p1 = pts2d[a], pts2d[b]
        if p0 is not None and p1 is not None:
            cv2.line(frame,
                     (int(p0[0]), int(p0[1])),
                     (int(p1[0]), int(p1[1])),
                     color, 1, cv2.LINE_AA)
