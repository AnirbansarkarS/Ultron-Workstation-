"""
ModelObject — wraps a loaded ModelMesh with a Matrix4 transform.
Mirrors VoxelGrid's interface so existing gesture code works identically.
"""

import math
import numpy as np
from math3d.matrix import Matrix4


class ModelObject:
    """
    A loaded 3D mesh with an associated spatial transform.

    Attributes
    ----------
    mesh          : ModelMesh — geometry data (read-only after load)
    transform     : Matrix4  — current model-to-world transform
    visible       : bool     — whether to render
    wire_color    : (B, G, R) — OpenCV BGR colour for wireframe edges
    face_color    : (B, G, R) — OpenCV BGR colour for filled faces
    face_alpha    : float [0..1] — face fill opacity
    wireframe_only: bool     — True → skip face fill (perf mode)
    """

    # Face limit above which we force wireframe-only rendering
    WIREFRAME_THRESHOLD = 5000

    def __init__(self, mesh,
                 wire_color=(0, 255, 180),
                 face_color=(0, 120, 80),
                 face_alpha=0.45):
        self.mesh           = mesh
        self.transform      = Matrix4.identity()
        self.visible        = True
        self.wire_color     = wire_color      # BGR
        self.face_color     = face_color      # BGR
        self.face_alpha     = face_alpha
        self.wireframe_only = mesh.face_count > self.WIREFRAME_THRESHOLD

        if self.wireframe_only:
            print(f"[ModelObject] '{mesh.name}' has {mesh.face_count} faces "
                  f"(>{self.WIREFRAME_THRESHOLD}) → wireframe-only mode.")

    # ─────────────────────── transform helpers ────────────────────────

    def translate(self, x, y, z):
        """Apply a translation (additive) to current transform."""
        t = Matrix4.from_translation(x, y, z)
        self.transform = t.multiply(self.transform)

    def rotate(self, axis: str, angle: float):
        """
        Apply rotation around 'x', 'y', or 'z' axis.
        angle is in radians.
        """
        axis = axis.lower()
        if axis == 'x':
            r = Matrix4.from_rotation_x(angle)
        elif axis == 'y':
            r = Matrix4.from_rotation_y(angle)
        elif axis == 'z':
            r = Matrix4.from_rotation_z(angle)
        else:
            return
        self.transform = r.multiply(self.transform)

    def scale(self, factor: float):
        """Apply uniform scaling."""
        s = Matrix4.from_scale(factor, factor, factor)
        self.transform = s.multiply(self.transform)

    def reset_transform(self):
        """Reset to identity (origin, no rotation, scale=1)."""
        self.transform = Matrix4.identity()

    # ─────────────────────── world-space vertices ────────────────────

    def get_world_vertices(self):
        """
        Apply the current transform to all mesh vertices.

        Returns
        -------
        np.ndarray shape (N, 3) float32 — vertices in world space.
        Fast vectorised path using numpy.
        """
        verts  = self.mesh.vertices         # (N, 3)
        m      = self.transform.data        # 4×4 list-of-lists

        # Build numpy matrix for speed
        M = np.array(m, dtype=np.float32)  # (4, 4)

        # Homogeneous coords: add w=1 column → (N, 4)
        ones    = np.ones((len(verts), 1), dtype=np.float32)
        verts_h = np.hstack([verts, ones])  # (N, 4)

        # Transform: (N,4) @ (4,4)^T  → (N, 4)
        transformed = verts_h @ M.T

        # Return only xyz
        return transformed[:, :3]

    # ─────────────────────── face / edge depth sort ──────────────────

    def get_sorted_faces(self, world_verts, camera_pos):
        """
        Return face indices sorted farthest-first (Painter's algorithm).

        Parameters
        ----------
        world_verts : np.ndarray (N,3) — already-transformed vertices
        camera_pos  : (cx, cy, cz)

        Returns
        -------
        np.ndarray (M,3) int — face index array sorted back-to-front
        """
        cx, cy, cz = camera_pos
        faces = self.mesh.faces              # (M,3)

        # Centroid of each face
        f_verts = world_verts[faces]         # (M, 3, 3)
        centroids = f_verts.mean(axis=1)     # (M, 3)

        # Squared distance to camera
        diff      = centroids - np.array([cx, cy, cz], dtype=np.float32)
        dist_sq   = (diff * diff).sum(axis=1)  # (M,)

        # Sort descending (farthest first)
        order = np.argsort(dist_sq)[::-1]
        return faces[order]

    def __repr__(self):
        return (f"ModelObject('{self.mesh.name}', "
                f"visible={self.visible}, "
                f"wireframe_only={self.wireframe_only})")
