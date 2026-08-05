"""
Model Loader — Load 3D mesh files into Ultron Workstation.

Supports: .obj, .stl, .glb, .gltf
Uses trimesh for parsing; normalizes the model to fit the workspace.
"""

import os
import numpy as np

# ────────────────────────────── dataclass ──────────────────────────────

class ModelMesh:
    """Loaded 3D mesh, normalized to workspace coordinate space."""

    def __init__(self, vertices, faces, edges, name, file_path):
        """
        Args:
            vertices : np.ndarray shape (N, 3) float32 — world-space positions
            faces    : np.ndarray shape (M, 3) int     — triangle indices
            edges    : np.ndarray shape (K, 2) int     — unique edge pairs
            name     : str  — display name (file basename)
            file_path: str  — original file path
        """
        self.vertices  = vertices
        self.faces     = faces
        self.edges     = edges
        self.name      = name
        self.file_path = file_path

    @property
    def face_count(self):
        return len(self.faces)

    @property
    def vertex_count(self):
        return len(self.vertices)

    def __repr__(self):
        return (f"ModelMesh('{self.name}', "
                f"{self.vertex_count} verts, {self.face_count} faces)")


# ────────────────────────── loader function ────────────────────────────

def load_model(file_path: str, target_size: float = 8.0) -> ModelMesh:
    """
    Load a 3D model file and normalize it to fit inside the workspace.

    Args:
        file_path   : Absolute or relative path to model file.
        target_size : Diagonal extent of the normalized model (default 8 units).

    Returns:
        ModelMesh instance ready for rendering.

    Raises:
        FileNotFoundError : if the file doesn't exist.
        ImportError       : if trimesh is not installed.
        ValueError        : if the file can't be parsed as a mesh.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")

    try:
        import trimesh
    except ImportError:
        raise ImportError(
            "trimesh is required to load 3D models.\n"
            "Install it with:  pip install trimesh"
        )

    ext = os.path.splitext(file_path)[1].lower()

    # ── Load with trimesh ──────────────────────────────────────────────
    try:
        # force='mesh' to merge scene graphs into one mesh
        loaded = trimesh.load(file_path, force='mesh')
    except Exception as e:
        raise ValueError(f"Failed to load '{file_path}': {e}")

    if loaded is None or not hasattr(loaded, 'vertices'):
        raise ValueError(f"Could not extract a mesh from '{file_path}'")

    vertices = np.array(loaded.vertices, dtype=np.float32)   # (N, 3)
    faces    = np.array(loaded.faces,    dtype=np.int32)      # (M, 3)

    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError(f"Mesh in '{file_path}' has no geometry.")

    # ── Normalize: center + scale ──────────────────────────────────────
    centroid = vertices.mean(axis=0)
    vertices -= centroid                          # center at origin

    # Scale so the bounding diagonal == target_size
    extents  = vertices.max(axis=0) - vertices.min(axis=0)
    diagonal = float(np.linalg.norm(extents))
    if diagonal > 1e-6:
        vertices *= (target_size / diagonal)

    # ── Build unique edge list (wireframe) ────────────────────────────
    # Each triangle contributes 3 edges; deduplicate by sorting each pair
    edge_set = set()
    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        edge_set.add((min(a, b), max(a, b)))
        edge_set.add((min(b, c), max(b, c)))
        edge_set.add((min(a, c), max(a, c)))
    edges = np.array(list(edge_set), dtype=np.int32)          # (K, 2)

    name = os.path.splitext(os.path.basename(file_path))[0]

    print(f"[ModelLoader] Loaded '{name}': "
          f"{len(vertices)} verts, {len(faces)} faces, {len(edges)} edges")

    return ModelMesh(
        vertices  = vertices,
        faces     = faces,
        edges     = edges,
        name      = name,
        file_path = file_path,
    )


# ────────────────────── OBJ fallback (no trimesh) ──────────────────────

def load_obj_fallback(file_path: str, target_size: float = 8.0) -> ModelMesh:
    """
    Minimal OBJ parser — works without trimesh.
    Only supports triangulated OBJ files (no quads, no materials).
    """
    vertices = []
    faces    = []

    with open(file_path, 'r', errors='replace') as f:
        for line in f:
            line = line.strip()
            if line.startswith('v '):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '):
                parts = line.split()[1:]
                # Support: "f 1 2 3" and "f 1/1/1 2/2/2 3/3/3"
                indices = [int(p.split('/')[0]) - 1 for p in parts]
                if len(indices) == 3:
                    faces.append(indices)
                elif len(indices) == 4:
                    # Triangulate quad
                    faces.append([indices[0], indices[1], indices[2]])
                    faces.append([indices[0], indices[2], indices[3]])

    if not vertices:
        raise ValueError(f"No vertices found in '{file_path}'")

    vertices = np.array(vertices, dtype=np.float32)
    faces    = np.array(faces,    dtype=np.int32) if faces else np.zeros((0, 3), dtype=np.int32)

    # Normalize
    centroid  = vertices.mean(axis=0)
    vertices -= centroid
    extents   = vertices.max(axis=0) - vertices.min(axis=0)
    diagonal  = float(np.linalg.norm(extents))
    if diagonal > 1e-6:
        vertices *= (target_size / diagonal)

    # Edges
    edge_set = set()
    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        edge_set.add((min(a, b), max(a, b)))
        edge_set.add((min(b, c), max(b, c)))
        edge_set.add((min(a, c), max(a, c)))
    edges = np.array(list(edge_set), dtype=np.int32)

    name = os.path.splitext(os.path.basename(file_path))[0]
    print(f"[ModelLoader/OBJ-fallback] Loaded '{name}': "
          f"{len(vertices)} verts, {len(faces)} faces")

    return ModelMesh(vertices=vertices, faces=faces, edges=edges,
                     name=name, file_path=file_path)


def smart_load(file_path: str, target_size: float = 8.0) -> ModelMesh:
    """
    Try trimesh first; fall back to built-in OBJ parser if unavailable.
    """
    ext = os.path.splitext(file_path)[1].lower()
    try:
        return load_model(file_path, target_size)
    except ImportError:
        if ext == '.obj':
            print("[ModelLoader] trimesh not found — using OBJ fallback parser.")
            return load_obj_fallback(file_path, target_size)
        else:
            raise ImportError(
                f"trimesh is required for '{ext}' files.\n"
                "Install with:  pip install trimesh"
            )
