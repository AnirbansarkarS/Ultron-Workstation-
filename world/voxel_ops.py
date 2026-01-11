"""
Add / remove / color voxels.
"""
import cv2
import numpy as np

def add_voxel(grid, pos, color):
    grid.set_voxel(pos, color)

def remove_voxel(grid, pos):
    if pos in grid.grid:
        del grid.grid[pos]

def get_voxel_cube_vertices(position, size=1.0):
    """
    Get 8 corner vertices of a voxel cube.
    
    Args:
        position: Center position (x, y, z)
        size: Size of the voxel cube
    
    Returns:
        List of 8 vertices as (x, y, z) tuples
    """
    x, y, z = position
    half = size / 2.0
    
    # 8 corners of cube
    vertices = [
        (x - half, y - half, z - half),  # 0: bottom-left-back
        (x + half, y - half, z - half),  # 1: bottom-right-back
        (x + half, y + half, z - half),  # 2: top-right-back
        (x - half, y + half, z - half),  # 3: top-left-back
        (x - half, y - half, z + half),  # 4: bottom-left-front
        (x + half, y - half, z + half),  # 5: bottom-right-front
        (x + half, y + half, z + half),  # 6: top-right-front
        (x - half, y + half, z + half),  # 7: top-left-front
    ]
    
    return vertices

def get_voxel_faces():
    """
    Get face indices for drawing a cube.
    Each face is defined by 4 vertex indices.
    
    Returns:
        List of 6 faces, each with 4 vertex indices
    """
    return [
        [0, 1, 2, 3],  # Back face
        [4, 5, 6, 7],  # Front face
        [0, 1, 5, 4],  # Bottom face
        [2, 3, 7, 6],  # Top face
        [0, 3, 7, 4],  # Left face
        [1, 2, 6, 5],  # Right face
    ]

def sort_voxels_by_depth(voxels, camera):
    """
    Sort voxels by distance from camera (painter's algorithm).
    Farther voxels come first (drawn first).
    
    Args:
        voxels: List of (position, color) tuples
        camera: Camera3D instance
    
    Returns:
        Sorted list of voxels (farthest to nearest)
    """
    from math3d.vector import Vector3
    
    cam_pos = camera.position
    
    def distance_squared(voxel):
        pos, _ = voxel
        dx = pos[0] - cam_pos.x
        dy = pos[1] - cam_pos.y
        dz = pos[2] - cam_pos.z
        return dx*dx + dy*dy + dz*dz
    
    # Sort descending (farthest first)
    return sorted(voxels, key=distance_squared, reverse=True)

def draw_voxel(frame, voxel_vertices_2d, color, zbuffer=None, alpha=0.7):
    """
    Draw a single voxel cube on frame (OPTIMIZED VERSION).
    
    Args:
        frame: OpenCV image to draw on
        voxel_vertices_2d: List of projected 2D vertices [(x, y, depth), ...]
        color: RGB color tuple
        zbuffer: Optional ZBuffer for depth testing
        alpha: Transparency (0-1) - ignored for performance
    
    Returns:
        True if voxel was drawn, False if culled
    """
    # Filter out None vertices (clipped)
    valid_vertices = [v for v in voxel_vertices_2d if v is not None]
    
    if len(valid_vertices) < 4:  # Need at least 4 points for a face
        return False
    
    # Get face definitions
    faces = get_voxel_faces()
    
    # OPTIMIZATION: Only draw front 3 faces for speed
    # Front face (4), top face (3), right face (5)
    priority_faces = [faces[1], faces[3], faces[5]]
    
    # Draw visible faces
    drawn = False
    for face_indices in priority_faces:
        # Get face vertices
        face_verts = [voxel_vertices_2d[i] for i in face_indices]
        
        # Skip if any vertex is clipped
        if None in face_verts:
            continue
        
        # Extract 2D points and average depth
        points_2d = np.array([(int(v[0]), int(v[1])) for v in face_verts], dtype=np.int32)
        avg_depth = sum(v[2] for v in face_verts) / len(face_verts)
        
        # Get face center for Z-buffer test
        center_x = int(np.mean(points_2d[:, 0]))
        center_y = int(np.mean(points_2d[:, 1]))
        
        # Z-buffer test (if enabled)
        if zbuffer is not None:
            if not zbuffer.test_and_set(center_x, center_y, avg_depth):
                continue  # Failed depth test
        
        # OPTIMIZED: Draw solid polygon directly (no transparency blending)
        cv2.fillPoly(frame, [points_2d], color)
        
        # OPTIMIZED: Simpler edge drawing
        cv2.polylines(frame, [points_2d], True, (255, 255, 255), 1)
        
        drawn = True
    
    return drawn
