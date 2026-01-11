"""
Depth illusion & projection logic.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from math3d.projection import viewport_transform

def project_3d_to_2d(point_3d, camera, screen_width, screen_height):
    """
    Project a 3D point to 2D screen coordinates using full MVP pipeline.
    
    Args:
        point_3d: 3D point as (x, y, z) tuple or Vector3
        camera: Camera3D instance
        screen_width: Screen width in pixels
        screen_height: Screen height in pixels
    
    Returns:
        (screen_x, screen_y, depth) tuple, or None if point is clipped
        depth is normalized [0, 1] where 0 = near plane, 1 = far plane
    """
    # Get MVP matrices
    view = camera.get_view_matrix()
    aspect = screen_width / screen_height
    proj = camera.get_projection_matrix(aspect)
    
    # Extract coordinates
    if hasattr(point_3d, 'x'):
        x, y, z = point_3d.x, point_3d.y, point_3d.z
    else:
        x, y, z = point_3d
    
    # View transform (world -> camera space)
    view_space = view.transform_point((x, y, z))
    vx, vy, vz, vw = view_space
    
    # Clip points behind camera (negative Z in view space)
    if vz >= 0:
        return None
    
    # Projection transform (camera -> clip space)
    clip_space = proj.transform_point((vx, vy, vz))
    cx, cy, cz, cw = clip_space
    
    # Perspective divide (clip -> NDC)
    if abs(cw) < 1e-6:  # Avoid division by zero
        return None
    
    ndc_x = cx / cw
    ndc_y = cy / cw
    ndc_z = cz / cw
    
    # Frustum culling (optional optimization)
    if abs(ndc_x) > 1.5 or abs(ndc_y) > 1.5:  # Allow some margin
        return None
    
    # Viewport transform (NDC -> screen space)
    screen_x, screen_y = viewport_transform((ndc_x, ndc_y), screen_width, screen_height)
    
    # Normalize depth to [0, 1] range for Z-buffer
    # NDC Z is typically in [-1, 1], map to [0, 1]
    depth = (ndc_z + 1.0) * 0.5
    
    return (screen_x, screen_y, depth)

def is_point_in_frustum(point_3d, camera):
    """
    Check if a 3D point is inside the camera frustum (optional optimization).
    
    Args:
        point_3d: 3D point as (x, y, z) tuple or Vector3
        camera: Camera3D instance
    
    Returns:
        True if point is visible, False otherwise
    """
    # Quick frustum test using projection
    result = project_3d_to_2d(point_3d, camera, 1920, 1080)
    return result is not None
