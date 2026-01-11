"""
Projection math helpers.
"""
import math
from .matrix import Matrix4
from .vector import Vector3

def perspective_matrix(fov, aspect, near, far):
    """
    Create perspective projection matrix.
    
    Args:
        fov: Field of view in degrees
        aspect: Aspect ratio (width / height)
        near: Near clipping plane distance
        far: Far clipping plane distance
    
    Returns:
        Matrix4 perspective projection matrix
    """
    m = Matrix4()
    
    # Convert FOV to radians and calculate scale
    fov_rad = math.radians(fov)
    tan_half_fov = math.tan(fov_rad / 2.0)
    
    # Perspective projection matrix (OpenGL style)
    m.data[0][0] = 1.0 / (aspect * tan_half_fov)
    m.data[1][1] = 1.0 / tan_half_fov
    m.data[2][2] = -(far + near) / (far - near)
    m.data[2][3] = -(2.0 * far * near) / (far - near)
    m.data[3][2] = -1.0
    m.data[3][3] = 0.0
    
    return m

def view_matrix(position, rotation):
    """
    Create view matrix from camera position and rotation.
    
    Args:
        position: Camera position as (x, y, z) tuple or Vector3
        rotation: Camera rotation as (rx, ry, rz) tuple in radians
    
    Returns:
        Matrix4 view matrix
    """
    if isinstance(position, Vector3):
        pos = position
    else:
        pos = Vector3(*position)
    
    rx, ry, rz = rotation
    
    # Create rotation matrix (inverse of camera rotation)
    rot = Matrix4.from_rotation_xyz(-rx, -ry, -rz)
    
    # Create translation matrix (inverse of camera position)
    trans = Matrix4.from_translation(-pos.x, -pos.y, -pos.z)
    
    # View matrix = rotation * translation
    return rot.multiply(trans)

def viewport_transform(ndc_point, width, height):
    """
    Transform normalized device coordinates [-1, 1] to screen coordinates.
    
    Args:
        ndc_point: (x, y) in NDC space
        width: Screen width in pixels
        height: Screen height in pixels
    
    Returns:
        (x, y) in screen pixel coordinates
    """
    x_ndc, y_ndc = ndc_point
    
    # NDC to screen space
    screen_x = (x_ndc + 1.0) * 0.5 * width
    screen_y = (1.0 - y_ndc) * 0.5 * height  # Flip Y (screen Y goes down)
    
    return (int(screen_x), int(screen_y))
