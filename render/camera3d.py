"""
Virtual camera math.
"""
import math
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from math3d.vector import Vector3
from math3d.projection import perspective_matrix, view_matrix

class Camera3D:
    def __init__(self, position=(0, 0, -5), rotation=(0, 0, 0), fov=60, near=0.1, far=100):
        """
        Initialize 3D camera.
        
        Args:
            position: Camera position (x, y, z)
            rotation: Camera rotation in radians (rx, ry, rz)
            fov: Field of view in degrees
            near: Near clipping plane distance
            far: Far clipping plane distance
        """
        self.position = Vector3(*position) if not isinstance(position, Vector3) else position
        self.rotation = rotation  # (rx, ry, rz) in radians
        self.fov = fov
        self.near = near
        self.far = far
    
    def set_position(self, x, y, z):
        """Update camera position."""
        self.position = Vector3(x, y, z)
    
    def set_rotation(self, rx, ry, rz):
        """Update camera rotation (radians)."""
        self.rotation = (rx, ry, rz)
    
    def update_position(self, delta):
        """Move camera by delta vector."""
        if isinstance(delta, Vector3):
            self.position = self.position + delta
        else:
            self.position = self.position + Vector3(*delta)
    
    def update_rotation(self, drx, dry, drz):
        """Rotate camera by delta angles (radians)."""
        rx, ry, rz = self.rotation
        self.rotation = (rx + drx, ry + dry, rz + drz)
    
    def look_at(self, target):
        """
        Point camera at target position.
        
        Args:
            target: Target position as Vector3 or (x, y, z)
        """
        if not isinstance(target, Vector3):
            target = Vector3(*target)
        
        # Calculate direction vector
        direction = (target - self.position).normalize()
        
        # Calculate pitch (rotation around X)
        pitch = math.asin(-direction.y)
        
        # Calculate yaw (rotation around Y)
        yaw = math.atan2(direction.x, direction.z)
        
        self.rotation = (pitch, yaw, 0)
    
    def get_view_matrix(self):
        """Get view matrix for current camera transform."""
        return view_matrix(self.position, self.rotation)
    
    def get_projection_matrix(self, aspect_ratio):
        """
        Get projection matrix for current camera settings.
        
        Args:
            aspect_ratio: Screen width / height
        """
        return perspective_matrix(self.fov, aspect_ratio, self.near, self.far)
