import math
import numpy as np
from .vector import Vector3

class Matrix4:
    def __init__(self):
        """Initialize 4x4 matrix with zeros."""
        self.data = [[0]*4 for _ in range(4)]
    
    @staticmethod
    def identity():
        """Create identity matrix."""
        m = Matrix4()
        for i in range(4):
            m.data[i][i] = 1.0
        return m
    
    @staticmethod
    def from_translation(x, y, z):
        """Create translation matrix."""
        m = Matrix4.identity()
        m.data[0][3] = x
        m.data[1][3] = y
        m.data[2][3] = z
        return m
    
    @staticmethod
    def from_scale(sx, sy, sz):
        """Create scaling matrix."""
        m = Matrix4.identity()
        m.data[0][0] = sx
        m.data[1][1] = sy
        m.data[2][2] = sz
        return m
    
    @staticmethod
    def from_rotation_x(angle):
        """Create rotation matrix around X-axis (angle in radians)."""
        m = Matrix4.identity()
        c = math.cos(angle)
        s = math.sin(angle)
        m.data[1][1] = c
        m.data[1][2] = -s
        m.data[2][1] = s
        m.data[2][2] = c
        return m
    
    @staticmethod
    def from_rotation_y(angle):
        """Create rotation matrix around Y-axis (angle in radians)."""
        m = Matrix4.identity()
        c = math.cos(angle)
        s = math.sin(angle)
        m.data[0][0] = c
        m.data[0][2] = s
        m.data[2][0] = -s
        m.data[2][2] = c
        return m
    
    @staticmethod
    def from_rotation_z(angle):
        """Create rotation matrix around Z-axis (angle in radians)."""
        m = Matrix4.identity()
        c = math.cos(angle)
        s = math.sin(angle)
        m.data[0][0] = c
        m.data[0][1] = -s
        m.data[1][0] = s
        m.data[1][1] = c
        return m
    
    @staticmethod
    def from_rotation_xyz(rx, ry, rz):
        """Create rotation matrix from Euler angles (radians)."""
        # Apply rotations in order: Z * Y * X
        mx = Matrix4.from_rotation_x(rx)
        my = Matrix4.from_rotation_y(ry)
        mz = Matrix4.from_rotation_z(rz)
        return mz.multiply(my).multiply(mx)
    
    def multiply(self, other):
        """Multiply this matrix by another matrix."""
        result = Matrix4()
        for i in range(4):
            for j in range(4):
                result.data[i][j] = sum(
                    self.data[i][k] * other.data[k][j]
                    for k in range(4)
                )
        return result
    
    def transform_point(self, point):
        """
        Transform a 3D point (or Vector3) by this matrix.
        Returns (x, y, z, w) tuple.
        """
        if isinstance(point, Vector3):
            x, y, z = point.x, point.y, point.z
        else:
            x, y, z = point
        
        # Apply 4x4 transformation (homogeneous coordinates)
        w = 1.0
        tx = self.data[0][0] * x + self.data[0][1] * y + self.data[0][2] * z + self.data[0][3] * w
        ty = self.data[1][0] * x + self.data[1][1] * y + self.data[1][2] * z + self.data[1][3] * w
        tz = self.data[2][0] * x + self.data[2][1] * y + self.data[2][2] * z + self.data[2][3] * w
        tw = self.data[3][0] * x + self.data[3][1] * y + self.data[3][2] * z + self.data[3][3] * w
        
        return (tx, ty, tz, tw)
        
    def inverse(self):
        """Return the inverse of the matrix."""
        # Convert to numpy array
        arr = np.array(self.data)
        try:
            inv_arr = np.linalg.inv(arr)
        except np.linalg.LinAlgError:
            return None # Singular matrix
            
        # Convert back to Matrix4
        res = Matrix4()
        res.data = inv_arr.tolist()
        return res
    
    def __repr__(self):
        rows = []
        for row in self.data:
            rows.append("[" + ", ".join(f"{v:7.2f}" for v in row) + "]")
        return "Matrix4[\n  " + "\n  ".join(rows) + "\n]"
