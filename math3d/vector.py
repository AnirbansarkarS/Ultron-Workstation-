"""
Vector operations for 3D space.
"""
import math

class Vector3:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z
    
    def __add__(self, other):
        """Add two vectors."""
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        """Subtract two vectors."""
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar):
        """Multiply vector by scalar."""
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __rmul__(self, scalar):
        """Right multiply (scalar * vector)."""
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar):
        """Divide vector by scalar."""
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)
    
    def __neg__(self):
        """Negate vector."""
        return Vector3(-self.x, -self.y, -self.z)
    
    def length(self):
        """Calculate vector magnitude."""
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
    
    def length_squared(self):
        """Calculate squared magnitude (faster, no sqrt)."""
        return self.x * self.x + self.y * self.y + self.z * self.z
    
    def normalize(self):
        """Return normalized vector (length = 1)."""
        length = self.length()
        if length > 0:
            return self / length
        return Vector3(0, 0, 0)
    
    def dot(self, other):
        """Dot product with another vector."""
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other):
        """Cross product with another vector."""
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def to_tuple(self):
        """Convert to tuple (x, y, z)."""
        return (self.x, self.y, self.z)
    
    def __repr__(self):
        return f"Vector3({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"
