"""
3D voxel data structure.
"""

import sys
import os
# Ensure we can import from math3d if running as script or module
if __name__ != "__main__":
    from math3d.matrix import Matrix4
else:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from math3d.matrix import Matrix4

class VoxelGrid:
    def __init__(self, size=(32, 32, 32), create_sample=False):
        self.size = size
        self.grid = {} # Map (x, y, z) -> color/type
        self.transform = Matrix4.identity()
        
        if create_sample:
            self._create_sample_voxels()
    
    def _create_sample_voxels(self):
        """Create a sample floating voxel structure for testing."""
        # Create a smaller 3x3x3 cube for better performance
        import random
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
        ]
        
        # Centered floating cube - SMALLER for performance
        center_x, center_y, center_z = 0, 0, 0
        for x in range(-1, 2):  # -1, 0, 1 (3x3x3 instead of 5x5x5)
            for y in range(-1, 2):
                for z in range(-1, 2):
                    # Only create outer shell (hollow cube)
                    if abs(x) == 1 or abs(y) == 1 or abs(z) == 1:
                        pos = (center_x + x, center_y + y, center_z + z)
                        color = random.choice(colors)
                        self.set_voxel(pos, color)

    def set_voxel(self, pos, value):
        self.grid[pos] = value

    def get_voxel(self, pos):
        return self.grid.get(pos)
    
    def get_all_voxels(self):
        """
        Get iterator of all voxels.
        
        Yields:
            (position, color) tuples where position is (x, y, z)
        """
        for pos, color in self.grid.items():
            yield (pos, color)
    
    def get_bounds(self):
        """
        Get bounding box of all voxels.
        
        Returns:
            ((min_x, min_y, min_z), (max_x, max_y, max_z)) or None if empty
        """
        if not self.grid:
            return None
        
        positions = list(self.grid.keys())
        min_x = min(p[0] for p in positions)
        max_x = max(p[0] for p in positions)
        min_y = min(p[1] for p in positions)
        max_y = max(p[1] for p in positions)
        min_z = min(p[2] for p in positions)
        max_z = max(p[2] for p in positions)
        
        return ((min_x, min_y, min_z), (max_x, max_y, max_z))
    
    def count(self):
        """Return number of voxels."""
        return len(self.grid)

    def translate(self, x, y, z):
        """Apply translation to the grid."""
        t = Matrix4.from_translation(x, y, z)
        # Apply T * current (Global transform)
        self.transform = t.multiply(self.transform)
        
    def rotate(self, axis, angle):
        """Apply rotation to the grid (radians)."""
        if axis.lower() == 'x':
            r = Matrix4.from_rotation_x(angle)
        elif axis.lower() == 'y':
            r = Matrix4.from_rotation_y(angle)
        elif axis.lower() == 'z':
            r = Matrix4.from_rotation_z(angle)
        else:
            return
        
        # Apply R * current (Global rotation)
        self.transform = r.multiply(self.transform)
        
    def scale(self, factor):
        """Apply uniform scaling."""
        s = Matrix4.from_scale(factor, factor, factor)
        self.transform = s.multiply(self.transform)
        
    def set_transform(self, matrix):
        self.transform = matrix
