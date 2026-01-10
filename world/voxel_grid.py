"""
3D voxel data structure.
"""

class VoxelGrid:
    def __init__(self, size=(32, 32, 32)):
        self.size = size
        self.grid = {} # Map (x, y, z) -> color/type

    def set_voxel(self, pos, value):
        self.grid[pos] = value

    def get_voxel(self, pos):
        return self.grid.get(pos)
