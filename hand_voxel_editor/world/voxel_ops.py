"""
Add / remove / color voxels.
"""

def add_voxel(grid, pos, color):
    grid.set_voxel(pos, color)

def remove_voxel(grid, pos):
    if pos in grid.grid:
        del grid.grid[pos]
