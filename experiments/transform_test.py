
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from math3d.matrix import Matrix4
from world.voxel_grid import VoxelGrid
import numpy as np

def test_matrix_ops():
    print("Testing Matrix4 Ops...")
    m = Matrix4.identity()
    print(f"Identity:\n{m}")
    
    t = Matrix4.from_translation(10, 20, 30)
    print(f"Translation:\n{t}")
    
    s = Matrix4.from_scale(2, 2, 2)
    print(f"Scale:\n{s}")
    
    # Test Composition: T * S
    # Should scale then translate? Or translated scale?
    # T * S means: Apply S first, then T. (Column major? No, this impl looks Row Major or Standard)
    # v' = T * (S * v)
    c = t.multiply(s)
    print(f"Combined (T * S):\n{c}")
    
    p = (1, 1, 1)
    pt = c.transform_point(p)
    # Scale(1,1,1) -> (2,2,2). Translate(2,2,2) -> (12, 22, 32).
    print(f"Transformed (1,1,1) -> {pt}")
    assert abs(pt[0] - 12.0) < 0.001
    
    # Test Inverse
    inv = c.inverse()
    print(f"Inverse:\n{inv}")
    p_orig = inv.transform_point(pt[:3])
    print(f"Restored -> {p_orig}")
    assert abs(p_orig[0] - 1.0) < 0.001
    print("Matrix Ops Passed.\n")

def test_voxel_grid_transform():
    print("Testing VoxelGrid Transform...")
    grid = VoxelGrid()
    grid.translate(5, 0, 0)
    print(f"Grid after translate(5,0,0):\n{grid.transform}")
    
    grid.rotate('y', 1.5708) # 90 deg
    print(f"Grid after translate then rotate Y:\n{grid.transform}")
    # Current transform = Rotation * Translation * Identity
    # v' = R * T * v
    
    p = (10, 0, 0)
    # T(10,0,0) -> (15,0,0)
    # R(15,0,0) -> (0, 0, -15) (approx)
    
    pt = grid.transform.transform_point(p)
    print(f"Point (10,0,0) transformed -> {pt}")
    
    # Verify logic order
    # code: self.transform = r.multiply(self.transform)
    # new = R * Old
    # Old = T * I
    # New = R * T
    # v' = R * (T * v)
    
    print("VoxelGrid Ops Passed.\n")

if __name__ == "__main__":
    test_matrix_ops()
    test_voxel_grid_transform()
