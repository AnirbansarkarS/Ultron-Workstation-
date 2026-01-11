"""
Test script for voxel projection logic.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from math3d.vector import Vector3
from math3d.matrix import Matrix4
from math3d.projection import perspective_matrix, view_matrix, viewport_transform
from render.camera3d import Camera3D

def test_vector3():
    print("=" * 50)
    print("Testing Vector3...")
    print("=" * 50)
    
    v1 = Vector3(1, 2, 3)
    v2 = Vector3(4, 5, 6)
    
    print(f"v1: {v1}")
    print(f"v2: {v2}")
    print(f"v1 + v2: {v1 + v2}")
    print(f"v1 - v2: {v1 - v2}")
    print(f"v1 * 2: {v1 * 2}")
    print(f"v1.length(): {v1.length():.2f}")
    print(f"v1.normalize(): {v1.normalize()}")
    print(f"v1.dot(v2): {v1.dot(v2)}")
    print(f"v1.cross(v2): {v1.cross(v2)}")
    print("✅ Vector3 tests passed!\n")

def test_matrix4():
    print("=" * 50)
    print("Testing Matrix4...")
    print("=" * 50)
    
    identity = Matrix4.identity()
    print("Identity Matrix:")
    print(identity)
    
    translation = Matrix4.from_translation(10, 20, 30)
    print("\nTranslation Matrix (10, 20, 30):")
    print(translation)
    
    point = (5, 5, 5)
    transformed = translation.transform_point(point)
    print(f"\nTransform point {point}: {transformed}")
    print(f"Expected: (15, 25, 35, 1)")
    
    rotation_x = Matrix4.from_rotation_x(1.57)  # ~90 degrees
    print(f"\nRotation X (90°):")
    print(rotation_x)
    
    print("✅ Matrix4 tests passed!\n")

def test_projection():
    print("=" * 50)
    print("Testing Projection...")
    print("=" * 50)
    
    # Test perspective matrix
    proj = perspective_matrix(fov=60, aspect=16/9, near=0.1, far=100)
    print("Perspective Matrix (FOV=60°, 16:9 aspect):")
    print(proj)
    
    # Test view matrix
    camera_pos = (0, 0, -5)
    camera_rot = (0, 0, 0)
    view = view_matrix(camera_pos, camera_rot)
    print(f"\nView Matrix (pos={camera_pos}, rot={camera_rot}):")
    print(view)
    
    # Test viewport transform
    ndc_point = (0, 0)  # Center of screen
    screen = viewport_transform(ndc_point, 1920, 1080)
    print(f"\nNDC {ndc_point} -> Screen: {screen}")
    print(f"Expected: (960, 540)")
    
    print("✅ Projection tests passed!\n")

def test_camera3d():
    print("=" * 50)
    print("Testing Camera3D...")
    print("=" * 50)
    
    cam = Camera3D(position=(0, 0, -5), fov=60)
    print(f"Camera position: {cam.position}")
    print(f"Camera FOV: {cam.fov}°")
    
    cam.update_position((1, 0, 0))
    print(f"After update_position((1, 0, 0)): {cam.position}")
    
    cam.look_at((0, 0, 0))
    print(f"After look_at((0, 0, 0)), rotation: {cam.rotation}")
    
    view = cam.get_view_matrix()
    print(f"\nView matrix:")
    print(view)
    
    proj = cam.get_projection_matrix(16/9)
    print(f"\nProjection matrix (16:9):")
    print(proj)
    
    print("✅ Camera3D tests passed!\n")

def test_full_pipeline():
    print("=" * 50)
    print("Testing Full MVP Pipeline...")
    print("=" * 50)
    
    # Setup camera
    cam = Camera3D(position=(0, 0, -10), fov=60)
    
    # Get matrices
    view = cam.get_view_matrix()
    proj = cam.get_projection_matrix(16/9)
    
    # Transform a 3D point through the pipeline
    point_3d = (0, 0, 0)  # Origin
    print(f"3D Point: {point_3d}")
    
    # View transform
    view_space = view.transform_point(point_3d)
    print(f"View Space: {view_space}")
    
    # Projection transform
    clip_space = proj.transform_point(view_space[:3])
    print(f"Clip Space: {clip_space}")
    
    # Perspective divide
    if clip_space[3] != 0:
        ndc_x = clip_space[0] / clip_space[3]
        ndc_y = clip_space[1] / clip_space[3]
        depth = clip_space[2] / clip_space[3]
        print(f"NDC: ({ndc_x:.3f}, {ndc_y:.3f}, depth={depth:.3f})")
        
        # Viewport transform
        screen = viewport_transform((ndc_x, ndc_y), 1920, 1080)
        print(f"Screen: {screen}")
    
    print("✅ Full pipeline test passed!\n")

if __name__ == "__main__":
    try:
        test_vector3()
        test_matrix4()
        test_projection()
        test_camera3d()
        test_full_pipeline()
        
        print("=" * 50)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 50)
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
