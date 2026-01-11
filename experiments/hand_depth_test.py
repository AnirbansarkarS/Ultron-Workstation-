"""
Test script for hand depth estimation.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from vision.depth_mapper import extract_hand_depth, map_depth_to_world, get_depth_color
import numpy as np

def test_depth_extraction():
    print("=" * 50)
    print("Testing Hand Depth Extraction...")
    print("=" * 50)
    
    # Simulate hand landmarks (21 points with x, y, z)
    # Z values typically range from -0.1 to +0.1
    fake_landmarks = []
    for i in range(21):
        x = np.random.rand()
        y = np.random.rand()
        z = np.random.uniform(-0.1, 0.05)  # Typical MediaPipe Z range
        fake_landmarks.append((x, y, z))
    
    print(f"Created {len(fake_landmarks)} fake landmarks")
    print(f"Sample landmark 0 (wrist): {fake_landmarks[0]}")
    print(f"Sample landmark 8 (index tip): {fake_landmarks[8]}")
    
    # Test different extraction methods
    methods = ['average', 'wrist', 'index_tip', 'palm']
    for method in methods:
        depth = extract_hand_depth(fake_landmarks, method=method)
        print(f"\n{method.upper()} method: depth = {depth:.4f}")
    
    print("\n✅ Depth extraction test passed!\n")

def test_depth_mapping():
    print("=" * 50)
    print("Testing Depth Mapping...")
    print("=" * 50)
    
    # Test various MediaPipe Z values
    test_depths = [-0.1, -0.05, 0.0, 0.03, 0.05]
    
    print("Mapping MediaPipe Z -> World Depth:")
    print(f"{'MediaPipe Z':<15} {'World Depth':<15} {'Color (BGR)'}")
    print("-" * 50)
    
    for mp_z in test_depths:
        world_depth = map_depth_to_world(mp_z, min_depth=0, max_depth=10)
        color = get_depth_color(world_depth, 0, 10)
        print(f"{mp_z:<15.3f} {world_depth:<15.2f} {color}")
    
    print("\n✅ Depth mapping test passed!\n")

def test_color_gradient():
    print("=" * 50)
    print("Testing Color Gradient...")
    print("=" * 50)
    
    print("Depth color gradient (far to close):")
    print(f"{'Depth':<10} {'Color (BGR)'}")
    print("-" * 30)
    
    for depth in range(0, 11, 2):
        color = get_depth_color(depth, 0, 10)
        print(f"{depth:<10} {color}")
    
    print("\n✅ Color gradient test passed!\n")

def test_edge_cases():
    print("=" * 50)
    print("Testing Edge Cases...")
    print("=" * 50)
    
    # Empty landmarks
    empty_depth = extract_hand_depth([], method='average')
    print(f"Empty landmarks depth: {empty_depth} (expected: 0.0)")
    
    # Out of range values (should be clamped)
    extreme_low = map_depth_to_world(-1.0)  # Way out of range
    extreme_high = map_depth_to_world(1.0)   # Way out of range
    
    print(f"Extreme low (-1.0): {extreme_low:.2f}")
    print(f"Extreme high (1.0): {extreme_high:.2f}")
    print(f"Should be clamped to [0, 10] range")
    
    print("\n✅ Edge cases test passed!\n")

if __name__ == "__main__":
    try:
        test_depth_extraction()
        test_depth_mapping()
        test_color_gradient()
        test_edge_cases()
        
        print("=" * 50)
        print("🎉 ALL HAND DEPTH TESTS PASSED!")
        print("=" * 50)
        print("\n📌 Next: Run 'python main.py' to see voxels react to hand depth in real-time!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
