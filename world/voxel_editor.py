import numpy as np
import time
import math

class VoxelEditor:
    def __init__(self, voxel_grid, camera):
        """
        Initialize voxel editor.
        
        Args:
            voxel_grid: VoxelGrid instance to edit
            camera: Camera3D instance for coordinate mapping
        """
        self.voxel_grid = voxel_grid
        self.camera = camera
        self.mode = "IDLE"  # IDLE, DRAW, ERASE, ROTATE, HOLD, TRANSLATE, SCALE, ROTATE_OBJ
        self.cursor_pos = (0, 0, 0)
        self.max_voxels = 100  # Performance limit
        
        # Placement control (prevent spam)
        self.last_placed_pos = None
        self.placement_cooldown = 0.3  # seconds between placements
        self.last_placement_time = 0.0
        
        # Erase control (prevent spam)
        self.last_erased_pos = None
        self.erase_cooldown = 0.2
        self.last_erase_time = 0.0
        
        # Color palette for drawing
        self.colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
        ]
        self.current_color_index = 0
        
        # Camera Rotate control
        self.rotation_base = None
        self.rotation_start = (0, 0, 0)
        
        # Object Manipulation Control
        self.manip_start_pos = None
        self.manip_start_scale = 1.0
        self.manip_start_rotation = None
        self.manip_initial_centroids = None
        self.manip_initial_dist = 0.0
        
        # Color cycle control
        self.last_color_cycle_time = 0.0
        self.color_cycle_cooldown = 0.5
    
    def hand_to_world(self, hand_x, hand_y, hand_z, screen_width, screen_height):
        """
        Convert hand position to 3D world coordinates.
        
        Args:
            hand_x, hand_y: Normalized hand position [0, 1]
            hand_z: Hand depth (MediaPipe Z-value)
            screen_width, screen_height: Screen dimensions
        
        Returns:
            (x, y, z) in world space (snapped to grid)
        """
        from vision.depth_mapper import map_depth_to_world
        
        # Map hand X/Y to world X/Y (centered, scaled)
        # Hand [0, 1] → World [-5, 5] for a 10-unit workspace
        world_x = (hand_x - 0.5) * 10
        world_y = -(hand_y - 0.5) * 10  # Flip Y (hand Y goes down)
        
        # Map hand Z to world Z
        world_z = map_depth_to_world(hand_z, min_depth=-3, max_depth=3)
        
        # Snap to integer grid
        grid_x = round(world_x)
        grid_y = round(world_y)
        grid_z = round(world_z)
        
        # Apply inverse object transform to get local grid coordinates
        # We use floating point for the cursor, but keep it relative to grid
        raw_point = (world_x, world_y, world_z)
        
        if self.voxel_grid.transform:
            inv = self.voxel_grid.transform.inverse()
            if inv:
                lx, ly, lz, _ = inv.transform_point(raw_point)
                return (round(lx), round(ly), round(lz))
        
        return (grid_x, grid_y, grid_z)
    
    def update_mode(self, gesture):
        """
        Update edit mode based on current gesture.
        
        Args:
            gesture: Gesture name (e.g., "pointer", "pinch", "open_palm")
        """
        if gesture == "pointer":  # Thumb + index up (gun gesture) - FAR APART
            self.mode = "DRAW"
        elif gesture == "index_point":  # Fallback legacy gesture
            self.mode = "DRAW"
        elif gesture == "pinch":  # Thumb + index CLOSE together
            self.mode = "ERASE"
        elif gesture == "open_palm":
            self.mode = "ROTATE" # Camera Rotate
        if gesture == "pointer":  # Thumb + index up (gun gesture) - FAR APART
            self.mode = "DRAW"
        elif gesture == "index_point":  # Fallback legacy gesture
            self.mode = "DRAW"
        elif gesture == "pinch_hold" or gesture == "GRAB_DRAG":
            self.mode = "GRAB"
        elif gesture == "pinch":  # Thumb + index CLOSE together
            self.mode = "ERASE"
        elif gesture == "open_palm":
            self.mode = "ROTATE_CAM" # Camera Rotate
        elif gesture == "SCALE_OBJECT" or gesture == "ZOOM":
            self.mode = "SCALE"
        elif gesture == "fist":
            self.mode = "HOLD"
        else:
            self.mode = "IDLE"
            # Reset manip state
            self.manip_start_pos = None
            self.manip_initial_dist = 0.0
    
    def place_voxel(self, position):
        """
        Place a voxel at the given position.
        
        Args:
            position: (x, y, z) grid position
        
        Returns:
            True if placed, False if failed
        """
        now = time.time()
        
        # Check cooldown timer
        if now - self.last_placement_time < self.placement_cooldown:
            return False
        
        # Check if same position as last placement
        if position == self.last_placed_pos:
            return False
        
        # Check voxel limit
        if self.voxel_grid.count() >= self.max_voxels:
            return False
        
        # Check if voxel already exists
        if self.voxel_grid.get_voxel(position) is not None:
            return False
        
        # Place voxel with current color
        color = self.colors[self.current_color_index]
        self.voxel_grid.set_voxel(position, color)
        
        # Update tracking
        self.last_placed_pos = position
        self.last_placement_time = now
        
        return True
    
    def erase_voxel(self, position):
        """
        Erase voxel at position.
        
        Args:
            position: (x, y, z) grid position
        
        Returns:
            True if erased, False if no voxel there
        """
        now = time.time()
        
        # Check cooldown
        if now - self.last_erase_time < self.erase_cooldown:
            return False
        
        # Check if same position as last erase
        if position == self.last_erased_pos:
            return False
        
        # Erase voxel
        if self.voxel_grid.get_voxel(position) is not None:
            if position in self.voxel_grid.grid:
                del self.voxel_grid.grid[position]
            
            # Update tracking
            self.last_erased_pos = position
            self.last_erase_time = now
            return True
        
        return False
    
    def find_nearest_voxel(self, position, max_distance=2):
        """
        Find nearest voxel to given position.
        
        Args:
            position: (x, y, z) target position
            max_distance: Maximum distance to search
        
        Returns:
            (x, y, z) of nearest voxel or None
        """
        px, py, pz = position
        nearest = None
        min_dist = float('inf')
        
        for voxel_pos, _ in self.voxel_grid.get_all_voxels():
            vx, vy, vz = voxel_pos
            dist = ((px - vx)**2 + (py - vy)**2 + (pz - vz)**2)**0.5
            
            if dist < min_dist and dist <= max_distance:
                min_dist = dist
                nearest = voxel_pos
        
        return nearest
    
    def cycle_color(self):
        """Cycle to next color in palette."""
        now = time.time()
        
        # Check cooldown
        if now - self.last_color_cycle_time < self.color_cycle_cooldown:
            return
        
        self.current_color_index = (self.current_color_index + 1) % len(self.colors)
        self.last_color_cycle_time = now
    
    def get_current_color(self):
        """Get current drawing color."""
        return self.colors[self.current_color_index]
    
    def start_rotation(self, hand_x, hand_y):
        """
        Start camera rotation control.
        
        Args:
            hand_x, hand_y: Normalized hand position
        """
        self.rotation_base = (hand_x, hand_y)
        self.rotation_start = self.camera.rotation
    
    def update_rotation(self, hand_x, hand_y):
        """
        Update camera rotation based on hand movement.
        
        Args:
            hand_x, hand_y: Current normalized hand position
        """
        if self.rotation_base is None:
            self.start_rotation(hand_x, hand_y)
            return
        
        # Calculate hand movement delta
        base_x, base_y = self.rotation_base
        delta_x = (hand_x - base_x) * 3.0  # Scale for sensitivity
        delta_y = (hand_y - base_y) * 3.0
        
        # Apply rotation
        rx, ry, rz = self.rotation_start
        new_rotation = (rx + delta_y, ry + delta_x, rz)
        self.camera.rotation = new_rotation
    
    
    def update_manipulation(self, landmarks_list, screen_w, screen_h):
        """
        Handle object manipulation based on mode.
        """
        if self.mode == "GRAB":
            # Use Index tip of first hand
            if not landmarks_list: return
            landmarks = landmarks_list[0]
            ind = landmarks[8] # Index tip
            wrist = landmarks[0]
            mid_mcp = landmarks[9]
            
            # --- TRANSLATION ---
            # Convert to world scale (approx)
            wx = (ind[0] - 0.5) * 10
            wy = -(ind[1] - 0.5) * 10
            # Use constant depth for now or map hand Z
            # For natural grab, we should track Z delta too
            from vision.depth_mapper import map_depth_to_world
            wz = map_depth_to_world(ind[2], min_depth=-3, max_depth=3)
            
            current_pos = np.array([wx, wy, wz])
            
            # --- ROTATION (Orientation) ---
            # Vector from Wrist to Middle MCP (Hand direction)
            v_current = np.array([mid_mcp[0] - wrist[0], -(mid_mcp[1] - wrist[1]), mid_mcp[2] - wrist[2]])
            # Normalize
            norm = np.linalg.norm(v_current)
            if norm > 0: v_current /= norm
            else: v_current = np.array([0, 1, 0])

            if self.manip_start_pos is None:
                self.manip_start_pos = current_pos
                angle = math.acos(dot)
                
                # Threshold to avoid jitter
                if abs(angle) > 0.02: # ~1 degree
                    # Axis needs to be normalized
                    axis_norm = np.linalg.norm(axis)
                    if axis_norm > 0.001:
                        axis /= axis_norm
                        # Create rotation matrix
                        from math3d.matrix import Matrix4
                        # Note: We need a generic axis rotation. Matrix4 currently only supports X/Y/Z.
                        # I will implement a quick Rodrigues' rotation or approximation
                        # Or just decomposed Euler.
                        # For now, let's map main components to X/Y/Z rotations.
                        
                        # Simplified: Pitch (Y movement of vector) -> X rotation
                        # Yaw (X movement of vector) -> Y rotation
                        
                        # Better: Update Matrix4 in next step to support Axis-Angle?
                        # Or just use the largest component.
                        if abs(axis[0]) > abs(axis[1]) and abs(axis[0]) > abs(axis[2]):
                            self.voxel_grid.rotate('x', angle * np.sign(axis[0]))
                        elif abs(axis[1]) > abs(axis[2]):
                            self.voxel_grid.rotate('y', angle * np.sign(axis[1]))
                        else:
                            self.voxel_grid.rotate('z', angle * np.sign(axis[2]))
                            
                            # Update reference vector so we don't spin infinitely
                            self.manip_start_rotation = v_current

            # Update start pos
            self.manip_start_pos = current_pos
            
        elif self.mode == "SCALE":
            if len(landmarks_list) < 2: return
            
            # Distance between wrists or index tips
            p1 = np.array(landmarks_list[0][0]) # Wrist 1
            p2 = np.array(landmarks_list[1][0]) # Wrist 2
            
            dist = np.linalg.norm(p1 - p2)
            
            if self.manip_initial_dist == 0.0:
                self.manip_initial_dist = dist
                return
            
            # Scaling factor
            if dist > 0:
                scale_delta = dist / self.manip_initial_dist
                # Dampen
                # scale_factor = 1.0 + (scale_delta - 1.0) * 0.1
                # But matrix multiplication accumulates.
                # If we want 1:1 scaling mapping:
                # We need to apply relative scale.
                
                # Option A: Apply small incremental scale.
                # If dist > initial, scale > 1.
                # We need to reset initial dist every frame? No, that drifts.
                # Better: compare to previous frame's dist?
                pass
            
            # Incremental approach
            if self.manip_initial_dist > 0:
                ratio = dist / self.manip_initial_dist
                # Apply if significant change
                if abs(ratio - 1.0) > 0.01:
                    self.voxel_grid.scale(ratio)
                    # Reset baseline to avoid exponential growth
                    self.manip_initial_dist = dist

    def reset_rotation(self):
        """Reset rotation control."""
        self.rotation_base = None