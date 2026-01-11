"""
Interactive voxel editing with hand gestures.
"""
import numpy as np
import time

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
        self.mode = "IDLE"  # IDLE, DRAW, ERASE, ROTATE, HOLD
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
        
        # Rotation control
        self.rotation_base = None
        self.rotation_start = (0, 0, 0)
        
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
            self.mode = "ROTATE"
        elif gesture == "fist":
            self.mode = "HOLD"
        else:
            self.mode = "IDLE"
    
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
    
    def reset_rotation(self):
        """Reset rotation control."""
        self.rotation_base = None